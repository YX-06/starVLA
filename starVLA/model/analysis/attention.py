"""Observe actual SDPA inputs without replacing the model's attention processor.

Only explicitly scoped action-expert cross-attention is observed. The original
SDPA call still produces the action output; diagnostic probabilities use FP32.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import subprocess
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.overrides import TorchFunctionMode


def attention_probabilities(q, k, mask=None, *, scale=None, is_causal=False, query_indices=None):
    """Return head-averaged probabilities, normalized over ALL keys.

    q/k: [batch, heads, sequence, head_dim]. Boolean masks follow SDPA
    semantics (True means visible); floating masks are additive. GQA keys
    are expanded exactly as in SDPA. Slice queries before allocating logits.
    """
    if q.ndim != 4 or k.ndim != 4 or q.shape[1] % k.shape[1]:
        raise ValueError("Expected 4D Q/K with divisible query/KV head counts")
    full_query_length = q.shape[-2]
    indices = torch.arange(full_query_length, device=q.device) if query_indices is None else query_indices
    indices = torch.as_tensor(indices, device=q.device, dtype=torch.long)
    q = q.index_select(-2, indices).float()
    k = k.float().repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    logits = (q @ k.transpose(-2, -1)) * (scale if scale is not None else q.shape[-1] ** -0.5)
    if mask is not None:
        if mask.ndim >= 2 and mask.shape[-2] == full_query_length:
            mask = mask.index_select(-2, indices)
        logits = logits.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else logits + mask
    if is_causal:
        visible = torch.arange(k.shape[-2], device=q.device)[None, :] <= indices[:, None]
        logits = logits.masked_fill(~visible, float("-inf"))
    # SDPA defines all-masked rows as zero, including additive -inf masks.
    return logits.softmax(-1).nan_to_num().mean(1)


def image_token_maps(input_ids, image_grid_thw, merge_size, image_token_id, camera_names):
    """Map Qwen's merged image-token positions to row-major camera grids."""
    ids = torch.as_tensor(input_ids).cpu()
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError("Attention export requires batch_size=1")
    grids = torch.as_tensor(image_grid_thw).cpu().tolist()
    if len(grids) != len(camera_names) or len(set(camera_names)) != len(camera_names):
        raise ValueError("Supply one unique camera name per input image")
    positions = (ids[0] == image_token_id).nonzero().flatten().tolist()
    result, cursor = [], 0
    for name, (time, height, width) in zip(camera_names, grids, strict=False):
        if time != 1 or height % merge_size or width % merge_size:
            raise ValueError("Only still images with spatially divisible grids are supported")
        height, width = height // merge_size, width // merge_size
        count = height * width
        selected = positions[cursor : cursor + count]
        if len(selected) != count or selected != list(range(selected[0], selected[0] + count)):
            raise ValueError("Image token span does not match processor grid")
        result.append({"name": name, "grid_hw": [height, width], "token_indices": selected})
        cursor += count
    if cursor != len(positions):
        raise ValueError("Unmapped image tokens")
    return result


class _ObserveSDPA(TorchFunctionMode):
    def __init__(self, session):
        self.session = session

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is F.scaled_dot_product_attention and self.session.current_layer is not None:
            self.session.observe(args, kwargs)
        return func(*args, **kwargs)


class AttentionRecorder:
    """Server-owned configuration; each request gets a fresh bounded session."""

    def __init__(self, config, provenance=None):
        if isinstance(config, (str, Path)):
            config = json.loads(Path(config).read_text())
        self.config = dict(config)
        unknown = set(config) - {"output_dir", "layers", "denoise_steps", "action_queries", "interval"}
        if unknown:
            raise ValueError(f"Unknown attention settings: {sorted(unknown)}")
        self.output_dir = Path(config["output_dir"]).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if int(config.get("interval", 1)) < 1:
            raise ValueError("interval must be positive")
        for key in ("layers", "denoise_steps", "action_queries"):
            values = config.get(key)
            if values is not None and (not values or len(set(values)) != len(values) or min(values) < 0):
                raise ValueError(f"{key} must be null or a nonempty list of unique nonnegative indices")
        self.requests = 0
        self.provenance = dict(provenance or {})
        self.provenance["dependencies"] = {}
        for dist in ("torch", "numpy", "pillow", "transformers", "diffusers"):
            try:
                self.provenance["dependencies"][dist] = importlib.metadata.version(dist)
            except importlib.metadata.PackageNotFoundError:
                pass
        self.provenance["code_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        self.provenance["code_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
        diff = subprocess.check_output(["git", "diff", "HEAD"])
        self.provenance["tracked_diff_sha256"] = hashlib.sha256(diff).hexdigest()

    def session(self, framework, examples, trace_context=None):
        index = self.requests
        self.requests += 1
        if index % int(self.config.get("interval", 1)):
            return contextlib.nullcontext(None)
        return AttentionSession(self, framework, examples, trace_context or {}, index)


class AttentionSession(contextlib.ExitStack):
    def __init__(self, recorder, framework, examples, context, request_index):
        super().__init__()
        if len(examples) != 1:
            raise ValueError("Attention export requires batch_size=1")
        self.recorder, self.framework = recorder, framework
        self.examples, self.context = examples, dict(context)
        self.trace_id = uuid.uuid4().hex
        self.request_index = request_index
        self.current_layer = None
        self.calls = defaultdict(int)
        self.records = {}
        self.timesteps = {}
        self.maps, self.images, self.input_metadata = [], [], {}
        self.timestep = None
        action_head = getattr(framework, "action_model", framework)
        self.horizon = int(action_head.action_horizon)
        self.query_indices = recorder.config.get("action_queries") or list(range(self.horizon))
        if max(self.query_indices) >= self.horizon:
            raise ValueError("action_queries exceeds checkpoint action horizon")

    def __enter__(self):
        super().__enter__()
        try:
            interface = self.framework.qwen_vl_interface
            original = interface.build_qwenvl_inputs

            def build_inputs(*args, **kwargs):
                result = original(*args, **kwargs)
                images = kwargs.get("images", args[0] if args else None)
                instructions = kwargs.get("instructions", args[1] if len(args) > 1 else None)
                self.capture_inputs(interface, result, images, instructions)
                return result

            self._patch(interface, "build_qwenvl_inputs", build_inputs)
            if hasattr(self.framework, "expert_layers"):
                # Published QwenPI_v4 has a separate cross-attention per expert
                # block. Self-attention calls must remain outside the scope.
                modules = [(i, layer.cross_attn) for i, layer in enumerate(self.framework.expert_layers)]
                embed_timestep = self.framework._embed_timestep

                def embed_and_capture(t):
                    self.timestep = t.detach().cpu().tolist()
                    return embed_timestep(t)

                self._patch(self.framework, "_embed_timestep", embed_and_capture)
            else:
                head = self.framework.action_model
                modules = [(i, layer.attn1) for i, layer in enumerate(head.model.transformer_blocks)]
                self.callback(head.action_encoder.register_forward_pre_hook(self._time_hook).remove)
            for index, module in modules:

                def before(mod, args, kwargs, layer=index):
                    cross = hasattr(mod, "q_proj") or kwargs.get("encoder_hidden_states") is not None
                    self.current_layer = layer if cross else None

                def after(mod, args, output):
                    self.current_layer = None

                self.callback(module.register_forward_pre_hook(before, with_kwargs=True).remove)
                self.callback(module.register_forward_hook(after, always_call=True).remove)
            self.enter_context(_ObserveSDPA(self))
            return self
        except BaseException:
            self.close()
            raise

    def _patch(self, obj, name, value):
        # Restore the instance dictionary exactly, including descriptor binding.
        previous = obj.__dict__.get(name)
        existed = name in obj.__dict__
        setattr(obj, name, value)
        self.callback(setattr if existed else delattr, obj, name, *([previous] if existed else []))

    def _time_hook(self, module, args):
        self.timestep = args[1].detach().cpu().tolist()

    def capture_inputs(self, interface, inputs, images, instructions):
        if self.maps:
            raise ValueError("Multiple prefix encodings in one request are not supported")
        count = len(images[0])
        names = self.context.get("camera_names", [f"camera_{i}" for i in range(count)])
        processor = interface.processor.image_processor
        merge = int(processor.merge_size)
        token = int(interface.model.config.image_token_id)
        self.maps = image_token_maps(inputs["input_ids"], inputs["image_grid_thw"], merge, token, names)
        self.images = [
            image.copy() if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image)) for image in images[0]
        ]
        patch_size = int(processor.patch_size)
        for mapping, image in zip(self.maps, self.images, strict=False):
            mapping["input_size_wh"] = list(image.size)
            mapping["processor_size_wh"] = [size * merge * patch_size for size in mapping["grid_hw"][::-1]]
            mapping["transform"] = {
                "resize_from_wh": list(image.size),
                "resize_to_wh": mapping["processor_size_wh"],
                "padding": [0, 0, 0, 0],
                "rotation_degrees": 0,
            }
        for index, mapping in enumerate(self.maps):
            source = self.examples[0]["image"][index]
            source_wh = list(source.size) if isinstance(source, Image.Image) else list(np.asarray(source).shape[1::-1])
            mapping["framework_transform"] = {
                "resize_from_wh": source_wh,
                "resize_to_wh": mapping["input_size_wh"],
                "padding": [0, 0, 0, 0],
                "rotation_degrees": 0,
            }
        self.input_metadata = {
            "input_ids": inputs["input_ids"].detach().cpu().tolist(),
            "image_grid_thw": inputs["image_grid_thw"].detach().cpu().tolist(),
            "spatial_merge_size": merge,
            "instructions": instructions,
        }

    @torch.no_grad()
    def observe(self, args, kwargs):
        layer = self.current_layer
        step = self.calls[layer]
        self.calls[layer] += 1
        cfg = self.recorder.config
        if (cfg.get("layers") is not None and layer not in cfg["layers"]) or (
            cfg.get("denoise_steps") is not None and step not in cfg["denoise_steps"]
        ):
            return
        if not self.maps:
            raise RuntimeError("No processor token mapping available")
        q, k = args[:2]
        indices = [q.shape[-2] - self.horizon + i for i in self.query_indices]
        if min(indices) < 0:
            raise ValueError("Attention query sequence is shorter than action horizon")
        mask = kwargs.get("attn_mask", args[3] if len(args) > 3 else None)
        causal = kwargs.get("is_causal", args[5] if len(args) > 5 else False)
        # Explicitly disable autocast so the diagnostic matmul remains FP32.
        with torch.autocast(q.device.type, enabled=False):
            probabilities = attention_probabilities(
                q, k, mask, scale=kwargs.get("scale"), is_causal=causal, query_indices=indices
            )
        for camera, mapping in enumerate(self.maps):
            weights = probabilities[0, :, mapping["token_indices"]]
            self.records[step, layer, camera] = weights.reshape(len(indices), *mapping["grid_hw"]).cpu().numpy()
        self.timesteps[step] = self.timestep

    def save(self):
        if not self.records:
            raise RuntimeError("No supported cross-attention calls recorded; check processor and selection")
        steps = sorted({key[0] for key in self.records})
        layers = sorted({key[1] for key in self.records})
        for key, selected in (("layers", layers), ("denoise_steps", steps)):
            if self.recorder.config.get(key) is not None and set(selected) != set(self.recorder.config[key]):
                raise ValueError(f"Requested {key} did not all execute")
        root = self.recorder.output_dir / self.trace_id
        root.mkdir(exist_ok=False)
        arrays = {}
        for camera, mapping in enumerate(self.maps):
            array = np.stack([np.stack([self.records[step, layer, camera] for layer in layers]) for step in steps])
            arrays[f"camera_{camera}"] = array
            mapping["array_key"] = f"camera_{camera}"
            mapping["attention_mass_mean"] = float(array.sum(axis=(-1, -2)).mean())
            mapping["image_file"] = f"camera_{camera}.png"
            self.images[camera].save(root / mapping["image_file"])
        np.savez_compressed(root / "attention.npz", **arrays)
        metadata = {
            "schema_version": 1,
            "trace_id": self.trace_id,
            "request_index": self.request_index,
            "trace_context": self.context,
            "provenance": self.recorder.provenance,
            "framework": type(self.framework).__name__,
            "torch_seed": torch.initial_seed(),
            "cuda_seed": torch.cuda.initial_seed() if torch.cuda.is_available() else None,
            "denoise_steps": steps,
            "timesteps": self.timesteps,
            "layers": layers,
            "action_queries": self.query_indices,
            "head_reduction": "mean",
            "normalization": "softmax over full key sequence; image mass preserved",
            "axes": ["denoise_step", "layer", "action_query", "grid_h", "grid_w"],
            "cameras": self.maps,
            **self.input_metadata,
        }
        (root / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        return self.trace_id

    def __exit__(self, *exc):
        try:
            return super().__exit__(*exc)
        finally:
            self.current_layer = None
            self.records.clear()
            self.images.clear()
