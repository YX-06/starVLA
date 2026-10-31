"""Focused CPU/GPU checks: python -m pytest examples/attention_analysis -q."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from examples.attention_analysis.client_trace import ChunkTrace
from examples.attention_analysis.render import render_trace, restore_map
from starVLA.model.analysis.attention import AttentionRecorder, attention_probabilities, image_token_maps


@pytest.mark.parametrize("mask_type", ["none", "bool", "additive", "all_masked", "causal"])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_probabilities(mask_type, kv_heads, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device is unavailable")
    torch.manual_seed(42)
    q, k = torch.randn(1, 4, 7, 8, device=device), torch.randn(1, kv_heads, 11, 8, device=device)
    mask = None
    if mask_type in ("bool", "additive", "all_masked"):
        mask = torch.rand(1, 4, 7, 11, device=device) > 0.3
        if mask_type == "all_masked":
            mask[:] = False
        if mask_type == "additive":
            mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(~mask, float("-inf"))
    indices = [6, 3, 5]
    observed = attention_probabilities(q, k, mask, scale=0.17, is_causal=mask_type == "causal", query_indices=indices)
    # Identity values expose the full reference SDPA probability matrix.
    values = torch.eye(11, device=device).expand(1, 4, 11, 11)
    expected = F.scaled_dot_product_attention(
        q, k.repeat_interleave(4 // kv_heads, 1), values, attn_mask=mask, scale=0.17, is_causal=mask_type == "causal"
    )
    torch.testing.assert_close(observed, expected.mean(1)[:, indices], atol=1e-6, rtol=1e-5)


def test_non_square_multicamera_mapping():
    mapping = image_token_maps(
        [[1] + [99] * 6 + [2, 3] + [99] * 4 + [4]], [[1, 4, 6], [1, 4, 4]], 2, 99, ["head", "wrist"]
    )
    assert mapping[0]["grid_hw"] == [2, 3]
    assert mapping[0]["token_indices"] == list(range(1, 7))
    assert mapping[1]["token_indices"] == list(range(9, 13))
    with pytest.raises(ValueError):
        image_token_maps([[99, 99]], [[1, 4, 4]], 2, 99, ["head"])


def test_inverse_image_transform():
    grid = np.arange(6, dtype=np.float32).reshape(2, 3)
    rotated = grid[::-1, ::-1].copy()
    restored = restore_map(
        rotated, [{"resize_from_wh": [3, 2], "resize_to_wh": [3, 2], "rotation_degrees": 180, "padding": [0, 0, 0, 0]}]
    )
    np.testing.assert_array_equal(restored, grid)


class FakeInterface(nn.Module):
    def __init__(self):
        super().__init__()
        self.processor = SimpleNamespace(image_processor=SimpleNamespace(merge_size=2, patch_size=14))
        self.model = SimpleNamespace(config=SimpleNamespace(image_token_id=99))

    def build_qwenvl_inputs(self, images, instructions):
        return {
            "input_ids": torch.tensor([[1] + [99] * 6 + [2] + [99] * 4]),
            "image_grid_thw": torch.tensor([[1, 4, 6], [1, 4, 4]]),
        }


class MiniFramework(nn.Module):
    def __init__(self, layerwise=False):
        super().__init__()
        from diffusers.models.attention_processor import Attention

        self.layerwise = layerwise
        self.qwen_vl_interface = FakeInterface()
        self.action_model = nn.Module()
        self.action_model.action_horizon = 3
        self.action_model.action_encoder = nn.Identity()
        # Identity doesn't accept timestep: use a tiny two-argument encoder.
        self.action_model.action_encoder.forward = lambda x, t: x
        self.action_model.model = nn.Module()
        self.action_model.model.transformer_blocks = nn.ModuleList(
            [
                nn.ModuleDict({"attn1": Attention(query_dim=8, cross_attention_dim=8, heads=2, dim_head=4)})
                for _ in range(4)
            ]
        )
        # Session expects block.attn1, as in the real DiT.
        for block in self.action_model.model.transformer_blocks:
            assert block.attn1 is not None

    def predict_action(self, examples):
        self.qwen_vl_interface.build_qwenvl_inputs(images=[examples[0]["image"]], instructions=["numbered grid"])
        prefix = torch.arange(12 * 8, dtype=torch.float32).reshape(1, 12, 8) / 100
        hidden = torch.randn(1, 8, 8)  # state + future + 3 action queries
        for step in range(2):
            hidden = self.action_model.action_encoder(hidden, torch.tensor([step * 500]))
            for index, block in enumerate(self.action_model.model.transformer_blocks):
                encoder = prefix if self.layerwise or index % 2 == 0 else None
                hidden = block.attn1(hidden, encoder_hidden_states=encoder)
        return hidden[:, -3:]


@pytest.mark.parametrize("layerwise", [False, True])
def test_recorder_preserves_output_rng_and_cleanup(tmp_path, layerwise):
    framework = MiniFramework(layerwise).eval()
    image = Image.fromarray(np.zeros((56, 84, 3), dtype=np.uint8))
    examples = [{"image": [image, image.resize((56, 56))], "lang": "numbered grid"}]
    recorder = AttentionRecorder({"output_dir": str(tmp_path)})
    original_method = framework.qwen_vl_interface.build_qwenvl_inputs
    torch.manual_seed(5)
    baseline = framework.predict_action(examples)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(5)
    with recorder.session(framework, examples, {"camera_names": ["head", "wrist"], "env_step": 0}) as session:
        actual = framework.predict_action(examples)
        trace = session.save()
    torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng)
    assert framework.qwen_vl_interface.build_qwenvl_inputs == original_method
    assert not session.records
    metadata = json.loads((tmp_path / trace / "metadata.json").read_text())
    assert metadata["layers"] == ([0, 1, 2, 3] if layerwise else [0, 2])
    assert metadata["timesteps"] == {"0": [0], "1": [500]}
    with np.load(tmp_path / trace / "attention.npz") as arrays:
        assert arrays["camera_0"].shape == (2, len(metadata["layers"]), 3, 2, 3)
        assert np.all(arrays["camera_0"].sum((-1, -2)) < 1)
    frame, _ = render_trace(tmp_path / trace, tmp_path / "rendered")
    assert frame.width > 0
    # Failure paths remove hooks and patched methods too.
    with pytest.raises(RuntimeError), recorder.session(framework, examples):
        raise RuntimeError("intentional failure")
    assert framework.qwen_vl_interface.build_qwenvl_inputs == original_method
    for block in framework.action_model.model.transformer_blocks:
        assert not block.attn1._forward_hooks and not block.attn1._forward_pre_hooks


def test_chunk_alignment_and_episode_reset(tmp_path, monkeypatch):
    path = tmp_path / "alignment.jsonl"
    monkeypatch.setenv("STARVLA_TRACE_ALIGNMENT", str(path))
    trace = ChunkTrace()
    trace.received({"trace_id": "first"}, 0)
    for step in range(3):
        trace.executed(step, step)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["action_query"] for row in rows] == [0, 1, 2]
    assert {row["source_env_step"] for row in rows} == {0}
    trace.reset()
    assert trace.trace_id is None and trace.source_step is None


if __name__ == "__main__":
    import argparse

    from PIL import ImageDraw

    parser = argparse.ArgumentParser(description="Create clearly labelled synthetic validation artifacts")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    from pathlib import Path

    output = Path(args.output)
    framework = MiniFramework().eval()
    recorder = AttentionRecorder({"output_dir": str(output / "traces")}, provenance={"synthetic": True})
    images = []
    for width, height in ((168, 112), (112, 112)):
        image = Image.new("RGB", (width, height), (20, 40, 60))
        draw = ImageDraw.Draw(image)
        for index in range(width // 56 * (height // 56)):
            x, y = index % (width // 56) * 56, index // (width // 56) * 56
            draw.rectangle((x, y, x + 55, y + 55), outline="white")
            draw.text((x + 3, y + 3), str(index), fill="white")
        draw.text((3, height - 15), "SYNTHETIC", fill="yellow")
        images.append(image)
    for step in range(3):
        with recorder.session(
            framework,
            [{"image": images}],
            {"run": "synthetic_validation", "env_step": step, "camera_names": ["head", "wrist"]},
        ) as session:
            framework.predict_action([{"image": images}])
            session.save()
