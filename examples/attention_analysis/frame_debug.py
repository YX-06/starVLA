"""Optional, lossless evidence around policy calls, simulator steps and encoding.

The recorder itself has no simulator/OpenGL dependency and uses no RNG. Its
context callback is optional, so CPU regressions never initialize a GPU backend.
"""

import hashlib
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image


def renderer_context_snapshot(env):
    simulator = getattr(env, "sim", None)
    if simulator is None:
        simulator = getattr(getattr(env, "env", None), "sim", None)
    renderer = getattr(simulator, "_render_context_offscreen", None)
    handle = getattr(getattr(renderer, "gl_ctx", None), "_context", None)
    state = {"backend": os.environ.get("MUJOCO_GL"), "owner_context": getattr(handle, "address", None)}
    if handle and state["backend"] == "egl":
        # Called only for an already-created real renderer. Do not consume GL errors.
        from OpenGL import EGL

        current = EGL.eglGetCurrentContext()
        state["current_context"] = getattr(current, "address", None)
    return state


class FrameBoundaryAudit:
    def __init__(self, directory, context_reader=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.context_reader = context_reader
        self.hashes = {}
        self.events = 0

    def capture(self, stage, step, images, **extra):
        context = self.context_reader() if self.context_reader is not None else None
        event = {
            "stage": stage,
            "env_step": int(step),
            "thread_id": threading.get_ident(),
            "monotonic_ns": time.monotonic_ns(),
            "context": context,
            "images": [],
            **extra,
        }
        for name, source in images.items():
            key = stage, int(step), name
            if key in self.hashes:
                raise ValueError(f"Duplicate frame diagnostic: {key}")
            array = np.asarray(source)
            if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
                raise ValueError("Frame diagnostics require HWC uint8 RGB arrays")
            snapshot = np.array(array, copy=True, order="C")
            digest = hashlib.sha256(snapshot.tobytes()).hexdigest()
            filename = f"step{step:06d}_{stage}_{name}.png"
            Image.fromarray(snapshot).save(self.directory / filename)
            self.hashes[key] = digest
            event["images"].append(
                {
                    "camera": name,
                    "file": filename,
                    "sha256": digest,
                    "shape": list(array.shape),
                    "strides": list(array.strides),
                    "owns_memory": bool(array.flags.owndata),
                    "c_contiguous": bool(array.flags.c_contiguous),
                }
            )
        with (self.directory / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        self.events += 1

    def finish(self):
        issues, comparisons = [], 0
        for (stage, step, camera), digest in self.hashes.items():
            reference = {
                "after_policy": "before_policy",
                "before_encode": "before_policy",
                "before_policy": "after_env_step",
            }.get(stage)
            key = reference, step, camera
            if reference is not None and key in self.hashes:
                comparisons += 1
                if digest != self.hashes[key]:
                    issues.append({"env_step": step, "camera": camera, "from": reference, "to": stage})
        result = {
            "events": self.events,
            "hash_comparisons": comparisons,
            "buffer_changes": issues,
            "buffer_integrity": "failed" if issues else ("passed" if comparisons else "not_checked"),
            "scope": "Equal hashes establish buffer integrity only, not visually correct rendering.",
        }
        (self.directory / "summary.json").write_text(json.dumps(result, indent=2))
        return result
