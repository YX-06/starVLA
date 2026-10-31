"""Strict checkpoint load, action/RNG parity, repeated-memory and cost report."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from deployment.model_server.policy_wrapper import PolicyServerWrapper
from starVLA.model.analysis.attention import AttentionRecorder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cameras", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # This is explicitly a synthetic observation, never a rollout success.
    images = []
    for camera in range(args.cameras):
        image = Image.new("RGB", (336, 224), (30 + camera * 40, 40, 70))
        draw = ImageDraw.Draw(image)
        for index in range(24):
            x, y = (index % 6) * 56, (index // 6) * 56
            draw.rectangle((x, y, x + 55, y + 55), outline="white")
            draw.text((x + 8, y + 8), str(index), fill="white")
        images.append(np.asarray(image))
    example = {"image": images, "lang": "Pick up the bottle."}
    wrapper = PolicyServerWrapper(args.checkpoint)
    wrapper.predict_action([example])  # Warm kernels before comparing latency/peak memory.
    torch.cuda.synchronize()
    recorder = AttentionRecorder(
        {"output_dir": str(args.output / "traces")}, provenance={"checkpoint": args.checkpoint, "synthetic": True}
    )
    report = {
        "checkpoint": args.checkpoint,
        "strict_load": "passed",
        "synthetic_input": True,
        "seed": args.seed,
        "measurements": [],
    }
    outputs, states = [], []
    for enabled in (False, True):
        torch.manual_seed(args.seed)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        wrapper._attention_recorder = recorder if enabled else None
        start = time.perf_counter()
        response = wrapper.predict_action(
            [example],
            trace_context={
                "run": "synthetic_smoke",
                "env_step": 0,
                "camera_names": [f"camera_{i}" for i in range(args.cameras)],
            },
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        outputs.append(response["actions"])
        states.append((torch.get_rng_state(), torch.cuda.get_rng_state()))
        report["measurements"].append(
            {
                "recorder": enabled,
                "seconds": elapsed,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "allocated_after_bytes": torch.cuda.memory_allocated(),
            }
        )
    report["actions_identical"] = bool(np.array_equal(*outputs))
    report["max_action_difference"] = float(np.abs(outputs[0] - outputs[1]).max())
    report["rng_identical"] = all(torch.equal(left, right) for left, right in zip(*states, strict=False))
    memory = []
    for step in range(args.repeats):
        wrapper.predict_action([example], trace_context={"run": "synthetic_smoke", "env_step": step + 1})
        torch.cuda.synchronize()
        memory.append(torch.cuda.memory_allocated())
    report["repeated_allocated_bytes"] = memory
    (args.output / "verification.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(args.output / "actions.npz", disabled=outputs[0], enabled=outputs[1])
    report["stable_allocated_memory"] = len(set(memory)) <= 1
    (args.output / "verification.json").write_text(json.dumps(report, indent=2))
    if not report["actions_identical"] or not report["rng_identical"] or not report["stable_allocated_memory"]:
        raise RuntimeError("Recorder changed action output or RNG state")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
