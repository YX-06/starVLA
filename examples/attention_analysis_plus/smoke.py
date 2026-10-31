"""Check actual QwenPI_v4 actions and all RNG states with the recorder toggled."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from examples.attention_analysis_plus.model import load_framework, recorder_type, seed_all


def rng_state():
    return random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all()


def equal_rng(left, right):
    return (
        left[0] == right[0]
        and left[1][0] == right[1][0]
        and np.array_equal(left[1][1], right[1][1])
        and left[1][2:] == right[1][2:]
        and torch.equal(left[2], right[2])
        and all(torch.equal(a, b) for a, b in zip(left[3], right[3], strict=True))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    seed_all(7)
    framework = load_framework(args.checkpoint)
    images = [np.indices((224, 224)).sum(0).astype(np.uint8)[..., None].repeat(3, -1)] * 2
    examples = [{"image": images, "lang": "open the middle drawer of the cabinet"}]
    framework.predict_action(examples)
    recorder = recorder_type()({"output_dir": str(args.output / "traces")}, provenance={"synthetic": True})
    outputs, states, measurements = [], [], []
    for enabled in (False, True):
        seed_all(7)
        torch.cuda.synchronize()
        started = time.monotonic()
        if enabled:
            with recorder.session(framework, examples, {"camera_names": ["primary", "wrist"]}) as session:
                result = framework.predict_action(examples)
                session.save()
        else:
            result = framework.predict_action(examples)
        torch.cuda.synchronize()
        outputs.append(result["normalized_actions"])
        states.append(rng_state())
        measurements.append({"recorder": enabled, "seconds": time.monotonic() - started})
    report = {
        "strict_load": True,
        "synthetic_input": True,
        "seed": 7,
        "actions_identical": bool(np.array_equal(*outputs)),
        "rng_identical": equal_rng(*states),
        "max_action_difference": float(np.abs(outputs[0] - outputs[1]).max()),
        "measurements": measurements,
    }
    (args.output / "verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    if not report["actions_identical"] or not report["rng_identical"]:
        raise RuntimeError("Recorder changed actions or RNG")


if __name__ == "__main__":
    main()
