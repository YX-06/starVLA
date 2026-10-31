"""Verify every selected runtime name and initial state before model inference."""

import json
import os
from pathlib import Path

import numpy as np

from examples.attention_analysis_plus.tasks import SUITES, load_manifest, read_map, select_tasks, validate_mapping


def main():
    from libero.libero import benchmark

    root = Path(os.environ["PLUS_ROOT"])
    manifest = load_manifest(root / "task_manifest.json")
    folder = root / "third_party/LIBERO-plus/libero/libero/benchmark"
    original = root.parent / "attention_analysis/third_party/LIBERO/libero/libero/benchmark/libero_suite_task_map.py"
    selected = select_tasks(
        json.loads((folder / "task_classification.json").read_text()),
        read_map(original),
        read_map(folder / "libero_suite_task_map.py"),
    )
    if selected != manifest["tasks"]:
        raise ValueError("Frozen selection differs from recomputation against pinned sources")
    checked = []
    for suite_name in SUITES:
        suite = benchmark.get_benchmark_dict()[suite_name](task_order_index=0)
        rows = [row for row in selected if row["task_suite"] == suite_name]
        validate_mapping(rows, suite)
        for row in rows:
            states = np.asarray(suite.get_task_init_states(row["task_id"]))
            if states.ndim != 2 or states.shape[0] < 1 or not np.isfinite(states[0]).all():
                raise ValueError(f"Invalid initial_states[0]: {row}")
            checked.append({"task_suite": suite_name, "task_id": row["task_id"], "state_shape": list(states.shape)})
    output = root / "runs/reproducibility/preflight.json"
    output.write_text(json.dumps({"passed": True, "task_count": len(checked), "tasks": checked}, indent=2))
    print(f"Verified {len(checked)} task names and initial_states[0]", flush=True)


if __name__ == "__main__":
    main()
