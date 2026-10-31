"""Validate and render a snapshot of completed tasks while rollouts continue."""

import argparse
import json
import os
from pathlib import Path

from examples.attention_analysis_plus.report import validate_task
from examples.attention_analysis_plus.run import task_directory, write_json
from examples.attention_analysis_plus.tasks import SUITES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label")
    args = parser.parse_args()
    root = Path(os.environ["PLUS_ROOT"])
    run = root / "runs" / args.label
    stats = json.loads((root / "checkpoints/libero_plus_pi/dataset_statistics.json").read_text())["franka"]["action"]
    for suite in SUITES:
        path = run / suite / "episodes.jsonl"
        if not path.exists():
            continue
        contents = path.read_text()
        lines = contents.splitlines()
        if contents and not contents.endswith("\n"):
            lines = lines[:-1]
        for row in (json.loads(line) for line in lines):
            task = run / suite / task_directory(row)
            if row["status"] == "error" or (task / "validation.json").exists():
                continue
            write_json(task / "validation.json", validate_task(task, run / suite, row, stats))
            print(suite, row["official_id"], "validated", flush=True)


if __name__ == "__main__":
    main()
