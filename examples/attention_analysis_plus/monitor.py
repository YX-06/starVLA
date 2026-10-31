"""Read progress without importing model or simulator dependencies."""

import argparse
import json
import os
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label")
    args = parser.parse_args()
    run = Path(os.environ["PLUS_ROOT"]) / "runs" / args.label
    totals, suites, errors = Counter(), {}, []
    for name in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        path = run / name / "episodes.jsonl"
        rows = []
        if path.exists():
            contents = path.read_text()
            lines = contents.splitlines()
            if contents and not contents.endswith("\n"):
                lines = lines[:-1]
            rows = [json.loads(line) for line in lines]
        counts = Counter(row["status"] for row in rows)
        suites[name] = {"completed": len(rows), **counts}
        totals.update(counts)
        errors.extend(
            {"suite": name, "official_id": row["official_id"], "error": row.get("error")}
            for row in rows
            if row["status"] == "error"
        )
    summary = run / "summary.json"
    report = json.loads(summary.read_text()) if summary.exists() else {}
    complete = report.get("evaluation_complete", False)
    print(
        json.dumps(
            {
                "completed": sum(totals.values()),
                **totals,
                "suites": suites,
                "errors": errors,
                "evaluation_complete": complete,
                "validation_errors": report.get("validation_errors", {}),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
