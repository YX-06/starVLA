"""Assemble independently seeded suites; later sources replace an entire suite per model."""

import argparse
import json
import sys
import time
from pathlib import Path

from examples.attention_analysis.suite_report import build_report


def assemble_manifest(sources):
    manifest, selected = None, {}
    for source in sources:
        source = Path(source).resolve()
        current = json.loads((source / "manifest.json").read_text())
        if manifest is None:
            manifest = {key: current[key] for key in ("seed", "episodes", "max_tasks", "task_suites")}
        if any(current[key] != manifest[key] for key in ("seed", "episodes", "max_tasks")):
            raise ValueError(f"Cannot mix different seeds, episode counts or task limits: {source}")
        seen = set()
        for job in current["jobs"]:
            key = job["name"], job["task_suite"]
            if key in seen:
                raise ValueError(f"Duplicate model/suite in {source}: {key}")
            seen.add(key)
            selected[key] = {**job, "source_report": str(source)}
    if manifest is None:
        raise ValueError("No source reports")
    manifest.update(
        status="waiting",
        jobs=list(selected.values()),
        task_suites=list(dict.fromkeys(job["task_suite"] for job in selected.values())),
        source_reports=[str(Path(p).resolve()) for p in sources],
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    manifest = assemble_manifest(args.sources)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest["assembly_command"] = [sys.executable, "-m", "examples.attention_analysis.assemble_report", *sys.argv[1:]]
    manifest["reproduction_command"] = [
        "bash",
        "examples/attention_analysis/launch_all_tasks.sh",
        "NEW_RUN_LABEL",
    ]
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.output / "run_notes.json").write_text(
        json.dumps(
            {
                "selection": (
                    "Later sources replace the entire model/suite; no selection of individual episode outcomes."
                ),
                "independence": (
                    "Each suite starts a fresh policy server and simulator seed=7; "
                    "whole-suite replacement preserves the protocol."
                ),
                "source_reports": manifest["source_reports"],
                "egl_cleanup": (
                    "The final libero_10 retry binds the owning EGL context before renderer destruction, "
                    "then restores the caller's context."
                ),
            },
            indent=2,
        )
    )
    if args.wait:
        while True:
            states = [
                json.loads((Path(job["run_dir"]) / "status.json").read_text())["status"] for job in manifest["jobs"]
            ]
            if all(state in ("executed", "error") for state in states):
                break
            time.sleep(10)
    report = build_report(args.output)
    manifest["status"] = "finished" if args.wait else "snapshot"
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"accepted": report["accepted"], "models": report["models"]}, indent=2), flush=True)
    raise SystemExit(0 if report["accepted"] else 1)


if __name__ == "__main__":
    main()
