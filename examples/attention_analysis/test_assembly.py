"""An explicit whole-suite retry must not cherry-pick episode outcomes or mix seeds."""

import json

import pytest

from examples.attention_analysis.assemble_report import assemble_manifest
from examples.attention_analysis.run_rollouts import TASK_SUITES, build_job


def test_assembly_replaces_entire_suites_and_checks_seeds(tmp_path):
    sources = [tmp_path / "base", tmp_path / "last_suite"]
    for source, suites in zip(sources, (TASK_SUITES, ["libero_10"]), strict=True):
        source.mkdir()
        jobs = [
            build_job(f"libero_{model}", gpu, port, source, 1, 7, "run", suite, -1, True)
            for model, gpu, port in (("pi", 0, 10093), ("groot", 1, 10094))
            for suite in suites
        ]
        (source / "manifest.json").write_text(
            json.dumps(
                {
                    "seed": 7,
                    "episodes": 1,
                    "max_tasks": -1,
                    "task_suites": list(suites),
                    "jobs": jobs,
                }
            )
        )
    merged = assemble_manifest(sources)
    assert len(merged["jobs"]) == 8
    for job in merged["jobs"]:
        assert job["source_report"] == str(sources[job["task_suite"] == "libero_10"].resolve())
    path = sources[-1] / "manifest.json"
    changed = json.loads(path.read_text())
    changed["seed"] = 8
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Cannot mix"):
        assemble_manifest(sources)
    changed["seed"] = 7
    changed["jobs"].append(changed["jobs"][0])
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Duplicate"):
        assemble_manifest(sources)


def test_assembly_includes_new_suite_beyond_completed_subset(tmp_path):
    sources = [tmp_path / "completed", tmp_path / "remaining"]
    for source, suites in zip(sources, (TASK_SUITES[:3], TASK_SUITES[3:]), strict=True):
        source.mkdir()
        jobs = [
            build_job(f"libero_{model}", gpu, 10093, tmp_path, 1, 7, suite=suite)
            for model, gpu in (("pi", 0), ("groot", 1))
            for suite in suites
        ]
        (source / "manifest.json").write_text(
            json.dumps(
                {
                    "seed": 7,
                    "episodes": 1,
                    "max_tasks": -1,
                    "task_suites": list(suites),
                    "jobs": jobs,
                }
            )
        )
    merged = assemble_manifest(sources)
    assert len(merged["jobs"]) == 8 and merged["task_suites"] == list(TASK_SUITES)
