"""Packaging preserves complete suites, normal failures and standalone task files."""

import json

import pytest

from examples.attention_analysis.publish_results import publish_results, refresh_indexes


def make_source(root):
    report, run, media = root / "report", root / "run", root / "media"
    report.mkdir()
    run.mkdir()
    rows, alignment = [], []
    for task_id in range(10):
        trace = run / "traces" / f"trace{task_id}"
        trace.mkdir(parents=True)
        context = {"run": "source", "task_suite": "libero_goal", "task_id": task_id, "episode_idx": 0}
        (trace / "metadata.json").write_text(
            json.dumps({"trace_context": context, "cameras": [{"image_file": "in.png"}]})
        )
        (trace / "attention.npz").write_bytes(f"numeric{task_id}".encode())
        (trace / "in.png").write_bytes(b"image")
        task = media / str(task_id)
        task.mkdir(parents=True)
        for name in ("raw.mp4", "attention.mp4", "trace_combined.png"):
            (task / name).write_bytes(b"media")
        (task / "render.json").write_text(
            json.dumps({**context, "trace_paths": [str(trace)], "video_path": str(task / "attention.mp4")})
        )
        alignment.append({**context, "env_step": 0, "source_env_step": 0, "action_query": 0, "trace_id": trace.name})
        rows.append(
            {
                "model": "libero_pi",
                "task_suite": "libero_goal",
                "task_id": task_id,
                "task_name": f"Task {task_id}",
                "episode_idx": 0,
                "initial_state_index": 0,
                "seed": 7,
                "status": "failure" if task_id == 6 else "success",
                "success": task_id != 6,
                "execution_steps": 1,
                "elapsed_seconds": 1,
                "run_dir": str(run),
                "trace_index_path": str(task / "render.json"),
                "overlay_dir": str(task),
                "video_path": str(task / "raw.mp4"),
                "attention_video_path": str(task / "attention.mp4"),
            }
        )
    (run / "alignment.jsonl").write_text("\n".join(json.dumps(row) for row in alignment))
    (report / "manifest.json").write_text(json.dumps({"seed": 7, "episodes": 1, "max_tasks": -1}))
    (report / "summary.json").write_text(json.dumps({"accepted": True, "episodes": rows}))
    return report


def test_published_tasks_are_standalone_and_normal_failure_is_preserved(tmp_path):
    source = make_source(tmp_path)
    output = tmp_path / "deliverables"
    result = publish_results(source, output)
    assert (result["completed"], result["pending"]) == (10, 70)
    assert result["outcomes"] == {"success": 9, "failure": 1}
    assert [p.name for p in output.iterdir() if p.is_dir()] == ["libero-goal-pi"]
    task = output / "libero-goal-pi/task06_task_6"
    assert json.loads((task / "episode.json").read_text())["status"] == "failure"
    render = json.loads((task / "render.json").read_text())
    assert render["trace_paths"] == ["traces/trace6"]
    assert render["video_path"] == "attention.mp4"
    assert (task / render["trace_paths"][0] / "attention.npz").read_bytes() == b"numeric6"
    assert all(not p.is_symlink() for p in task.rglob("*"))
    # Re-publishing the same source is safe, but cannot replace a normal failure.
    assert publish_results(source, output)["outcomes"] == result["outcomes"]
    episode = json.loads((task / "episode.json").read_text())
    episode["visual_quality"] = {"status": "failed"}
    (task / "episode.json").write_text(json.dumps(episode))
    assert refresh_indexes(output)["visual_quality_failed_tasks"] == 1
    assert "Visual quality FAILED" in (output / "INDEX.md").read_text()
    report = json.loads((source / "summary.json").read_text())
    report["episodes"][6].update(status="success", success=True)
    (source / "summary.json").write_text(json.dumps(report))
    with pytest.raises(FileExistsError, match="outcome"):
        publish_results(source, output)


def test_refuse_unvalidated_or_incomplete_suite(tmp_path):
    source = make_source(tmp_path)
    path = source / "summary.json"
    report = json.loads(path.read_text())
    report["accepted"] = False
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="validated"):
        publish_results(source, tmp_path / "rejected")
    report["accepted"] = True
    report["episodes"].pop()
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Incomplete"):
        publish_results(source, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()
