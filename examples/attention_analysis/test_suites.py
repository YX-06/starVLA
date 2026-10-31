"""Focused regressions for multi-suite scheduling, reporting and streamed episode videos."""

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from PIL import Image

from examples.attention_analysis.client_trace import ChunkTrace
from examples.attention_analysis.render import render_exports
from examples.attention_analysis.run_rollouts import TASK_SUITES, build_job, job_environment, parse_args, run_lane
from examples.attention_analysis.suite_report import aggregate_job, build_report


def test_smoke_defaults_and_explicit_all_tasks(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    smoke = parse_args([])
    assert smoke.task_suites == ["libero_goal"] and smoke.max_tasks == 1
    full = parse_args(["--task-suites", *TASK_SUITES, "--max-tasks", "-1", "--gpus", "0", "1"])
    assert full.task_suites == list(TASK_SUITES) and full.max_tasks == -1


@pytest.mark.parametrize(
    "arguments",
    [
        ["--gpus", "0", "0"],
        ["--episodes", "0"],
        ["--max-tasks", "0"],
        ["--task-suites", "libero_goal", "libero_goal"],
        ["--run-label", "../old"],
    ],
)
def test_invalid_arguments(arguments):
    with pytest.raises(SystemExit):
        parse_args(arguments)


def test_two_gpu_jobs_have_isolated_outputs_and_seeds(tmp_path):
    jobs = [
        build_job(f"libero_{model}", gpu, port, tmp_path, 1, 7, "alltasks_seed7", suite, -1, True)
        for model, gpu, port in (("pi", 0, 10093), ("groot", 1, 10094))
        for suite in TASK_SUITES
    ]
    assert len({j["run_dir"] for j in jobs}) == len({j["delivery_dir"] for j in jobs}) == 8
    for job in jobs:
        client, server = job["client_command"], job["server_command"]
        assert client[client.index("--args.seed") + 1] == server[server.index("--seed") + 1] == "7"
        assert client[client.index("--args.task-suite-name") + 1] == job["task_suite"]
        assert client[client.index("--args.max-tasks") + 1] == "-1"
        assert job["gpu"] == ("0" if job["name"] == "libero_pi" else "1")


def test_lane_continues_after_suite_error(monkeypatch):
    calls = []
    jobs = [{"run_name": suite, "status": "ready"} for suite in TASK_SUITES]

    def fake_run(job):
        calls.append(job["run_name"])
        job["status"] = "error" if len(calls) == 1 else "executed"

    monkeypatch.setattr("examples.attention_analysis.run_rollouts.run_one", fake_run)
    run_lane(jobs)
    assert calls == list(TASK_SUITES)
    assert jobs[-1]["status"] == "executed"


def test_cross_gpu_render_keeps_model_placement(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert parse_args(["--serial-cross-render"]).serial_cross_render
    for model, gpu, render_gpu in (("pi", "0", "1"), ("groot", "1", "0")):
        job = build_job(f"libero_{model}", gpu, 10093, tmp_path, 1, 7)
        job["render_gpu"] = render_gpu
        assert job_environment(job)["CUDA_VISIBLE_DEVICES"] == gpu
        assert job_environment(job, client=True)["CUDA_VISIBLE_DEVICES"] == render_gpu
        assert job_environment(job, client=True)["STARVLA_RENDER_GPU"] == render_gpu


def test_frame_debug_is_opt_in_and_does_not_change_policy_command(tmp_path, monkeypatch):
    normal = build_job("libero_pi", 0, 10093, tmp_path, 1, 7)
    debug = build_job("libero_pi", 0, 10093, tmp_path, 1, 7, frame_debug=True)
    assert "--args.frame-debug-dir" not in normal["client_command"]
    assert debug["client_command"][-2:] == ["--args.frame-debug-dir", str(tmp_path / "runs/libero_pi/frame_debug")]
    assert debug["server_command"] == normal["server_command"]
    assert parse_args(["--gpus", "0", "1", "--frame-debug"]).frame_debug
    monkeypatch.setenv("STARVLA_EGL_RENDER_GUARD", "0")
    legacy = build_job("libero_pi", 0, 10093, tmp_path, 1, 7)
    monkeypatch.delenv("STARVLA_EGL_RENDER_GUARD")
    assert not legacy["egl_render_guard"] and job_environment(legacy, client=True)["STARVLA_EGL_RENDER_GUARD"] == "0"


def test_trace_identity_restarts_episode_per_task(tmp_path, monkeypatch):
    log = tmp_path / "alignment.jsonl"
    monkeypatch.setenv("STARVLA_TRACE_ALIGNMENT", str(log))
    trace = ChunkTrace()
    for task_id in (0, 1):
        trace.reset("libero_spatial", task_id, 0)
        trace.received({"trace_id": str(task_id)}, 0)
        trace.executed(0, 0)
        context = trace.context("task", 0, ["primary", "wrist"])
        assert context["task_id"] == task_id and context["episode_idx"] == 0
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["task_id"] for r in rows] == [0, 1]
    assert {r["episode_idx"] for r in rows} == {0}


def make_traces(root):
    rows = []
    for task_id in (0, 1):
        trace = root / f"trace{task_id}"
        trace.mkdir(parents=True)
        cameras = []
        for camera in range(2):
            Image.new("RGB", (32, 32), "gray").save(trace / f"camera_{camera}.png")
            cameras.append(
                {
                    "name": ("primary", "wrist")[camera],
                    "array_key": f"camera_{camera}",
                    "image_file": f"camera_{camera}.png",
                    "grid_hw": [2, 2],
                }
            )
        np.savez_compressed(
            trace / "attention.npz", camera_0=np.full((1, 1, 2, 2, 2), 0.1), camera_1=np.full((1, 1, 2, 2, 2), 0.2)
        )
        context = {
            "run": "test",
            "task_suite": "libero_goal",
            "task_id": task_id,
            "episode": 0,
            "episode_idx": 0,
            "env_step": 0,
        }
        metadata = {
            "trace_id": trace.name,
            "request_index": task_id,
            "trace_context": context,
            "framework": "test",
            "cameras": cameras,
            "denoise_steps": [0],
            "layers": [0],
            "action_queries": [0, 1],
        }
        (trace / "metadata.json").write_text(json.dumps(metadata))
        for step in range(2):
            rows.append(
                {**context, "trace_id": trace.name, "env_step": step, "source_env_step": 0, "action_query": step}
            )
    return rows


def test_streamed_videos_group_by_task_and_share_camera_scale(tmp_path):
    rows = make_traces(tmp_path / "traces")
    alignment = tmp_path / "alignment.jsonl"
    alignment.write_text("\n".join(json.dumps(row) for row in rows))
    records = render_exports(tmp_path / "traces", tmp_path / "videos", alignment=alignment)
    assert len(records) == 2
    for task_id, record in enumerate(records):
        assert record["task_id"] == task_id and record["episode_idx"] == 0
        assert record["vmax"] == pytest.approx(0.2)
        assert record["frames"] == 2
        assert len(record["trace_paths"]) == 1 and record["trace_paths"][0].endswith(f"trace{task_id}")
        with imageio.get_reader(record["video_path"]) as reader:
            assert sum(1 for _ in reader) == 2
        assert len(list(Path(record["video_path"]).parent.glob("*_camera_*.png"))) == 2


def test_renderer_rejects_cross_task_alignment(tmp_path):
    rows = make_traces(tmp_path / "traces")
    rows[0]["task_id"] = 5
    alignment = tmp_path / "alignment.jsonl"
    alignment.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="Cross-episode"):
        render_exports(tmp_path / "traces", tmp_path / "videos", alignment=alignment)


def test_report_marks_all_80_missing_without_inventing_failures(tmp_path):
    jobs = [
        build_job(f"libero_{model}", gpu, port, tmp_path, 1, 7, "batch", suite, -1, True)
        for model, gpu, port in (("pi", 0, 10093), ("groot", 1, 10094))
        for suite in TASK_SUITES
    ]
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "manifest.json").write_text(json.dumps({"seed": 7, "max_tasks": -1, "episodes": 1, "jobs": jobs}))
    report = build_report(report_dir)
    assert len(report["episodes"]) == 80 and not report["accepted"]
    assert all(m["expected"] == m["missing"] == 40 for m in report["models"].values())
    assert all(m["failure"] == m["error"] == 0 for m in report["models"].values())
    assert (report_dir / "summary.csv").exists() and (report_dir / "INDEX.md").exists()
    # A corrupt suite must still occupy all ten expected rows in the report.
    run = Path(jobs[0]["run_dir"])
    run.mkdir(parents=True)
    (run / "episodes.jsonl").write_text("{bad json")
    corrupt_report = build_report(report_dir)
    assert len(corrupt_report["episodes"]) == 80
    assert all(m["expected"] == m["missing"] == 40 for m in corrupt_report["models"].values())
    assert corrupt_report["suites"][0]["validation"]["status"] == "error"


def test_report_separates_task_failure_and_program_error(tmp_path, monkeypatch):
    job = build_job("libero_pi", 0, 10093, tmp_path, 1, 7, "batch", "libero_goal", -1, True)
    run = Path(job["run_dir"])
    run.mkdir(parents=True)
    rows = [
        {
            "task_suite": "libero_goal",
            "task_id": task_id,
            "episode_idx": 0,
            "seed": 7,
            "initial_state_index": 0,
            "status": status,
            "success": success,
        }
        for task_id, (status, success) in enumerate((("success", True), ("failure", False), ("error", None)))
    ]
    (run / "episodes.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    monkeypatch.setattr("examples.attention_analysis.suite_report.validate_job", lambda *a: {"status": "passed"})
    result_rows, summary = aggregate_job(job, {"seed": 7, "max_tasks": -1, "episodes": 1})
    assert len(result_rows) == 10
    assert (summary["success"], summary["failure"], summary["error"], summary["missing"]) == (1, 1, 1, 7)
    (run / "episodes.jsonl").write_text("\n".join(json.dumps(row) for row in [rows[0], rows[0]]))
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_job(job, {"seed": 7, "max_tasks": -1, "episodes": 1})


def test_known_visual_corruption_blocks_acceptance_despite_structural_success(tmp_path, monkeypatch):
    job = build_job("libero_pi", 0, 10093, tmp_path, 1, 7, "quality", "libero_goal", -1, True)
    summary = {
        "model": job["name"],
        "task_suite": job["task_suite"],
        "expected": 10,
        "success": 10,
        "failure": 0,
        "error": 0,
        "missing": 0,
        "process_status": "executed",
        "process_errors": 0,
        "validation": {"status": "passed"},
        "visual_quality": {"status": "failed"},
    }
    monkeypatch.setattr("examples.attention_analysis.suite_report.aggregate_job", lambda *args: ([], summary))
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "manifest.json").write_text(json.dumps({"seed": 7, "jobs": [job]}))
    report = build_report(report_dir)
    assert report["structural_validation_passed"]
    assert not report["accepted"]
