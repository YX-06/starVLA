"""Two GPU lanes for LIBERO PI and GR00T attention rollouts."""

import argparse
import concurrent.futures
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from websockets.sync.client import connect

from examples.attention_analysis.prepare import MODELS

CONDA_ROOT = Path("/mnt/afs/wangyunxiangt/miniconda3")
TASK_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def build_job(
    name,
    gpu,
    port,
    root,
    episodes,
    seed,
    run_label="",
    suite="libero_goal",
    max_tasks=1,
    grouped=False,
    frame_debug=False,
):
    _, step = MODELS[name]
    checkpoint = root / "checkpoints" / name / "checkpoints" / f"steps_{step}_pytorch_model.pt"
    run_name = f"{name}_{run_label}" if run_label else name
    if grouped:
        run_name += f"/{suite}"
    run = root / "runs" / run_name
    delivery = root / "runs" / "rendered" / run_name
    server_cmd = [
        str(CONDA_ROOT / "envs/starvla-attn/bin/python"),
        "-m",
        "deployment.model_server.server_policy",
        "--ckpt_path",
        str(checkpoint),
        "--port",
        str(port),
        "--attention-config",
        str(run / "attention.json"),
        "--seed",
        str(seed),
    ]
    client_cmd = [
        str(CONDA_ROOT / "envs/libero-attn/bin/python"),
        "-m",
        "examples.LIBERO.eval_files.eval_libero",
        "--args.task-suite-name",
        suite,
        "--args.max-tasks",
        str(max_tasks),
        "--args.num-trials-per-task",
        str(episodes),
        "--args.port",
        str(port),
        "--args.seed",
        str(seed),
        "--args.video-out-path",
        str(run / "rollout_videos"),
        "--args.results-out-path",
        str(run / "episodes.jsonl"),
    ]
    if max_tasks != 1:
        client_cmd.append("--args.isolate-tasks")
    if frame_debug:
        client_cmd.extend(["--args.frame-debug-dir", str(run / "frame_debug")])
    return {
        "name": name,
        "gpu": str(gpu),
        "seed": seed,
        "task_suite": suite,
        "episodes": episodes,
        "max_tasks": max_tasks,
        "run_name": run_name,
        "run_dir": str(run),
        "delivery_dir": str(delivery),
        "server_command": server_cmd,
        "client_command": client_cmd,
        "status": "pending",
        "frame_debug": frame_debug,
        "egl_render_guard": os.environ.get("STARVLA_EGL_RENDER_GUARD", "1") != "0",
    }


def job_environment(job, client=False):
    return dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=job.get("render_gpu", job["gpu"]) if client else job["gpu"],
        STARVLA_RENDER_GPU=job.get("render_gpu", job["gpu"]),
        STARVLA_EGL_RENDER_GUARD="1" if job.get("egl_render_guard", True) else "0",
        STARVLA_TRACE_RUN=job["run_name"],
        STARVLA_TRACE_ALIGNMENT=str(Path(job["run_dir"]) / "alignment.jsonl"),
    )


def save_status(job):
    (Path(job["run_dir"]) / "status.json").write_text(json.dumps(job, indent=2))


def preflight(job):
    run = Path(job["run_dir"])
    command = [
        job["client_command"][0],
        "-m",
        "examples.attention_analysis.preflight",
        "--task-suite",
        job["task_suite"],
        "--max-tasks",
        str(job["max_tasks"]),
        "--episodes",
        str(job["episodes"]),
        "--seed",
        str(job["seed"]),
        "--output",
        str(run / "preflight"),
        "--client-args",
        *job["client_command"][3:],
    ]
    try:
        job["status"] = "preflight"
        save_status(job)
        with (run / "preflight.log").open("w") as log:
            subprocess.run(
                command,
                env=job_environment(job, client=True),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=600,
            )
        job["status"] = "ready"
    except Exception as error:
        job.update(status="error", error=f"Preflight: {error}")
    save_status(job)
    return job


def run_one(job):
    run = Path(job["run_dir"])
    env = job_environment(job)
    job.update(status="running", render_backend="gpu")
    save_status(job)
    server = None
    try:
        with (run / "server.log").open("w") as server_log, (run / "rollout.log").open("w") as client_log:
            server = subprocess.Popen(
                job["server_command"], env=env, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True
            )
            deadline = time.monotonic() + 1800
            port = job["client_command"][job["client_command"].index("--args.port") + 1]
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited with {server.returncode}; see {run / 'server.log'}")
                try:
                    with connect(f"ws://127.0.0.1:{port}", open_timeout=1, proxy=None) as connection:
                        connection.recv(timeout=5)
                        break
                except OSError:
                    time.sleep(2)
            else:
                raise TimeoutError("Policy server did not become ready")
            result = subprocess.run(
                job["client_command"],
                env=job_environment(job, client=True),
                stdout=client_log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            job["client_exit_code"] = result.returncode
        # Even an interrupted/failed client may have valid episodes worth delivering.
        if any((run / "traces").glob("*/metadata.json")):
            with (run / "render.log").open("w") as log:
                subprocess.run(
                    [
                        job["server_command"][0],
                        "-m",
                        "examples.attention_analysis.render",
                        str(run / "traces"),
                        "--output",
                        job["delivery_dir"],
                        "--alignment",
                        str(run / "alignment.jsonl"),
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        if result.returncode:
            raise RuntimeError(f"Rollout exited with {result.returncode}; see {run / 'rollout.log'}")
        job["status"] = "executed"
    except Exception as error:
        job.update(status="error", error=str(error))
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()
        save_status(job)
    return job


def run_lane(jobs, preflight_only=False):
    """One model/GPU; never overlap suites or re-use a server across suites."""
    for job in jobs:
        if preflight_only:
            preflight(job)
        elif job["status"] == "ready":
            run_one(job)
        print(f"{job['run_name']}: {job['status']}", flush=True)
    return jobs


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gpus", nargs=2, default=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")[:2])
    parser.add_argument("--benchmark", choices=["libero"], default="libero")
    parser.add_argument("--task-suites", nargs="+", choices=TASK_SUITES, default=["libero_goal"])
    parser.add_argument("--max-tasks", type=int, default=1, help="-1 evaluates every task; default is the smoke task")
    parser.add_argument("--run-label", default="", help="Optional suffix for a separate set of artifacts")
    parser.add_argument(
        "--frame-debug", action="store_true", help="Save lossless frames/hashes around policy and rendering"
    )
    parser.add_argument(
        "--serial-cross-render",
        action="store_true",
        help="Run models sequentially and render on the other GPU to avoid CUDA/EGL contention",
    )
    args = parser.parse_args(argv)
    if args.run_label and not all(c.isalnum() or c in "_-" for c in args.run_label):
        parser.error("run-label may contain only letters, digits, underscores and hyphens")
    if args.episodes < 1 or args.max_tasks == 0 or args.max_tasks < -1:
        parser.error("episodes must be positive and max-tasks must be positive or -1")
    if len(args.gpus) != 2 or args.gpus[0] == args.gpus[1]:
        parser.error("two distinct GPUs are required")
    if len(set(args.task_suites)) != len(args.task_suites):
        parser.error("task-suites must not contain duplicates")
    return args


def main():
    args = parse_args()
    root = Path(os.environ["ATTENTION_ROOT"]).resolve()
    grouped = args.task_suites != ["libero_goal"] or args.max_tasks != 1
    lanes = [
        [
            build_job(
                f"libero_{model}",
                gpu,
                10093 + index,
                root,
                args.episodes,
                args.seed,
                args.run_label,
                suite,
                args.max_tasks,
                grouped,
                frame_debug=args.frame_debug,
            )
            for suite in args.task_suites
        ]
        for index, (model, gpu) in enumerate(zip(("pi", "groot"), args.gpus, strict=True))
    ]
    report_dir = root / "runs" / "reports" / (args.run_label or "smoke_summary")
    if args.serial_cross_render:
        for index, lane in enumerate(lanes):
            for job in lane:
                job["render_gpu"] = str(args.gpus[1 - index])
    with (Path("/tmp") / f"starvla-attn-experiments-{os.getuid()}.lock").open("w") as lock:
        print("Waiting for the two-GPU experiment slot", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        for lane in lanes:
            for job in lane:
                if Path(job["run_dir"]).exists() or Path(job["delivery_dir"]).exists():
                    raise FileExistsError(f"Existing artifacts: {job['run_dir']}; use a new --run-label")
        report_dir.mkdir(parents=True, exist_ok=False)
        for lane in lanes:
            for job in lane:
                run = Path(job["run_dir"])
                run.mkdir(parents=True)
                (run / "attention.json").write_text(json.dumps({"output_dir": str(run / "traces"), "interval": 1}))
                save_status(job)
        manifest = {
            "status": "running",
            "seed": args.seed,
            "episodes": args.episodes,
            "max_tasks": args.max_tasks,
            "task_suites": args.task_suites,
            "command": [sys.executable, "-m", "examples.attention_analysis.run_rollouts", *sys.argv[1:]],
            "jobs": [job for lane in lanes for job in lane],
            "scheduling": "serial_cross_render" if args.serial_cross_render else "parallel",
        }
        (report_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            # Validate all selected scenes and states before loading either model.
            for stage in (True, False):
                if args.serial_cross_render and not stage:
                    for lane in lanes:
                        run_lane(lane)
                else:
                    futures = [pool.submit(run_lane, lane, stage) for lane in lanes]
                    for future in futures:
                        future.result()
        from examples.attention_analysis.suite_report import build_report

        manifest["status"] = "finished"
        (report_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        report = build_report(report_dir)
    if not report["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
