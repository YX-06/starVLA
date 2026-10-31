"""Run a smoke episode or the frozen 280-task subset with per-suite servers."""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from websockets.sync.client import connect

from examples.attention_analysis_plus.tasks import SUITES, load_manifest


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def task_directory(row):
    return f"task{row['official_id']:04d}_{row['variant_name']}"


def stop_server(server):
    if server is not None and server.poll() is None:
        os.killpg(server.pid, signal.SIGTERM)
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait()


def run_suite(args, suite, rows, output, checkpoint):
    suite_dir = output / suite
    suite_dir.mkdir()
    traces = suite_dir / "traces"
    server_cmd = [
        os.environ["PLUS_PYTHON"],
        "-u",
        "-m",
        "examples.attention_analysis_plus.server",
        "--checkpoint",
        str(checkpoint),
        "--traces",
        str(traces),
        "--port",
        str(args.port),
        "--seed",
        "7",
    ]
    server, episodes = None, []
    state = {
        "suite": suite,
        "status": "starting",
        "server_command": server_cmd,
        "episodes": [],
        "model_gpu": args.model_gpu,
        "render_gpu": args.render_gpu,
        "seed": 7,
        "rng_protocol": "new server per suite, continuous policy RNG within suite",
    }
    write_json(suite_dir / "status.json", state)
    try:
        with (suite_dir / "server.log").open("x") as log:
            server = subprocess.Popen(
                server_cmd,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.model_gpu),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + 1800
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Policy server exited: {server.returncode}; see {suite_dir / 'server.log'}")
                try:
                    with connect(f"ws://127.0.0.1:{args.port}", open_timeout=1, proxy=None) as connection:
                        connection.recv(timeout=5)
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Policy server startup timeout") from None
                    time.sleep(2)
            state["status"] = "running"
            write_json(suite_dir / "status.json", state)
            for row in rows:
                task_run = suite_dir / task_directory(row)
                command = [
                    os.environ["PLUS_SIM_PYTHON"],
                    "-u",
                    "-m",
                    "examples.attention_analysis_plus.worker",
                    "--manifest",
                    str(args.manifest),
                    "--suite",
                    suite,
                    "--task-id",
                    str(row["task_id"]),
                    "--run",
                    str(task_run),
                    "--checkpoint",
                    str(checkpoint),
                    "--port",
                    str(args.port),
                ]
                try:
                    with (suite_dir / f"task{row['official_id']:04d}.log").open("x") as task_log:
                        process = subprocess.run(
                            command,
                            env=dict(
                                os.environ, CUDA_VISIBLE_DEVICES=args.render_gpu, STARVLA_RENDER_GPU=args.render_gpu
                            ),
                            stdout=task_log,
                            stderr=subprocess.STDOUT,
                            timeout=args.task_timeout,
                            check=False,
                        )
                    episode = json.loads((task_run / "episode.json").read_text())
                    if process.returncode:
                        episode.update(
                            status="error",
                            success=None,
                            error=episode.get("error", f"Worker exited {process.returncode}"),
                        )
                except Exception as error:
                    episode = {**row, "status": "error", "success": None, "error": str(error)}
                episodes.append(episode)
                state["episodes"] = episodes
                write_json(suite_dir / "status.json", state)
                with (suite_dir / "episodes.jsonl").open("a") as stream:
                    stream.write(json.dumps(episode) + "\n")
                print(f"{suite} {row['official_id']} {episode['status']} " f"({len(episodes)}/{len(rows)})", flush=True)
        state["status"] = "executed" if all(r["status"] != "error" for r in episodes) else "error"
    except Exception as error:
        state.update(status="error", error=str(error))
        raise
    finally:
        stop_server(server)
        write_json(suite_dir / "status.json", state)
    return episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(os.environ["PLUS_ROOT"])
    parser.add_argument("--manifest", type=Path, default=root / "task_manifest.json")
    parser.add_argument("--label", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model-gpu", default="0")
    parser.add_argument("--render-gpu", default="1")
    parser.add_argument("--port", type=int, default=10103)
    parser.add_argument("--task-timeout", type=int, default=3600)
    parser.add_argument("--smoke-report", type=Path, default=root / "runs/smoke_seed7_r1/summary.json")
    args = parser.parse_args()
    if not args.label or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.label
    ):
        parser.error("label must contain only letters, digits, underscore and hyphen")
    if args.model_gpu == args.render_gpu:
        parser.error("Model and rendering require distinct GPUs")
    args.manifest = args.manifest.resolve()
    manifest = load_manifest(args.manifest)
    preflight = json.loads((root / "runs/reproducibility/preflight.json").read_text())
    parity = json.loads((root / "runs/reproducibility/model-parity.json").read_text())
    if not (
        preflight["passed"]
        and preflight["task_count"] == 280
        and parity["strict_load"]
        and parity["actions_identical"]
        and parity["rng_identical"]
    ):
        raise ValueError("Successful task preflight and actual-model recorder parity are required")
    if not args.smoke:
        smoke = json.loads(args.smoke_report.read_text())
        if not (smoke["smoke"] and smoke["structural_validation_passed"] and smoke["recorded"] == 1):
            raise ValueError("One structurally valid complete smoke episode is required before formal evaluation")
    output = root / "runs" / args.label
    checkpoint = root / "checkpoints/libero_plus_pi/checkpoints/steps_50000_pytorch_model.pt"
    started = time.monotonic()
    with Path(f"/tmp/starvla-attn-experiments-{os.getuid()}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        output.mkdir(parents=True, exist_ok=False)
        state = {
            "status": "running",
            "smoke": args.smoke,
            "manifest": str(args.manifest),
            "command": [sys.executable, "-m", "examples.attention_analysis_plus.run", *sys.argv[1:]],
            "expected_episodes": 1 if args.smoke else 280,
        }
        write_json(output / "status.json", state)
        try:
            suites = SUITES[:1] if args.smoke else SUITES
            for suite in suites:
                rows = [r for r in manifest["tasks"] if r["task_suite"] == suite]
                run_suite(args, suite, rows[:1] if args.smoke else rows, output, checkpoint)
            state["status"] = "executed"
        except Exception as error:
            state.update(status="error", error=str(error))
            raise
        finally:
            state["elapsed_seconds"] = time.monotonic() - started
            write_json(output / "status.json", state)
    # Postprocessing uses only the CPU, after releasing both GPUs.
    from examples.attention_analysis_plus.report import build_report

    report = build_report(output, args.manifest, publish=not args.smoke)
    if not report["structural_validation_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
