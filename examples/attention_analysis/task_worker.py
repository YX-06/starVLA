"""Isolate MuJoCo/EGL lifetimes while retaining one seeded policy server per suite."""

import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path


def eval_isolated_tasks(args):
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    n_tasks = suite.n_tasks if args.max_tasks < 1 else min(args.max_tasks, suite.n_tasks)
    output = Path(args.results_out_path or Path(args.video_out_path) / "episodes.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    workers = output.parent / "task_workers"
    workers.mkdir(exist_ok=False)
    errors = 0
    with output.open("x") as results:
        for task_id in range(n_tasks):
            task_output = workers / f"task{task_id:02d}.jsonl"
            config = workers / f"task{task_id:02d}.config.json"
            config.write_text(
                json.dumps(
                    dataclasses.asdict(
                        dataclasses.replace(
                            args,
                            task_id=task_id,
                            isolate_tasks=False,
                            results_out_path=str(task_output),
                        )
                    ),
                    indent=2,
                )
            )
            started = time.monotonic()
            reason, returncode = None, None
            try:
                with (workers / f"task{task_id:02d}.log").open("x") as log:
                    process = subprocess.run(
                        [sys.executable, "-m", "examples.attention_analysis.task_worker", str(config)],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                        timeout=args.task_timeout_seconds,
                    )
                returncode = process.returncode
                if returncode:
                    reason = f"Task worker exited with {returncode}; see {workers / f'task{task_id:02d}.log'}"
            except Exception as error:
                reason = f"Task worker: {type(error).__name__}: {error}"
            rows = []
            if task_output.exists():
                try:
                    rows = [json.loads(line) for line in task_output.read_text().splitlines() if line.strip()]
                except (ValueError, OSError) as error:
                    reason = f"Invalid task results: {error}"
            by_episode = {row["episode_idx"]: row for row in rows}
            if len(by_episode) != len(rows):
                reason = "Duplicate worker episode results"
            for episode_idx in range(args.num_trials_per_task):
                row = by_episode.get(
                    episode_idx,
                    {
                        "task_suite": args.task_suite_name,
                        "task_id": task_id,
                        "task_name": suite.get_task(task_id).language,
                        "episode_idx": episode_idx,
                        "seed": args.seed,
                        "initial_state_index": episode_idx,
                        "success": None,
                        "status": "error",
                        "execution_steps": 0,
                        "video_path": None,
                        "elapsed_seconds": time.monotonic() - started,
                        "error": reason or "Task worker did not write an episode result",
                    },
                )
                row["worker_exit_code"] = returncode
                if reason:
                    row.update(status="error", error=reason)
                errors += row["status"] == "error"
                results.write(json.dumps(row) + "\n")
                results.flush()
                print("Episode result:", json.dumps(row), flush=True)
    if errors:
        raise RuntimeError(f"{errors} isolated episode program errors; see {workers}")


if __name__ == "__main__":
    from examples.LIBERO.eval_files.eval_libero import Args, eval_libero

    eval_libero(Args(**json.loads(Path(sys.argv[1]).read_text())))
