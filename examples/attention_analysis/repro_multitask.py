"""Model-free reproduction of native EGL aborts after repeated LIBERO tasks.

Run in libero-attn with env.sh sourced. Compare the ordinary loop with --isolate-tasks.
No policy, attention capture or benchmark modification is involved.
"""

import argparse
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isolate-tasks", action="store_true")
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--steps", type=int, default=230)
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument(
        "--legacy-cleanup", action="store_true", help="Disable the local EGL finalizer fix for diagnosis"
    )
    args = parser.parse_args()
    if args.isolate_tasks:
        for task_id in range(10):
            command = [
                sys.executable,
                "-m",
                "examples.attention_analysis.repro_multitask",
                "--task-id",
                str(task_id),
                "--steps",
                str(args.steps),
                "--task-suite",
                args.task_suite,
            ]
            if args.legacy_cleanup:
                command.append("--legacy-cleanup")
            subprocess.run(
                command,
                check=True,
            )
        print("All ten isolated tasks passed", flush=True)
        return
    if args.legacy_cleanup:
        from examples.attention_analysis import egl_cleanup

        egl_cleanup.install_egl_cleanup = lambda: False
    from libero.libero import benchmark

    from examples.LIBERO.eval_files.eval_libero import LIBERO_DUMMY_ACTION, _get_libero_env

    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    for task_id in range(10) if args.task_id is None else [args.task_id]:
        print("BEGIN", task_id, flush=True)
        started = time.monotonic()
        env, _ = _get_libero_env(suite.get_task(task_id), 256, 7)
        try:
            env.reset()
            env.set_init_state(suite.get_task_init_states(task_id)[0])
            for _ in range(args.steps):
                env.step(LIBERO_DUMMY_ACTION)
        finally:
            env.close()
        print("DONE", task_id, "seconds", time.monotonic() - started, flush=True)


if __name__ == "__main__":
    main()
