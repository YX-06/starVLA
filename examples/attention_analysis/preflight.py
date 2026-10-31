"""Create a real simulator, render an observation and take one step before loading a policy."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def libero_check(output, suite_name="libero_goal", max_tasks=1, seed=7, episodes=1):
    from libero.libero import benchmark, get_libero_path
    from OpenGL import GL

    from examples.LIBERO.eval_files.eval_libero import LIBERO_DUMMY_ACTION, _get_libero_env

    suite = benchmark.get_benchmark_dict()[suite_name]()
    records = []
    for task_id in range(suite.n_tasks if max_tasks < 1 else min(max_tasks, suite.n_tasks)):
        task = suite.get_task(task_id)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        if not bddl.is_file():
            raise FileNotFoundError(bddl)
        states = suite.get_task_init_states(task_id)
        if len(states) < episodes:
            raise ValueError(f"{suite_name}/{task_id}: only {len(states)} initial states for {episodes} episodes")
        env, description = _get_libero_env(task, 256, seed)
        try:
            env.reset()
            observation = env.set_init_state(states[0])
            images = [observation[key] for key in ("agentview_image", "robot0_eye_in_hand_image")]
            for camera, image in zip(("primary", "wrist"), images, strict=True):
                if image.shape != (256, 256, 3) or np.std(image) < 1:
                    raise ValueError(f"Invalid render: {suite_name}/{task_id}/{camera}")
                Image.fromarray(image).save(output / f"task{task_id:02d}_{camera}.png")
            renderer = str(GL.glGetString(GL.GL_RENDERER))
            if "NVIDIA" not in renderer:
                raise RuntimeError(f"GPU rendering required, got {renderer}")
            env.step(LIBERO_DUMMY_ACTION)
            records.append(
                {
                    "task_id": task_id,
                    "task": description,
                    "bddl": str(bddl),
                    "initial_states": len(states),
                    "image_shapes": [list(image.shape) for image in images],
                    "renderer": renderer,
                    "vendor": str(GL.glGetString(GL.GL_VENDOR)),
                    "reset_render_step": "passed",
                }
            )
        finally:
            env.close()
    return {"benchmark": suite_name, "seed": seed, "tasks": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("libero",), default="libero")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-suite", default="libero_goal")
    parser.add_argument("--max-tasks", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--client-args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.benchmark == "libero" and args.client_args:
        import tyro.extras

        from examples.LIBERO.eval_files.eval_libero import eval_libero

        tyro.extras.get_parser(eval_libero).parse_args(args.client_args)
    result = libero_check(args.output, args.task_suite, args.max_tasks, args.seed, args.episodes)
    (args.output / "preflight.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
