"""A single task/episode in a fresh simulator process; no policy RNG resets."""

import argparse
import faulthandler
import json
import os
import random
import time
import traceback
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from examples.attention_analysis_plus.client import ModelClient
from examples.attention_analysis_plus.tasks import load_manifest, validate_mapping

MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}


def run_episode(selection, run, checkpoint, port):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    from examples.attention_analysis.egl_cleanup import install_egl_cleanup

    faulthandler.enable()
    random.seed(7)
    np.random.seed(7)
    install_egl_cleanup()
    suite = benchmark.get_benchmark_dict()[selection["task_suite"]](task_order_index=0)
    validate_mapping([selection], suite)
    task_id = selection["task_id"]
    task = suite.get_task(task_id)
    row = {
        **selection,
        "episode_idx": 0,
        "initial_state_index": 0,
        "seed": 7,
        "status": "error",
        "success": None,
        "execution_steps": 0,
        "task_name": task.language,
        "video_path": str((run / "rollout.mp4").resolve()),
        "wrist_video_path": str((run / "rollout_wrist.mp4").resolve()),
        "action_unnormalization": "published VLAct Plus min/max/mask; clip [-1,1]; gripper threshold=0.5",
        "visual_postprocessing": "none; original frames encoded unchanged",
        "egl_render_guard": os.environ.get("STARVLA_EGL_RENDER_GUARD"),
    }
    env, started = None, time.monotonic()
    try:
        states = suite.get_task_init_states(task_id)
        env = OffScreenRenderEnv(
            bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
            camera_heights=256,
            camera_widths=256,
        )
        env.seed(7)
        client = ModelClient(checkpoint, port, run, selection)
        env.reset()
        obs = env.set_init_state(states[0])
        for _ in range(10):
            obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
        frames, wrist_frames = [], []
        with (run / "alignment.jsonl").open("x") as alignment:
            for step in range(MAX_STEPS[selection["task_suite"]]):
                images = [
                    np.ascontiguousarray(obs[name][::-1, ::-1])
                    for name in ("agentview_image", "robot0_eye_in_hand_image")
                ]
                if any(image.shape != (256, 256, 3) or image.dtype != np.uint8 for image in images):
                    raise ValueError(f"Invalid raw camera images: {[(x.shape, str(x.dtype)) for x in images]}")
                frames.append(images[0].copy())
                wrist_frames.append(images[1].copy())
                action, source = client.action(images, task.language, step)
                delta = np.concatenate(
                    (np.asarray(action[:6], dtype=np.float32), [np.float32(1.0 - 2.0 * (float(action[6]) > 0.5))])
                )
                obs, _, done, _ = env.step(delta.tolist())
                alignment.write(
                    json.dumps({**source, "raw_action": action.tolist(), "executed_action": delta.tolist()}) + "\n"
                )
                alignment.flush()
                row["execution_steps"] = step + 1
                if done:
                    break
        row.update(success=bool(done), status="success" if done else "failure")
        for filename, images in (("rollout.mp4", frames), ("rollout_wrist.mp4", wrist_frames)):
            with imageio.get_writer(run / filename, fps=10) as writer:
                for frame in images:
                    writer.append_data(frame)
    except Exception as error:
        row.update(status="error", success=None, error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as error:
                row.update(status="error", success=None, error=f"Environment close: {error}")
        row["elapsed_seconds"] = time.monotonic() - started
        (run / "episode.json").write_text(json.dumps(row, ensure_ascii=False, indent=2))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--port", type=int, default=10103)
    args = parser.parse_args()
    rows = [
        r
        for r in load_manifest(args.manifest)["tasks"]
        if r["task_suite"] == args.suite and r["task_id"] == args.task_id
    ]
    if len(rows) != 1:
        raise ValueError("Task must occur exactly once in the frozen manifest")
    args.run.mkdir(parents=True, exist_ok=False)
    result = run_episode(rows[0], args.run, args.checkpoint, args.port)
    print(json.dumps(result), flush=True)
    raise SystemExit(1 if result["status"] == "error" else 0)


if __name__ == "__main__":
    main()
