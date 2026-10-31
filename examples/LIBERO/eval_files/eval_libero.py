import dataclasses
import faulthandler
import json
import logging
import math
import os
import pathlib
import random
import time

import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from examples.LIBERO.eval_files.model2libero_interface import ModelClient

# Keep Python stacks when MuJoCo/EGL aborts outside the exception handler.
faulthandler.enable()

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    max_tasks: int = -1  # If > 0, limit the number of tasks evaluated (smoke / quick check). -1 = run all.
    task_id: int = -1  # Internal worker selection; -1 uses the ordinary task range.
    isolate_tasks: bool = False  # Give each task its own simulator process / EGL lifetime.
    task_timeout_seconds: int = 3600  # Worker infrastructure timeout; action-step caps define task failure.

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "experiments/libero/logs"  # Path to save videos
    results_out_path: str = ""  # Optional append-and-flush JSONL, created exclusively for this run.
    frame_debug_dir: str = ""  # Opt-in lossless snapshots around policy, render and encoder boundaries.

    seed: int = 7  # Random Seed (for reproducibility)

    pretrained_path: str = ""

    # Dataset key for un-normalization. None = auto (only if model trained on a single dataset).
    unnorm_key: str | None = None

    post_process_action: bool = True

    job_name: str = "test"


def eval_libero(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    if args.isolate_tasks:
        from examples.attention_analysis.task_worker import eval_isolated_tasks

        return eval_isolated_tasks(args)

    # Set random seed
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    # args.video_out_path = f"{date_base}+{args.job_name}"

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client_model = ModelClient(
        host=args.host,
        port=args.port,
        unnorm_key=args.unnorm_key,
    )

    # Optional smoke-test cap (still useful for quick verification with -1 = full run).
    n_eval_tasks = num_tasks_in_suite if args.max_tasks <= 0 else min(args.max_tasks, num_tasks_in_suite)
    task_ids = range(n_eval_tasks)
    if args.task_id >= 0:
        if args.task_id >= num_tasks_in_suite:
            raise ValueError(f"Task ID outside suite: {args.task_id}")
        task_ids = [args.task_id]
    logging.info(f"Evaluating {n_eval_tasks} of {num_tasks_in_suite} tasks (max_tasks={args.max_tasks})")

    results_stream = None
    if args.results_out_path:
        result_path = pathlib.Path(args.results_out_path)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        results_stream = result_path.open("x")
    total_episodes, total_successes, total_errors = 0, 0, 0
    try:
        for task_id in tqdm.tqdm(task_ids):
            task = task_suite.get_task(task_id)
            env, initial_states, setup_error = None, None, None
            try:
                initial_states = task_suite.get_task_init_states(task_id)
                env, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
            except Exception as error:
                logging.exception("Task setup failed: %s/%s", args.task_suite_name, task_id)
                setup_error = error
            task_successes = 0
            try:
                for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
                    row = {
                        "task_suite": args.task_suite_name,
                        "task_id": task_id,
                        "task_name": task.language,
                        "episode_idx": episode_idx,
                        "seed": args.seed,
                        "initial_state_index": episode_idx,
                        "success": None,
                        "status": "error",
                        "execution_steps": 0,
                        "video_path": None,
                    }
                    started = time.monotonic()
                    try:
                        if setup_error is not None:
                            raise RuntimeError(f"Task setup failed: {setup_error}") from setup_error
                        _run_episode(args, env, task.language, initial_states, client_model, max_steps, row)
                        row["status"] = "success" if row["success"] else "failure"
                    except Exception as error:
                        row["error"] = f"{type(error).__name__}: {error}"
                        logging.exception("Episode failed: %s", row)
                        total_errors += 1
                    finally:
                        row["elapsed_seconds"] = time.monotonic() - started
                        if results_stream is not None:
                            results_stream.write(json.dumps(row) + "\n")
                            results_stream.flush()
                    total_episodes += 1
                    if row["status"] == "success":
                        total_successes += 1
                        task_successes += 1
                    logging.info("Episode result: %s", json.dumps(row))
                logging.info("Task %s successes: %s/%s", task_id, task_successes, args.num_trials_per_task)
            finally:
                if env is not None:
                    env.close()
    finally:
        if results_stream is not None:
            results_stream.close()
    logging.info("Total success rate: %s", total_successes / total_episodes if total_episodes else 0)
    logging.info("Total episodes: %s", total_episodes)
    if total_errors:
        raise RuntimeError(f"{total_errors} episode program errors; see structured results and rollout log")


def _run_episode(args, env, description, initial_states, client_model, max_steps, row):
    episode_idx = row["episode_idx"]
    client_model.reset(
        task_description=description,
        task_suite=args.task_suite_name,
        task_id=row["task_id"],
        episode_idx=episode_idx,
    )
    env.reset()
    obs = env.set_init_state(initial_states[episode_idx])
    t, step, done = 0, 0, False
    replay_images, full_actions = [], []
    task_description = description
    audit = None
    if args.frame_debug_dir:
        from examples.attention_analysis.frame_debug import FrameBoundaryAudit, renderer_context_snapshot

        audit = FrameBoundaryAudit(
            pathlib.Path(args.frame_debug_dir) / f"task{row['task_id']:02d}_episode{episode_idx:03d}",
            context_reader=lambda: renderer_context_snapshot(env),
        )
        row["frame_debug_path"] = str(audit.directory.resolve())
    try:
        while t < max_steps + args.num_steps_wait:
            # try:
            # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
            # and we need to wait for them to fall
            if t < args.num_steps_wait:
                obs, _reward, done, _info = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            # IMPORTANT: rotate 180 degrees to match train preprocessing
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

            # Save preprocessed image for replay video
            replay_images.append(img.copy())  # Isolate the encoder buffer from any policy-side preprocessing.
            if audit is not None:
                audit.capture("before_policy", step, {"primary": img, "wrist": wrist_img})

            state = np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            )

            observation = {  #
                "observation.primary": np.expand_dims(img, axis=0),  # (H, W, C), dtype=unit8, range(0-255)
                "observation.wrist_image": np.expand_dims(wrist_img, axis=0),  # (H, W, C)
                "observation.state": np.expand_dims(state, axis=0),
                "instruction": [str(task_description)],
            }

            # align key with model API --> two images provided here --> check training
            example_dict = {
                "image": [observation["observation.primary"][0], observation["observation.wrist_image"][0]],
                "lang": observation["instruction"][0],
                "image_transforms": [
                    {
                        "rotation_degrees": 180,
                        "resize_from_wh": [256, 256],
                        "resize_to_wh": [224, 224],
                        "padding": [0, 0, 0, 0],
                    }
                    for _ in range(2)
                ],
            }

            time.time()

            response = client_model.step(example=example_dict, step=step)
            if audit is not None:
                audit.capture(
                    "after_policy",
                    step,
                    {"primary": img, "wrist": wrist_img},
                    inference_requested=getattr(client_model, "last_inference_requested", None),
                    trace_id=getattr(getattr(client_model, "_trace", None), "trace_id", None),
                )

            time.time()
            # print(f"time: {end_time - start_time}")

            # #
            raw_action = response["raw_action"]

            world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
            rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
            open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
            gripper = _binarize_gripper_open(open_gripper)

            if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                logging.warning(
                    f"Unexpected action sizes: "
                    f"wv={world_vector_delta.shape}, rot={rotation_delta.shape}, grip={gripper.shape}. "
                    f"Falling back to LIBERO_DUMMY_ACTION."
                )
                raise ValueError(
                    f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                    f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                )
            else:
                delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)

            full_actions.append(delta_action)

            # __import__("ipdb").set_trace()
            # see ../robosuite/controllers/controller_factory.py
            obs, _reward, done, _info = env.step(delta_action.tolist())
            if audit is not None:
                audit.capture(
                    "after_env_step",
                    step + 1,
                    {
                        "primary": obs["agentview_image"][::-1, ::-1],
                        "wrist": obs["robot0_eye_in_hand_image"][::-1, ::-1],
                    },
                )
            step += 1
            row["execution_steps"] = step
            t += 1
            if done:
                break
        row["success"] = bool(done)
    finally:
        try:
            if replay_images:
                suffix = "success" if done else "failure"
                task_segment = "_".join(description.split()).replace("/", "_")
                video = pathlib.Path(args.video_out_path) / (
                    f"task{row['task_id']:02d}_{task_segment}_episode{episode_idx}_{suffix}.mp4"
                )
                with imageio.get_writer(video, fps=10) as writer:
                    for frame_index, image in enumerate(replay_images):
                        if audit is not None:
                            audit.capture("before_encode", frame_index, {"primary": image})
                        writer.append_data(np.asarray(image))
                row["video_path"] = str(video.resolve())
        finally:
            if audit is not None:
                row["frame_debug_summary"] = audit.finish()


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    from examples.attention_analysis.egl_cleanup import install_egl_cleanup

    install_egl_cleanup()
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def start_debugpy_once():
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s | %(message)s",
        datefmt="%m/%d [%H:%M:%S]",
        force=True,
    )
    if os.getenv("DEBUG", False):
        start_debugpy_once()
    tyro.cli(eval_libero)
