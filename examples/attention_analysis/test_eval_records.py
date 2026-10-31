"""Simulator-adapter regressions; run in the existing libero-attn environment."""

import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("robosuite")
from examples.LIBERO.eval_files import eval_libero as evaluator


@pytest.mark.parametrize("done_at", [1, 2, None])
def test_episode_counts_successful_final_step(tmp_path, monkeypatch, done_at):
    frames, calls, resets = [], [], []
    observation = {
        "agentview_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((256, 256, 3), dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.zeros(2),
    }

    class Env:
        count = 0

        def reset(self):
            pass

        def set_init_state(self, state):
            assert state == "initial0"
            return observation

        def step(self, action):
            self.count += 1
            return observation, 0, self.count == done_at, {}

    class Client:
        def reset(self, **kwargs):
            resets.append(kwargs)

        def step(self, example, step):
            calls.append(step)
            assert len(example["image"]) == 2
            return {"raw_action": {"world_vector": np.zeros(3), "rotation_delta": np.zeros(3), "open_gripper": [1]}}

    class Writer:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def append_data(self, image):
            frames.append(image)

    monkeypatch.setattr(evaluator.imageio, "get_writer", lambda *args, **kwargs: Writer())
    args = evaluator.Args(task_suite_name="libero_object", num_steps_wait=0, video_out_path=str(tmp_path))
    row = {"task_id": 3, "episode_idx": 0, "execution_steps": 0}
    evaluator._run_episode(args, Env(), "task", ["initial0"], Client(), 2, row)
    expected = done_at or 2
    assert row["execution_steps"] == len(frames) == len(calls) == expected
    assert row["success"] == (done_at is not None)
    assert calls == list(range(expected))
    assert resets == [{"task_description": "task", "task_suite": "libero_object", "task_id": 3, "episode_idx": 0}]


def test_structured_results_continue_after_episode_error(tmp_path, monkeypatch):
    closed, visited = [], []

    class Suite:
        n_tasks = 2

        def get_task(self, task_id):
            return SimpleNamespace(language=f"task{task_id}")

        def get_task_init_states(self, task_id):
            return [task_id]

    monkeypatch.setattr(evaluator.benchmark, "get_benchmark_dict", lambda: {"libero_goal": Suite})
    monkeypatch.setattr(evaluator, "ModelClient", lambda **kwargs: object())
    monkeypatch.setattr(
        evaluator,
        "_get_libero_env",
        lambda task, *args: (SimpleNamespace(close=lambda: closed.append(task.language)), task.language),
    )

    def run_episode(args, env, description, states, client, max_steps, row):
        visited.append(row["task_id"])
        if row["task_id"] == 0:
            raise ValueError("intentional episode error")
        row.update(success=True, execution_steps=5)

    monkeypatch.setattr(evaluator, "_run_episode", run_episode)
    output = tmp_path / "episodes.jsonl"
    args = evaluator.Args(num_trials_per_task=1, video_out_path=str(tmp_path), results_out_path=str(output))
    with pytest.raises(RuntimeError, match="1 episode program errors"):
        evaluator.eval_libero(args)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert visited == [0, 1] and closed == ["task0", "task1"]
    assert [row["status"] for row in rows] == ["error", "success"]
    assert [row["success"] for row in rows] == [None, True]
    assert all(row["seed"] == 7 and row["initial_state_index"] == 0 and row["elapsed_seconds"] >= 0 for row in rows)
    with pytest.raises(FileExistsError):
        evaluator.eval_libero(args)


@pytest.mark.parametrize("failure_mode", ["abort", "timeout"])
def test_isolated_workers_continue_after_native_abort(tmp_path, monkeypatch, failure_mode):
    from examples.attention_analysis.task_worker import eval_isolated_tasks

    class Suite:
        n_tasks = 2

        def get_task(self, task_id):
            return SimpleNamespace(language=f"task{task_id}")

    monkeypatch.setattr(evaluator.benchmark, "get_benchmark_dict", lambda: {"libero_goal": Suite})
    visited = []

    def run(command, **kwargs):
        config = json.loads(open(command[-1]).read())
        task_id = config["task_id"]
        visited.append(task_id)
        assert not config["isolate_tasks"] and config["seed"] == 7
        assert kwargs["timeout"] == config["task_timeout_seconds"] == 3600
        if task_id == 0:
            if failure_mode == "timeout":
                raise subprocess.TimeoutExpired(command, kwargs["timeout"])
            return subprocess.CompletedProcess(command, -6)
        row = {
            "task_suite": "libero_goal",
            "task_id": task_id,
            "episode_idx": 0,
            "seed": 7,
            "initial_state_index": 0,
            "status": "success",
            "success": True,
        }
        with open(config["results_out_path"], "w") as stream:
            stream.write(json.dumps(row) + "\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("examples.attention_analysis.task_worker.subprocess.run", run)
    output = tmp_path / "isolated.jsonl"
    args = evaluator.Args(num_trials_per_task=1, isolate_tasks=True, results_out_path=str(output))
    with pytest.raises(RuntimeError, match="1 isolated episode program errors"):
        eval_isolated_tasks(args)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert visited == [0, 1]
    assert [row["status"] for row in rows] == ["error", "success"]
    if failure_mode == "abort":
        assert rows[0]["worker_exit_code"] == -6 and "exited with -6" in rows[0]["error"]
    else:
        assert rows[0]["worker_exit_code"] is None and "TimeoutExpired" in rows[0]["error"]
    assert rows[1]["worker_exit_code"] == 0


@pytest.mark.parametrize("mutate_request", [False, True])
def test_policy_boundaries_keep_video_snapshots_and_expose_mutation(tmp_path, monkeypatch, mutate_request):
    frames, queries = [], []
    observation = {
        "agentview_image": np.full((256, 256, 3), 20, dtype=np.uint8),
        "robot0_eye_in_hand_image": np.full((256, 256, 3), 30, dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.zeros(2),
    }

    class Env:
        count = 0

        def reset(self):
            pass

        def set_init_state(self, state):
            return observation

        def step(self, action):
            self.count += 1
            # Deliberately reuse the observation buffer across steps.
            observation["agentview_image"][:] = 20 + self.count
            observation["robot0_eye_in_hand_image"][:] = 30 + self.count
            return observation, 0, False, {}

    class Client:
        def reset(self, **kwargs):
            pass

        def step(self, example, step):
            self.last_inference_requested = step % 8 == 0
            if self.last_inference_requested:
                queries.append(step)
                if mutate_request:
                    example["image"][0][:] = 0
            return {"raw_action": {"world_vector": np.zeros(3), "rotation_delta": np.zeros(3), "open_gripper": [1]}}

    class Writer:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def append_data(self, image):
            frames.append(image.copy())

    monkeypatch.setattr(evaluator.imageio, "get_writer", lambda *args, **kwargs: Writer())
    args = evaluator.Args(num_steps_wait=0, video_out_path=str(tmp_path), frame_debug_dir=str(tmp_path / "debug"))
    row = {"task_id": 0, "episode_idx": 0, "execution_steps": 0}
    env = Env()
    evaluator._run_episode(args, env, "task", [0], Client(), 17, row)
    assert queries == [0, 8, 16] and env.count == row["execution_steps"] == len(frames) == 17
    assert [int(frame[0, 0, 0]) for frame in frames] == list(range(20, 37))
    changes = row["frame_debug_summary"]["buffer_changes"]
    assert [r["env_step"] for r in changes] == ([0, 8, 16] if mutate_request else [])
    assert all(r["to"] == "after_policy" for r in changes)
    assert row["frame_debug_summary"]["buffer_integrity"] == ("failed" if mutate_request else "passed")
