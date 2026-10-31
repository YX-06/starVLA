"""Published Plus action contract with explicit per-chunk provenance."""

import json
from pathlib import Path

import cv2
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


def unnormalize_actions(normalized, stats):
    # Exact math from the pinned VLAct Plus ModelClient (min/max, threshold 0.5).
    mask = stats.get("mask", np.ones_like(stats["min"], dtype=bool))
    high, low = np.array(stats["max"]), np.array(stats["min"])
    values = np.clip(normalized, -1, 1)
    values[:, 6] = np.where(values[:, 6] < 0.5, 0, 1)
    return np.where(mask, 0.5 * (values + 1) * (high - low) + low, values)


class ModelClient:
    def __init__(self, checkpoint, port, run, selection):
        self.client = WebsocketClientPolicy("127.0.0.1", port)
        statistics = json.loads((Path(checkpoint).parents[1] / "dataset_statistics.json").read_text())
        if list(statistics) != ["franka"]:
            raise ValueError(f"Unexpected action statistics keys: {list(statistics)}")
        self.stats = statistics["franka"]["action"]
        self.run, self.selection = Path(run), selection
        self.run_name = f"{self.run.parent.parent.name}/{self.run.parent.name}"
        self.chunks = self.run / "chunks"
        self.chunks.mkdir(exist_ok=False)
        self.chunk = None
        self.trace_id = None

    def action(self, images, instruction, step):
        if step % 8 == 0:
            context = {
                "run": self.run_name,
                "task_suite": self.selection["task_suite"],
                "task_id": self.selection["task_id"],
                "episode": 0,
                "episode_idx": 0,
                "env_step": step,
                "camera_names": ["primary", "wrist"],
                "selection": self.selection,
                "client_transforms": (
                    [
                        {
                            "rotation_degrees": 180,
                            "resize_from_wh": [256, 256],
                            "resize_to_wh": [224, 224],
                            "padding": [0, 0, 0, 0],
                        }
                    ]
                    * 2
                ),
            }
            response = self.client.predict_action(
                {
                    "examples": [
                        {
                            "image": [cv2.resize(x, (224, 224), interpolation=cv2.INTER_AREA) for x in images],
                            "lang": instruction,
                        }
                    ],
                    "do_sample": False,
                    "use_ddim": True,
                    "num_ddim_steps": 10,
                    "trace_context": context,
                }
            )
            if not response.get("ok"):
                raise RuntimeError(f"Policy error: {response}")
            result = response["data"]
            normalized = np.asarray(result["normalized_actions"])
            if normalized.shape != (1, 8, 7) or not np.isfinite(normalized).all():
                raise ValueError(f"Invalid action chunk: {normalized.shape}")
            self.chunk = unnormalize_actions(normalized[0], self.stats)
            self.trace_id = result["trace_id"]
            np.savez_compressed(self.chunks / f"{self.trace_id}.npz", normalized=normalized[0], actions=self.chunk)
        return self.chunk[step % 8].copy(), {
            "run": self.run_name,
            "task_suite": self.selection["task_suite"],
            "task_id": self.selection["task_id"],
            "episode": 0,
            "episode_idx": 0,
            "env_step": step,
            "action_query": step % 8,
            "source_env_step": step - step % 8,
            "trace_id": self.trace_id,
        }
