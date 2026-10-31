"""Client-side action-chunk alignment, independent of any simulator package."""

import json
import os
from pathlib import Path


class ChunkTrace:
    def __init__(self):
        self.run = os.environ.get("STARVLA_TRACE_RUN", "default")
        self.log_path = os.environ.get("STARVLA_TRACE_ALIGNMENT")
        self.episode = -1
        self.reset()

    def reset(self, task_suite=None, task_id=None, episode_idx=None):
        self.episode = self.episode + 1 if episode_idx is None else episode_idx
        self.task_suite = task_suite
        self.task_id = task_id
        self.trace_id = None
        self.source_step = None

    def context(self, task, step, cameras, transforms=None):
        return {
            "run": self.run,
            "task": task,
            "episode": self.episode,
            "episode_idx": self.episode,
            "task_suite": self.task_suite,
            "task_id": self.task_id,
            "env_step": int(step),
            "camera_names": cameras,
            "client_transforms": transforms or [],
            "render_backend": os.environ.get("STARVLA_RENDER_BACKEND", "gpu"),
            "render_gpu": os.environ.get("STARVLA_RENDER_GPU"),
        }

    def received(self, response, step):
        self.trace_id = response.get("trace_id")
        self.source_step = int(step)

    def executed(self, step, query):
        if self.log_path:
            path = Path(self.log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "run": self.run,
                            "episode": self.episode,
                            "episode_idx": self.episode,
                            "task_suite": self.task_suite,
                            "task_id": self.task_id,
                            "env_step": int(step),
                            "action_query": int(query),
                            "source_env_step": self.source_step,
                            "trace_id": self.trace_id,
                        }
                    )
                    + "\n"
                )
