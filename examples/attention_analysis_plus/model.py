"""Load the published VLAct framework independently of this checkout's models."""

import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


def recorder_type():
    source = Path(__file__).resolve().parents[2] / "starVLA/model/analysis/attention.py"
    spec = importlib.util.spec_from_file_location("plus_attention_recorder", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AttentionRecorder


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_framework(checkpoint):
    from accelerate import PartialState

    root = Path(os.environ["PLUS_ROOT"])
    vlact = root / "third_party/VLAct"
    sys.path.insert(0, str(vlact))
    PartialState()
    from starVLA.model.framework.base_framework import baseframework

    if not Path(sys.modules["starVLA"].__file__).resolve().is_relative_to(vlact):
        raise RuntimeError("Another StarVLA package was imported before the isolated VLAct loader")
    return baseframework.from_pretrained(str(checkpoint)).to("cuda").eval()


class RecordedPolicy:
    def __init__(self, checkpoint, output):
        self.framework = load_framework(checkpoint)
        provenance = json.loads((Path(checkpoint).parents[1] / "revision.json").read_text())
        self.recorder = recorder_type()({"output_dir": str(output), "interval": 1}, provenance=provenance)

    def predict_action(self, examples, trace_context=None, **kwargs):
        if int(kwargs.get("diffusion_samples", 1)) != 1:
            raise ValueError("Export requires diffusion_samples=1")
        with self.recorder.session(self.framework, examples, trace_context) as session:
            result = self.framework.predict_action(examples=examples, **kwargs)
            trace_id = session.save()
        return {**result, "trace_id": trace_id}
