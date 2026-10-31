"""Focused checks for Plus selection, expert observation and result integrity."""

import ast
import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from examples.attention_analysis.test_attention import FakeInterface
from examples.attention_analysis_plus.client import unnormalize_actions
from examples.attention_analysis_plus.report import build_report, summarize
from examples.attention_analysis_plus.tasks import CATEGORIES, SUITES, select_tasks, validate_mapping
from starVLA.model.analysis.attention import AttentionRecorder


def fixture_tasks():
    original, runtime, classification = {}, {}, {}
    for suite in SUITES:
        original[suite], runtime[suite], classification[suite] = [], [], []
        for index in range(10):
            base = f"original_{index}"
            original[suite].append(base)
            for category_index, category in enumerate(CATEGORIES):
                for variant in range(3):
                    name = f"{base}_category{category_index}_variant{variant}"
                    runtime[suite].append(name)
                    classification[suite].append(
                        {"name": name, "id": len(runtime[suite]), "category": category, "difficulty_level": variant + 1}
                    )
    return classification, original, runtime


def test_selection_reproducible_with_independent_hash():
    classification, original, runtime = fixture_tasks()
    actual = select_tasks(classification, original, runtime)
    shuffled = {k: v[::-1] for k, v in classification.items()}
    assert actual == select_tasks(shuffled, original, runtime)
    assert len(actual) == 280
    for row in actual:
        candidates = [
            r
            for r in classification[row["task_suite"]]
            if r["name"].startswith(row["original_task"] + "_") and r["category"] == row["category"]
        ]
        expected = min(
            candidates,
            key=lambda r: hashlib.sha256(
                f"7|{row['task_suite']}|{row['original_task']}|{r['category']}|{r['id']}|{r['name']}".encode()
            ).digest(),
        )
        assert expected["id"] == row["official_id"]
        assert runtime[row["task_suite"]][row["task_id"]] == row["variant_name"]
    classification[SUITES[0]].pop()
    classification[SUITES[0]].append(copy.deepcopy(classification[SUITES[0]][0]))
    with pytest.raises(ValueError):
        select_tasks(classification, original, runtime)


def test_runtime_name_mismatch_fails():
    row = {"task_id": 3, "variant_name": "expected"}
    validate_mapping(
        [row], SimpleNamespace(get_task=lambda _: SimpleNamespace(name="expected", bddl_file="expected.bddl"))
    )
    with pytest.raises(ValueError, match="mismatch"):
        validate_mapping([row], SimpleNamespace(get_task=lambda _: SimpleNamespace(name="different")))


class CrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.k_proj = nn.Linear(8, 4)

    def forward(self, x, prefix):
        q = self.q_proj(x).reshape(1, 8, 2, 4).transpose(1, 2)
        k = self.k_proj(prefix).reshape(1, 12, 1, 4).transpose(1, 2).repeat_interleave(2, 1)
        return F.scaled_dot_product_attention(q, k, k).transpose(1, 2).reshape(1, 8, 8)


class MiniV4(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_horizon = 8
        self.qwen_vl_interface = FakeInterface()
        self.expert_layers = nn.ModuleList([nn.ModuleDict({"cross_attn": CrossAttention()}) for _ in range(28)])

    def _embed_timestep(self, t):
        return t

    def predict_action(self, examples):
        self.qwen_vl_interface.build_qwenvl_inputs([examples[0]["image"]], ["instruction"])
        x, prefix = torch.randn(1, 8, 8), torch.randn(1, 12, 8)
        for step in range(4):
            self._embed_timestep(torch.tensor([step / 4]))
            # An unrelated/self-attention SDPA must never be recorded.
            F.scaled_dot_product_attention(x[:, None], x[:, None], x[:, None])
            for layer in self.expert_layers:
                x = layer.cross_attn(x, prefix)
        return x


def test_v4_all_layers_steps_queries_and_request_isolation(tmp_path):
    framework = MiniV4().eval()
    examples = [{"image": [np.zeros((56, 84, 3), dtype=np.uint8), np.zeros((56, 56, 3), dtype=np.uint8)]}]
    recorder = AttentionRecorder({"output_dir": str(tmp_path)})
    torch.manual_seed(7)
    expected = framework.predict_action(examples)
    state = torch.get_rng_state().clone()
    method = framework._embed_timestep
    traces = []
    for task_id in (0, 123):
        torch.manual_seed(7)
        with recorder.session(
            framework, examples, {"task_id": task_id, "camera_names": ["primary", "wrist"]}
        ) as session:
            actual = framework.predict_action(examples)
            traces.append(session.save())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert torch.equal(state, torch.get_rng_state())
        assert framework._embed_timestep == method
        assert not session.records
        with np.load(tmp_path / traces[-1] / "attention.npz") as arrays:
            assert arrays["camera_0"].shape == (4, 28, 8, 2, 3)
            assert arrays["camera_1"].shape == (4, 28, 8, 2, 2)
        metadata = json.loads((tmp_path / traces[-1] / "metadata.json").read_text())
        assert metadata["timesteps"] == {"0": [0.0], "1": [0.25], "2": [0.5], "3": [0.75]}
        assert metadata["trace_context"]["task_id"] == task_id
    assert traces[0] != traces[1]
    with pytest.raises(RuntimeError), recorder.session(framework, examples):
        raise RuntimeError("failure")
    assert framework._embed_timestep == method
    assert not any(layer.cross_attn._forward_hooks for layer in framework.expert_layers)


def test_normal_failures_errors_missing_and_duplicate_summary():
    selected = select_tasks(*fixture_tasks())
    episodes = [{**row, "status": "success"} for row in selected]
    episodes[0]["status"] = "failure"
    episodes[1]["status"] = "error"
    result = summarize(episodes[:-1], selected)
    assert result["successes"] == 277 and result["normal_failures"] == result["program_errors"] == result["missing"] == 1
    assert all(r["expected"] == 70 for r in result["by_suite"].values())
    assert all(r["expected"] == 40 for r in result["by_category"].values())
    with pytest.raises(ValueError, match="Duplicate"):
        summarize(episodes + episodes[:1], selected)


def test_normalization_matches_published_client():
    from pathlib import Path

    source = Path(
        "playground/attention_analysis_plus/third_party/VLAct/examples/LIBERO-plus/eval_files/model2libero_interface.py"
    )
    if not source.exists():
        pytest.skip("Pinned VLAct source is not prepared")
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ModelClient")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "unnormalize_actions")
    method.decorator_list = []
    namespace = {"np": np, "Dict": dict}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    values = np.array([[-2, -0.7, 0.5, 0.0, 0.9, 2, 0.49], [1, 0.0, -1, 1, -1, 0, 0.5]])
    stats = {"min": [-1] * 6 + [0], "max": [3] * 6 + [1], "mask": [True] * 6 + [False]}
    np.testing.assert_array_equal(unnormalize_actions(values, stats), namespace["unnormalize_actions"](values, stats))


def test_report_uses_episode_suite_instead_of_last_scanned_suite(tmp_path, monkeypatch):
    from examples.attention_analysis_plus import report
    from examples.attention_analysis_plus.run import task_directory

    selected = select_tasks(*fixture_tasks())
    root = tmp_path / "experiment"
    checkpoint = root / "checkpoints/libero_plus_pi"
    checkpoint.mkdir(parents=True)
    (checkpoint / "dataset_statistics.json").write_text(json.dumps({"franka": {"action": {}}}))
    output = root / "runs/smoke"
    suite_run = output / "libero_spatial"
    suite_run.mkdir(parents=True)
    (output / "status.json").write_text(json.dumps({"smoke": True}))
    row = {
        **selected[0],
        "status": "failure",
        "success": False,
        "seed": 7,
        "initial_state_index": 0,
        "episode_idx": 0,
        "execution_steps": 220,
    }
    (suite_run / "episodes.jsonl").write_text(json.dumps(row) + "\n")
    (suite_run / task_directory(row)).mkdir()
    monkeypatch.setenv("PLUS_ROOT", str(root))
    monkeypatch.setattr(report, "load_manifest", lambda _: {"tasks": selected})
    seen = []

    def validate(task, suite, episode, stats):
        seen.append(suite)
        assert suite == suite_run
        return {"trace_ids": [], "requests": [], "camera_quality": {}}

    monkeypatch.setattr(report, "validate_task", validate)
    result = build_report(output, tmp_path / "manifest.json")
    assert seen == [suite_run]
    assert result["structural_validation_passed"] and result["normal_failures"] == 1
