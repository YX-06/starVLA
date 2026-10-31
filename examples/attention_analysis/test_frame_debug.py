"""CPU checks for evidence at the observation/policy/encoder boundaries."""

import json

import numpy as np
import pytest
from PIL import Image

from examples.attention_analysis.frame_debug import FrameBoundaryAudit, renderer_context_snapshot


def test_snapshot_detects_policy_mutation_and_preserves_earlier_pixels(tmp_path):
    source = np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3)[::-1, ::-1]
    original = source.copy()
    replay = source.copy()
    audit = FrameBoundaryAudit(tmp_path / "frames", context_reader=lambda: {"current_context": 7})
    audit.capture("before_policy", 0, {"primary": source})
    source[:] = 0  # Simulate an unexpected policy-side write to its input.
    audit.capture("after_policy", 0, {"primary": source}, inference_requested=True)
    audit.capture("before_encode", 0, {"primary": replay})
    result = audit.finish()
    assert result["buffer_changes"] == [
        {"env_step": 0, "camera": "primary", "from": "before_policy", "to": "after_policy"}
    ]
    np.testing.assert_array_equal(Image.open(audit.directory / "step000000_before_policy_primary.png"), original)
    events = [json.loads(line) for line in (audit.directory / "events.jsonl").read_text().splitlines()]
    assert events[0]["images"][0]["c_contiguous"] is False
    assert events[1]["inference_requested"] is True
    assert {row["context"]["current_context"] for row in events} == {7}


def test_readback_to_encode_pipeline_hashes_match_without_random_draws(tmp_path):
    before = np.random.get_state()
    audit = FrameBoundaryAudit(tmp_path / "frames")
    for step in range(3):
        images = {name: np.full((16, 16, 3), step + i, dtype=np.uint8) for i, name in enumerate(("primary", "wrist"))}
        audit.capture("after_env_step", step, images)
        audit.capture("before_policy", step, images)
        audit.capture("after_policy", step, images, inference_requested=step == 0)
        audit.capture("before_encode", step, {"primary": images["primary"].copy()})
    result = audit.finish()
    assert result["buffer_integrity"] == "passed" and result["hash_comparisons"] == 15
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    with pytest.raises(FileExistsError):
        FrameBoundaryAudit(audit.directory)


def test_fake_renderer_diagnostics_never_import_egl(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("MUJOCO_GL", "egl")
    assert renderer_context_snapshot(SimpleNamespace())["owner_context"] is None


def test_empty_audit_is_not_a_pass_and_duplicate_capture_cannot_overwrite_evidence(tmp_path):
    audit = FrameBoundaryAudit(tmp_path / "frames")
    assert audit.finish()["buffer_integrity"] == "not_checked"
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    audit.capture("before_policy", 0, {"primary": image})
    with pytest.raises(ValueError, match="Duplicate"):
        audit.capture("before_policy", 0, {"primary": image})
