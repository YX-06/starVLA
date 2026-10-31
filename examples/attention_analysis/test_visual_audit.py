"""Detect the known isolated corruption without treating smooth motion as a fault."""

from examples.attention_analysis.visual_audit import isolated_spikes


def test_periodic_single_frame_corruption():
    deltas = [0] + [1] * 24
    for frame in (1, 9, 17):
        deltas[frame] = deltas[frame + 1] = 65
    assert isolated_spikes(deltas) == [1, 9, 17]


def test_smooth_or_sustained_scene_change_is_not_an_isolated_spike():
    assert isolated_spikes([0, 3, 4, 5, 3, 4]) == []
    assert isolated_spikes([0, 1, 40, 40, 40, 40, 1]) == []
