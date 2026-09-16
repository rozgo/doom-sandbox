from fractions import Fraction
from threading import Event

import av
import numpy as np
import pytest

from doom_bert.overlay import LiveMetrics, RenderWorker
from doom_bert.play import legal_actions
from doom_bert.policy import decode_scores, selection_notes
from doom_bert.video import VideoRecorder


def test_throughput_counts_intervals_between_completed_decisions():
    metrics = LiveMetrics()
    assert metrics.update(10, 0.01)["rolling_decisions_per_second"] is None
    metrics.update(12, 0.04)
    metrics.update(15, 0.08)
    result = metrics.update(20, 0.11)
    assert result["rolling_decisions_per_second"] == pytest.approx(30)
    assert result["decision_p50_ms"] == 13.5
    assert result["decision_p95_ms"] == 20


def test_selection_explains_conflicts_and_state_mask():
    buttons = ["TURN_LEFT", "TURN_RIGHT", "ATTACK"]
    variables = {
        "HEALTH": 100,
        "DEAD": 0,
        "SELECTED_WEAPON": 3,
        "SELECTED_WEAPON_AMMO": 0,
    }
    candidates = legal_actions(buttons, variables)
    scores = {"TURN_LEFT": 0.7, "TURN_RIGHT": 0.9, "ATTACK": 0.99}
    action = decode_scores(scores, buttons, candidates)
    notes = selection_notes(scores, buttons, action, candidates)
    assert "ATTACK off: unavailable in current state." in notes
    assert "TURN_LEFT off: conflicts with TURN_RIGHT." in notes


def test_wall_clock_video_preserves_capture_timestamps(tmp_path):
    output = tmp_path / "wall.mp4"
    recorder = VideoRecorder(output, width=64, height=64, fps=35, clock="wall")
    timestamps = [0, 0.04, 0.13, 0.20]
    for index, timestamp in enumerate(timestamps):
        recorder.write(
            np.full((64, 64, 3), index * 50, dtype=np.uint8), elapsed_seconds=timestamp
        )
    recorder.close()
    with av.open(str(output)) as video:
        frames = list(video.decode(video=0))
    assert len(frames) == 4
    assert [float(frame.pts * frame.time_base) for frame in frames] == pytest.approx(
        timestamps, abs=0.001
    )
    assert frames[-1].time_base != Fraction(1, 35)


def test_rendering_does_not_block_decisions_and_discards_stale_frames():
    started = Event()
    release = Event()
    written = []

    class SlowRecorder:
        def write(self, screen, *, elapsed_seconds):
            started.set()
            assert release.wait(timeout=5)
            written.append(elapsed_seconds)

    worker = RenderWorker(SlowRecorder(), None)
    screen = np.zeros((4, 4, 3), dtype=np.uint8)
    try:
        worker.submit(screen, {}, 0)
        assert started.wait(timeout=5)
        # These return while encoding remains blocked. Only the newest is kept.
        worker.submit(screen, {}, 0.04)
        worker.submit(screen, {}, 0.08)
    finally:
        release.set()
        worker.close()
    assert written == [0, 0.08]
    assert worker.dropped_frames == 1
