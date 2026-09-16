import json
from itertools import pairwise
from pathlib import Path

import av
import pytest
from PIL import Image

from doom_bert.play import legal_actions, play


@pytest.mark.parametrize("ammo", [0, 10])
def test_action_mask(ammo):
    buttons = ["TURN_LEFT", "TURN_RIGHT", "ATTACK"]
    variables = {
        "HEALTH": 100,
        "DEAD": 0,
        "SELECTED_WEAPON": 3,
        "SELECTED_WEAPON_AMMO": ammo,
    }
    actions = legal_actions(buttons, variables)
    assert [0, 0, 0] in actions
    assert all(not (action[0] and action[1]) for action in actions)
    assert any(action[2] for action in actions) == (ammo > 0)
    variables["DEAD"] = 1
    assert legal_actions(buttons, variables) == [[0, 0, 0]]


def test_live_engine_observations_and_reproducibility(tmp_path: Path):
    runs = []
    for name in ("first", "second"):
        output = tmp_path / name
        summary = play(
            scenario="defend_the_center",
            seconds=3,
            seed=7,
            output=output,
            headless=True,
            realtime=False,
            video=name == "first",
            action_tics=35,
        )
        records = [
            json.loads(line)
            for line in (output / "states.jsonl").read_text().splitlines()
        ]
        observations = [row for row in records if row["type"] == "observation"]
        assert summary["observations"] == 4
        assert [row["second"] for row in observations] == [0, 1, 2, 3]
        assert [
            b["episode_tic"] - a["episode_tic"] for a, b in pairwise(observations)
        ] == [35] * 3
        assert observations[-1]["action"] is None
        for row in observations:
            with Image.open(output / row["screen"]) as frame:
                assert frame.size == (640, 480)
                assert frame.mode == "RGB"
            if row["action"] is not None:
                assert row["action"] in legal_actions(row["buttons"], row["variables"])
        runs.append([(row["action"], row["variables"]) for row in observations])
    assert runs[0] == runs[1]
    with av.open(str(tmp_path / "first" / "replay.mp4")) as video:
        stream = video.streams.video[0]
        assert stream.codec_context.name == "h264"
        assert stream.average_rate == 35
        assert stream.width == 640 and stream.height == 480
        frames = list(video.decode(video=0))
        assert len(frames) == 105
        assert float(stream.duration * stream.time_base) == pytest.approx(3)
        # Confirm real intermediate frames, rather than a 1 fps slideshow in MP4.
        assert len({frame.to_ndarray().tobytes() for frame in frames[:35]}) > 20
    assert not (tmp_path / "second" / "replay.mp4").exists()


def test_episode_restart(tmp_path: Path):
    output = tmp_path / "restart"
    summary = play(
        scenario="basic",
        seconds=12,
        seed=7,
        output=output,
        headless=True,
        realtime=False,
    )
    assert summary["completed_episodes"] >= 1
    assert summary["episodes"] >= 2
    records = [
        json.loads(line) for line in (output / "states.jsonl").read_text().splitlines()
    ]
    assert any(row["type"] == "episode_end" for row in records)
    with av.open(str(output / "replay.mp4")) as video:
        assert sum(1 for _ in video.decode(video=0)) == 12 * 35


def test_existing_output_is_preserved(tmp_path: Path):
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        play(scenario="basic", seconds=1, seed=7, output=tmp_path, headless=True)
    assert sentinel.read_text() == "keep"


def test_uncapped_model_decisions_use_fresh_states(tmp_path: Path, monkeypatch):
    from doom_bert import play as play_module
    from doom_bert.policy import BUTTONS

    def forbidden_sleep(_):
        raise AssertionError("Uncapped inference must never sleep")

    monkeypatch.setattr(play_module.time, "sleep", forbidden_sleep)
    seen_tics = []

    def policy(instruction, observation):
        assert instruction == "turn right"
        assert "screen_buffer" not in observation
        seen_tics.append(observation["episode_tic"])
        return {button: float(button == "TURN_RIGHT") for button in BUTTONS}

    output = tmp_path / "uncapped"
    summary = play(
        scenario="defend_the_center",
        seconds=1,
        seed=7,
        output=output,
        headless=True,
        video=False,
        instruction="turn right",
        policy=policy,
    )
    assert len(seen_tics) == 35
    assert all(b - a == 1 for a, b in pairwise(seen_tics))
    assert summary["decisions"] == 35
    assert summary["observations"] == 36
    assert len(list((output / "frames").glob("*.png"))) == 2
    records = [
        json.loads(line) for line in (output / "states.jsonl").read_text().splitlines()
    ]
    assert all(row["pressed"] == ["TURN_RIGHT"] for row in records[:-1])


def test_partial_action_batch_preserves_video_duration(tmp_path: Path):
    output = tmp_path / "partial"
    summary = play(
        scenario="defend_the_center",
        seconds=1,
        seed=7,
        output=output,
        headless=True,
        action_tics=4,
    )
    assert summary["decisions"] == 9
    assert summary["game_seconds"] == 1
    with av.open(str(output / "replay.mp4")) as video:
        assert sum(1 for _ in video.decode(video=0)) == 35
