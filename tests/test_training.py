import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from doom_bert.policy import serialize_categorical_observation
from doom_bert.train import make_dataset


def test_training_splits_hold_out_whole_state_combinations():
    rows = make_dataset()
    groups = {
        split: {r["state"] for r in rows if r["split"] == split}
        for split in ("train", "validation", "test")
    }
    assert len(groups["train"]) == 62
    assert len(groups["validation"]) == len(groups["test"]) == 8
    assert not (
        groups["train"] & groups["validation"]
        or groups["train"] & groups["test"]
        or groups["validation"] & groups["test"]
    )
    assert len({(r["state"], r["instruction"]) for r in rows}) == len(rows)
    assert any(r["split"] == "test" and r["labels"][0] for r in rows)


def test_perception_normalizes_live_numbers_without_action_commands():
    def observe(health, ammo, x, distance):
        return {
            "variables": {
                "HEALTH": health,
                "SELECTED_WEAPON_AMMO": ammo,
                "POSITION_X": 0,
                "POSITION_Y": 0,
            },
            "screen_width": 640,
            "visible_objects": [
                {
                    "id": 1,
                    "category": "Monster",
                    "box": [x, 100, 40, 80],
                    "position": [distance, 0, 0],
                }
            ],
        }

    assert serialize_categorical_observation(
        observe(81, 13, 480, 301)
    ) == serialize_categorical_observation(observe(100, 10, 460, 350))
    assert (
        serialize_categorical_observation(observe(30, 0, 300, 100))
        == "health=wounded ammo=empty nearest_enemy=center range=touching"
    )
    assert "nearest_enemy=left" in serialize_categorical_observation(
        observe(19, 5, 100, 650)
    )
