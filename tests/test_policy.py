import pytest

from doom_bert.play import legal_actions
from doom_bert.policy import decode_scores, serialize_observation


def test_scores_resolve_opposite_directions_and_ammo():
    buttons = ["TURN_LEFT", "TURN_RIGHT", "ATTACK"]
    variables = {
        "HEALTH": 100,
        "DEAD": 0,
        "SELECTED_WEAPON": 3,
        "SELECTED_WEAPON_AMMO": 0,
    }
    scores = {"TURN_LEFT": 0.7, "TURN_RIGHT": 0.9, "ATTACK": 0.99}
    assert decode_scores(scores, buttons, legal_actions(buttons, variables)) == [
        0,
        1,
        0,
    ]
    scores["ATTACK"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        decode_scores(scores, buttons, legal_actions(buttons, variables))


def test_serialization_filters_and_sorts_objects():
    def obj(index, category, distance):
        return {
            "id": index,
            "category": category,
            "name": f"object{index}",
            "position": [distance, 0, 0],
            "box": [300, 100, 40, 50],
        }

    observation = {
        "variables": {
            "HEALTH": 30,
            "SELECTED_WEAPON_AMMO": 12,
            "POSITION_X": 0,
            "POSITION_Y": 0,
        },
        "damage_since_observation": 5,
        "screen_width": 640,
        "visible_objects": [
            obj(0, "Self", 0),
            obj(1, "Weapon", 2),
            obj(2, "Monster", 100),
            obj(3, "Monster", 20),
        ],
    }
    text = serialize_observation(observation)
    assert "hit=yes" in text
    assert "visible_enemies=2" in text
    assert "object0" not in text and "object1" not in text
    assert text.index("object3") < text.index("object2")
    assert "screen_x=+0.00" in text
