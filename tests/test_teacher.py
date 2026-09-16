from doom_bert.play import legal_actions
from doom_bert.policy import BUTTONS, decode_scores
from doom_bert.teacher import TEACHERS, teacher_decision


def observation(*, health=100, ammo=20, bearing=0, distance=300, enemy=True):
    return {
        "variables": {
            "HEALTH": health,
            "SELECTED_WEAPON_AMMO": ammo,
            "SELECTED_WEAPON": 2,
            "DEAD": 0,
            "POSITION_X": 0,
            "POSITION_Y": 0,
        },
        "screen_width": 640,
        "visible_objects": [
            {
                "id": 1,
                "category": "Monster",
                "box": [320 + bearing * 320 - 20, 100, 40, 80],
                "position": [distance, 0, 0],
            }
        ]
        if enemy
        else [],
    }


def selected(state, mode="cautious"):
    scores, _ = teacher_decision(state, TEACHERS[mode])
    vector = decode_scores(
        scores, list(BUTTONS), legal_actions(list(BUTTONS), state["variables"])
    )
    return {button for button, on in zip(BUTTONS, vector, strict=True) if on}


def test_teacher_instruction_contrast_on_the_same_state():
    state = observation()
    assert "ATTACK" in selected(state, "aggressive")
    assert "ATTACK" not in selected(state, "evasive")
    assert "MOVE_BACKWARD" in selected(state, "evasive")


def test_teacher_aims_then_fires_and_retreats_when_needed():
    assert selected(observation(bearing=-0.5)) == {"TURN_LEFT"}
    assert selected(observation(bearing=0.5)) == {"TURN_RIGHT"}
    assert selected(observation()) == {"ATTACK"}
    for state in (observation(health=30), observation(ammo=0)):
        assert selected(state) == {"MOVE_BACKWARD", "MOVE_LEFT"}
    assert "MOVE_BACKWARD" in selected(observation(distance=100))


def test_teacher_searches_without_firing_and_stops_when_dead():
    assert selected(observation(enemy=False)) == {"TURN_RIGHT"}
    assert selected(observation(health=0)) == set()
