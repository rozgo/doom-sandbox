"""Small, explicit gameplay rules that can also supply imitation labels."""

import math
from dataclasses import dataclass

from doom_bert.policy import BUTTONS


@dataclass(frozen=True)
class TeacherSpec:
    instruction: str
    allow_shoot: bool
    retreat_hp: float
    personal_space: float


TEACHERS = {
    "aggressive": TeacherSpec("Attack enemies on sight.", True, 20, 160),
    "cautious": TeacherSpec(
        "Shoot at enemies, but retreat when health falls below 40.", True, 40, 240
    ),
    "evasive": TeacherSpec("Do not shoot. Evade the enemies.", False, 101, 400),
}


def teacher_decision(observation: dict, spec: TeacherSpec) -> tuple[dict, list[str]]:
    """Return illustrative 0.02/0.98 rule scores, not learned probabilities."""
    variables = observation["variables"]
    pressed = set()
    reasons = []
    if variables["HEALTH"] <= 0 or variables.get("DEAD", 0):
        return dict.fromkeys(BUTTONS, 0.02), ["Player is dead: release all buttons."]

    px, py = variables["POSITION_X"], variables["POSITION_Y"]
    width = observation["screen_width"]
    enemies = []
    for obj in observation["visible_objects"]:
        if obj.get("category") != "Monster":
            continue
        x, _, box_width, _ = obj["box"]
        bearing = (x + box_width / 2 - width / 2) / (width / 2)
        distance = math.hypot(obj["position"][0] - px, obj["position"][1] - py)
        enemies.append((distance, obj["id"], bearing))
    if not enemies:
        pressed.add("TURN_RIGHT")
        reasons.append("No visible enemy: turn to search.")
    else:
        distance, _, bearing = min(enemies)
        aligned = abs(bearing) <= 0.12
        if not aligned:
            pressed.add("TURN_RIGHT" if bearing > 0 else "TURN_LEFT")
            reasons.append("Turn toward the nearest visible enemy.")
        has_ammo = variables["SELECTED_WEAPON_AMMO"] > 0
        hurt = variables["HEALTH"] < spec.retreat_hp
        retreat = hurt or not has_ammo or distance < spec.personal_space
        if retreat:
            pressed.add("MOVE_BACKWARD")
            pressed.add("MOVE_LEFT" if bearing >= 0 else "MOVE_RIGHT")
            reasons.append("Back off and strafe: hurt, crowded, or no ammo.")
        elif aligned and distance > 500:
            pressed.add("MOVE_FORWARD")
            reasons.append("Approach the distant target.")
        if spec.allow_shoot and has_ammo and aligned and not hurt:
            pressed.add("ATTACK")
            reasons.append("Target lined up: fire.")
        elif not spec.allow_shoot:
            reasons.append("Evasive rule: never fire.")
    return {button: 0.98 if button in pressed else 0.02 for button in BUTTONS}, reasons
