"""Pixel-free policy features and constrained multi-label action decoding."""

import math

BUTTONS = (
    "ATTACK",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
)


def selection_notes(
    scores: dict[str, float] | None,
    buttons: list[str],
    action: list[int],
    candidates: list[list[int]],
) -> list[str]:
    if scores is None:
        return ["Uniform random sample from legal combinations."]
    notes = []
    pressed = {button for button, value in zip(buttons, action, strict=True) if value}
    for left, right in (
        ("TURN_LEFT", "TURN_RIGHT"),
        ("MOVE_LEFT", "MOVE_RIGHT"),
        ("MOVE_FORWARD", "MOVE_BACKWARD"),
    ):
        if scores.get(left, 0) > 0.5 and scores.get(right, 0) > 0.5:
            winner = left if left in pressed else right
            loser = right if winner == left else left
            notes.append(f"{loser} off: conflicts with {winner}.")
    if "ATTACK" in buttons:
        attack = buttons.index("ATTACK")
        if scores.get("ATTACK", 0) > 0.5 and not any(
            action[attack] for action in candidates
        ):
            notes.insert(0, "ATTACK off: unavailable in current state.")
    return notes


def serialize_observation(observation: dict) -> str:
    variables = observation["variables"]
    px, py = variables["POSITION_X"], variables["POSITION_Y"]
    width = observation["screen_width"]
    enemies = []
    for obj in observation["visible_objects"]:
        if obj.get("category") != "Monster":
            continue
        x, _, box_width, _ = obj["box"]
        distance = math.hypot(obj["position"][0] - px, obj["position"][1] - py)
        screen_x = (x + box_width / 2 - width / 2) / (width / 2)
        enemies.append((distance, obj["id"], obj["name"], screen_x))
    enemies.sort()
    parts = [
        f"health={variables['HEALTH']:.0f}",
        f"ammo={variables['SELECTED_WEAPON_AMMO']:.0f}",
        f"hit={'yes' if observation['damage_since_observation'] > 0 else 'no'}",
        f"visible_enemies={len(enemies)}",
        f"listed_enemies={min(3, len(enemies))}",
    ]
    parts.extend(
        f"{name} screen_x={screen_x:+.2f} dist={distance:.0f}"
        for distance, _, name, screen_x in enemies[:3]
    )
    return " ".join(parts)


def serialize_categorical_observation(observation: dict) -> str:
    """Controlled perception vocabulary, with no action labels in the input."""
    variables = observation["variables"]
    health = variables["HEALTH"]
    condition = "critical" if health < 20 else ("wounded" if health < 40 else "healthy")
    ammo = "loaded" if variables["SELECTED_WEAPON_AMMO"] > 0 else "empty"
    px, py = variables["POSITION_X"], variables["POSITION_Y"]
    width = observation["screen_width"]
    enemies = []
    for obj in observation["visible_objects"]:
        if obj.get("category") != "Monster":
            continue
        x, _, box_width, _ = obj["box"]
        distance = math.hypot(obj["position"][0] - px, obj["position"][1] - py)
        bearing = (x + box_width / 2 - width / 2) / (width / 2)
        enemies.append((distance, obj["id"], bearing))
    side, proximity = "none", "unknown"
    if enemies:
        distance, _, bearing = min(enemies)
        side = (
            "center" if abs(bearing) <= 0.12 else ("left" if bearing < 0 else "right")
        )
        proximity = (
            "touching"
            if distance < 160
            else (
                "near"
                if distance < 240
                else ("medium" if distance <= 500 else "distant")
            )
        )
    return f"health={condition} ammo={ammo} nearest_enemy={side} range={proximity}"


def decode_scores(
    scores: dict[str, float], buttons: list[str], candidates: list[list[int]]
) -> list[int]:
    probabilities = [float(scores[button]) for button in buttons]
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
        raise ValueError("Policy scores must be finite probabilities between 0 and 1")
    # Maximum-likelihood valid button vector under independent Bernoulli outputs.
    # This resolves opposite directions while respecting state/weapon masks.
    probabilities = [min(1 - 1e-7, max(1e-7, value)) for value in probabilities]
    return max(
        candidates,
        key=lambda action: sum(
            math.log(probability if pressed else 1 - probability)
            for probability, pressed in zip(probabilities, action, strict=True)
        ),
    )
