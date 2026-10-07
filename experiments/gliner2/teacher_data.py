"""Generate teacher-labelled GLiNER training data with continuous health and ammo.

The first attempt trained on the 468 controlled pairs, whose text only ever said
health 10, 30 or 100: the 40-health boundary of the cautious instruction could
not be learned from them. Here every row samples health, ammo, enemy side and
range, and the scripted teacher labels all three behaviours.

Held out, so the original evaluation stays honest:
- the original validation and test state combinations (health and ammo
  categories, enemy side and range) never appear in training; validation rows
  are drawn only from the validation combinations, and test combinations are
  never generated;
- the fixed novel test phrasings in compare.py and the new conditional probe
  phrasing in check_conditionals.py are never used.

    uv run python teacher_data.py --output ../../data/gliner2-teacher-v2.jsonl
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

from compare import NOVEL_INSTRUCTIONS, ROOT
from gliner_policy import describe_state

from doom_bert.policy import BUTTONS, serialize_categorical_observation
from doom_bert.teacher import TEACHERS, teacher_decision

TRAIN_INSTRUCTIONS = {
    "aggressive": [
        "Attack enemies on sight.",
        "Shoot enemies whenever you can.",
        "Kill every monster you see.",
        "Fight every enemy that appears.",
        "Destroy the monsters.",
        "Be aggressive and shoot first.",
    ],
    "cautious": [
        "Shoot at enemies, but retreat when health falls below 40.",
        "Fight, but stop shooting and retreat below 40 health.",
        "Attack while your health is 40 or more; below 40, back off and hold fire.",
        "Fight carefully: when health drops under 40, stop firing and retreat.",
        "Keep shooting unless health is below 40, then withdraw without firing.",
    ],
    "evasive": [
        "Do not shoot. Evade the enemies.",
        "Never fire. Keep away from enemies.",
        "Stay away from the monsters and don't shoot.",
        "Run from enemies without firing a shot.",
        "Hold your fire and avoid the monsters.",
    ],
}
HELD_OUT_PROBE = "Fire only when your health is at least 40. If lower, do not fire and move backward."
SIDES = {
    "none": None,
    "left": (-1.0, -0.13),
    "center": (0.0, 0.12),
    "right": (0.13, 1.0),
}
RANGES = {
    "touching": (60, 159),
    "near": (160, 239),
    "medium": (240, 500),
    "distant": (501, 1200),
}


def observation(
    health: int, ammo: int, side: str, range_: str, rng: random.Random
) -> dict:
    objects = []
    if side != "none":
        bearing = rng.uniform(*SIDES[side])
        distance = rng.uniform(*RANGES[range_])
        centre_x = 320 + bearing * 320
        objects.append(
            {
                "id": 1,
                "name": "Enemy",
                "category": "Monster",
                "box": [centre_x - 20, 100, 40, 80],
                "position": [distance, 0, 0],
            }
        )
    return {
        "variables": {
            "HEALTH": float(health),
            "SELECTED_WEAPON_AMMO": float(ammo),
            "SELECTED_WEAPON": 2.0,
            "DEAD": 0.0,
            "POSITION_X": 0.0,
            "POSITION_Y": 0.0,
        },
        "screen_width": 640,
        "damage_since_observation": 0,
        "visible_objects": objects,
    }


def sample_state(rng: random.Random) -> tuple[int, int, str, str]:
    # Half the health values sit near the 20 and 40 boundaries the teacher uses.
    health = rng.randint(1, 100) if rng.random() < 0.5 else rng.randint(12, 48)
    ammo = 0 if rng.random() < 0.25 else rng.randint(1, 50)
    side = rng.choices(list(SIDES), weights=[0.2, 0.25, 0.3, 0.25])[0]
    range_ = rng.choice(list(RANGES)) if side != "none" else "unknown"
    return health, ammo, side, range_


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/gliner2-teacher-v2.jsonl"
    )
    parser.add_argument("--train-states", type=int, default=4000)
    parser.add_argument("--validation-states", type=int, default=400)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    held_out = {i for phrasings in NOVEL_INSTRUCTIONS.values() for i in phrasings} | {
        HELD_OUT_PROBE
    }
    if held_out & {i for phrasings in TRAIN_INSTRUCTIONS.values() for i in phrasings}:
        raise SystemExit("A held-out phrasing leaked into the training phrasings")
    original = [
        json.loads(line) for line in (ROOT / "data/controlled-demo.jsonl").open()
    ]
    combos = {
        split: {r["state"] for r in original if r["split"] == split}
        for split in ("validation", "test")
    }

    rng = random.Random(args.seed)
    rows, counts = [], {"train": 0, "validation": 0}
    targets = {"train": args.train_states, "validation": args.validation_states}
    while (
        counts["train"] < targets["train"]
        or counts["validation"] < targets["validation"]
    ):
        health, ammo, side, range_ = sample_state(rng)
        obs = observation(health, ammo, side, range_, rng)
        combo = serialize_categorical_observation(obs)
        if combo in combos["test"]:
            continue
        split = "validation" if combo in combos["validation"] else "train"
        if counts[split] >= targets[split]:
            continue
        counts[split] += 1
        for mode, spec in TEACHERS.items():
            scores, _ = teacher_decision(obs, spec)
            rows.append(
                {
                    "split": split,
                    "mode": mode,
                    "instruction": rng.choice(TRAIN_INSTRUCTIONS[mode]),
                    "state_text": describe_state(obs),
                    "combo": combo,
                    "health": health,
                    "ammo": ammo,
                    "labels": [int(scores[b] > 0.5) for b in BUTTONS],
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r) + "\n" for r in rows))
    positives = {
        b: sum(r["labels"][i] for r in rows if r["split"] == "train")
        for i, b in enumerate(BUTTONS)
    }
    print(
        json.dumps(
            {
                "rows": len(rows),
                "train_rows": sum(r["split"] == "train" for r in rows),
                "validation_rows": sum(r["split"] == "validation" for r in rows),
                "train_combos": len(
                    {r["combo"] for r in rows if r["split"] == "train"}
                ),
                "train_positive_labels": positives,
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
