"""Score zero-shot perception against recorded Doom game state before playing.

Runs the exact ``Perception`` used in gameplay on unique screenshots from earlier
runs and compares every reading with the logged ground truth: monster presence,
bearing, the fire decision, turn direction, HEALTH, AMMO and instruction mode.

    uv run python probe_zero_shot.py --frames 150
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
from gemma_doom import (
    AIM_WINDOW,
    WINDOW_WIDTH,
    Perception,
    control,
    default_dtype,
    load_encoder,
    select_device,
)
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
MONSTERS = {"MarineChainsawVzd", "Demon"}
INSTRUCTIONS = {
    "Attack enemies on sight.": "aggressive",
    "Shoot enemies whenever you can.": "aggressive",
    "Kill every monster you can see.": "aggressive",
    "Shoot at enemies, but retreat when health falls below 40.": "cautious",
    "Fight, but stop shooting and retreat below 40 health.": "cautious",
    "Keep fighting, but back away and hold fire once health drops under 40.": "cautious",
    "Do not shoot. Evade the enemies.": "evasive",
    "Never fire. Keep away from enemies.": "evasive",
    "Stay away from the monsters and don't fire your weapon.": "evasive",
}


def labelled_frames() -> list[dict]:
    """Unique logged screenshots with monster bearings, health and ammo."""
    frames, seen = [], set()
    for states in sorted(REPO.glob("runs/*/states.jsonl")):
        for line in states.open():
            r = json.loads(line)
            if r.get("type") != "observation" or not r.get("screen"):
                continue
            path = states.parent / r["screen"]
            if not path.exists():
                continue
            digest = hashlib.md5(path.read_bytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            v = r["variables"]
            width = r.get("screen_width", 640)
            bearings, nearest = [], None
            for o in r["visible_objects"]:
                if o.get("category", "Monster" if o["name"] in MONSTERS else None) != "Monster":
                    continue
                x, _, w, _ = o["box"]
                bearing = (x + w / 2 - width / 2) / (width / 2)
                bearings.append(bearing)
                d = float(np.hypot(o["position"][0] - v["POSITION_X"], o["position"][1] - v["POSITION_Y"]))
                if nearest is None or d < nearest[0]:
                    nearest = (d, bearing)
            frames.append(
                {
                    "path": str(path),
                    "bearings": bearings,
                    "nearest": None if nearest is None else nearest[1],
                    "health": v["HEALTH"],
                    "ammo": v["SELECTED_WEAPON_AMMO"],
                }
            )
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--vision-tokens", type=int, default=140, choices=[70, 140, 280, 560, 1120])
    parser.add_argument("--window-width", type=int, default=WINDOW_WIDTH)
    parser.add_argument("--output", type=Path, default=REPO / "runs/embeddinggemma2/zero-shot-probe.json")
    args = parser.parse_args()

    device = select_device(args.device)
    perception = Perception(load_encoder(device, default_dtype(device), args.vision_tokens), args.window_width)
    frames = labelled_frames()
    random.Random(0).shuffle(frames)
    frames = frames[: args.frames]

    readings = []
    started = time.perf_counter()
    for n, f in enumerate(frames):
        with Image.open(f["path"]) as image:
            readings.append(perception(image.convert("RGB")))
        if n % 25 == 24:
            print(f"perceived {n + 1}/{len(frames)} frames", flush=True)
    seconds = time.perf_counter() - started

    has = np.array([bool(f["bearings"]) for f in frames])
    visible = np.array([r["visible"] for r in readings])
    in_window = np.array([any(abs(b) <= AIM_WINDOW for b in f["bearings"]) for f in frames])
    estimate = np.array([np.nan if r["bearing"] is None else r["bearing"] for r in readings])
    nearest = np.array([np.nan if f["nearest"] is None else f["nearest"] for f in frames])
    closest = np.array(
        [
            min(abs(e - b) for b in f["bearings"]) if f["bearings"] and not np.isnan(e) else np.nan
            for e, f in zip(estimate, frames, strict=True)
        ]
    )
    fires = np.array(["ATTACK" in control({**r, "health": 100, "ammo": 10}, "aggressive")[0] for r in readings])

    def turn(pressed: set[str]) -> int:
        return 1 if "TURN_RIGHT" in pressed else (-1 if "TURN_LEFT" in pressed else 0)

    policy_turn = np.array([turn(control({**r, "health": 100, "ammo": 10}, "aggressive")[0]) for r in readings])
    # Scripted teacher: turn toward the nearest monster unless lined up; nothing visible -> right.
    teacher_turn = np.where(
        np.isnan(nearest), 1, np.where(np.abs(nearest) <= AIM_WINDOW, 0, np.sign(np.nan_to_num(nearest)))
    )
    hp = np.array([f["health"] for f in frames])
    am = np.array([f["ammo"] for f in frames])
    health = np.array([r["health"] for r in readings])
    ammo = np.array([r["ammo"] for r in readings])
    behaviour = {i: perception.behaviour(i)[0] for i in INSTRUCTIONS}
    tp = int((fires & in_window).sum())
    report = {
        "device": device,
        "vision_tokens": args.vision_tokens,
        "window_width": args.window_width,
        "crops_per_decision": readings[0]["crops"],
        "frames": len(frames),
        "seconds_per_frame": seconds / len(frames),
        "frames_with_monsters": int(has.sum()),
        "presence_accuracy": float((visible == has).mean()),
        "presence_always_yes_baseline": float(has.mean()),
        "bearing_error_to_nearest_median": float(np.nanmedian(np.abs(estimate - nearest)[has & visible])),
        "bearing_error_to_closest_monster_median": float(np.nanmedian(closest[has & visible])),
        "fire_count": int(fires.sum()),
        "fire_precision_monster_in_aim_window": tp / max(int(fires.sum()), 1),
        "fire_recall": tp / max(int(in_window.sum()), 1),
        "frames_with_monster_in_aim_window": int(in_window.sum()),
        "turn_agrees_with_teacher": float((policy_turn == teacher_turn).mean()),
        "turn_opposite_to_teacher": float((policy_turn * teacher_turn == -1).mean()),
        "health_mean_abs_error": float(np.abs(health - hp).mean()),
        "health_below_40_accuracy": float(((health < 40) == (hp < 40)).mean()),
        "health_below_40_frames": int((hp < 40).sum()),
        "ammo_exact": float((ammo == am).mean()),
        "ammo_empty_frames": int((am == 0).sum()),
        "ammo_empty_accuracy": float(((ammo == 0) == (am == 0)).mean()),
        "behaviour_accuracy": float(np.mean([behaviour[i] == m for i, m in INSTRUCTIONS.items()])),
        "behaviour": behaviour,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
