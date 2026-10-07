"""Zero-shot command choice from one multimodal situation embedding.

Each situation interleaves the instruction, the game state as text and view
image(s) into one input. The command whose text prompt has the highest cosine
similarity wins. Ground truth is the scripted teacher's action for the same
logged state, reduced to the same five commands. Variants isolate what each
modality contributes.

    uv run python probe_commands.py --frames 100
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from gemma_doom import BAND_BOTTOM, BAND_TOP, STATUS_BAR_Y, THIRDS, default_dtype, load_encoder, select_device
from PIL import Image
from probe_zero_shot import MONSTERS, labelled_frames

from doom_bert.teacher import TEACHERS, teacher_decision

REPO = Path(__file__).resolve().parents[2]
COMMANDS = {
    "fire": "shoot the monster straight ahead",
    "turn_left": "turn left toward the monster on the left",
    "turn_right": "turn right toward the monster on the right",
    "back_away": "back away from the monster",
    "search": "no monster in sight: turn around to search",
}
VARIANTS = ("image", "image + state text", "state text", "3 labelled images + state text")


def teacher_command(record: dict, mode: str) -> str:
    scores, _ = teacher_decision(record, TEACHERS[mode])
    pressed = {b for b, s in scores.items() if s > 0.5}
    if "ATTACK" in pressed:
        return "fire"
    if "MOVE_BACKWARD" in pressed:
        return "back_away"
    if not any(o.get("category") == "Monster" for o in record["visible_objects"]):
        return "search"
    return "turn_left" if "TURN_LEFT" in pressed else ("turn_right" if "TURN_RIGHT" in pressed else "fire")


def situation(variant: str, instruction: str, health: float, ammo: float, frame: Image.Image) -> dict | str:
    state = f"Player health {health:.0f}%, ammo {ammo:.0f}."
    head = f"task: classification | query: Doom game. Instruction: {instruction}"
    if variant == "image":
        return {"text": f"{head} View: <|image|>", "image": [frame.crop((0, 0, 640, STATUS_BAR_Y))]}
    if variant == "image + state text":
        return {"text": f"{head} {state} View: <|image|>", "image": [frame.crop((0, 0, 640, STATUS_BAR_Y))]}
    if variant == "state text":
        return f"{head} {state}"
    thirds = [frame.crop((a, BAND_TOP, z, BAND_BOTTOM)) for a, z in THIRDS.values()]
    return {"text": f"{head} {state} Left: <|image|> Front: <|image|> Right: <|image|>", "image": thirds}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--vision-tokens", type=int, default=140, choices=[70, 140, 280, 560, 1120])
    parser.add_argument("--output", type=Path, default=REPO / "runs/embeddinggemma2/command-probe.json")
    args = parser.parse_args()

    device = select_device(args.device)
    model = load_encoder(device, default_dtype(device), args.vision_tokens)
    names = list(COMMANDS)
    commands = model.encode(list(COMMANDS.values()), prompt_name="SearchQuery", convert_to_numpy=True)

    # Re-read the full logged records so the teacher sees the exact state.
    records = {}
    for states in REPO.glob("runs/*/states.jsonl"):
        for line in states.open():
            r = json.loads(line)
            if r.get("type") == "observation" and r.get("screen"):
                for o in r["visible_objects"]:
                    o.setdefault("category", "Monster" if o["name"] in MONSTERS else None)
                r.setdefault("screen_width", 640)
                records[str(states.parent / r["screen"])] = r
    frames = labelled_frames()
    random.Random(0).shuffle(frames)
    frames = frames[: args.frames]

    rows = []
    for n, f in enumerate(frames):
        record = records[f["path"]]
        with Image.open(f["path"]) as image:
            frame = image.convert("RGB")
        for mode, spec in TEACHERS.items():
            truth = teacher_command(record, mode)
            inputs = [situation(v, spec.instruction, f["health"], f["ammo"], frame) for v in VARIANTS]
            emb = np.stack([model.encode(x, convert_to_numpy=True) for x in inputs])
            sims = emb @ commands.T
            rows.append(
                {
                    "mode": mode,
                    "truth": truth,
                    **{v: names[int(s.argmax())] for v, s in zip(VARIANTS, sims, strict=True)},
                }
            )
        if n % 20 == 19:
            print(f"{n + 1}/{len(frames)} frames", flush=True)

    truth_counts = Counter(r["truth"] for r in rows)
    report = {
        "frames": len(frames),
        "situations": len(rows),
        "commands": COMMANDS,
        "teacher_command_counts": dict(truth_counts),
        "majority_baseline": max(truth_counts.values()) / len(rows),
        "variants": {},
    }
    for v in VARIANTS:
        report["variants"][v] = {
            "accuracy": float(np.mean([r[v] == r["truth"] for r in rows])),
            "by_mode": {m: float(np.mean([r[v] == r["truth"] for r in rows if r["mode"] == m])) for m in TEACHERS},
            "predicted_counts": dict(Counter(r[v] for r in rows)),
            "evasive_fires": int(sum(r[v] == "fire" for r in rows if r["mode"] == "evasive")),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(rows)} situations; teacher commands {dict(truth_counts)}; majority {report['majority_baseline']:.1%}")
    for v, r in report["variants"].items():
        print(
            f"{v:32s} accuracy {r['accuracy']:.1%}  by mode { ({m: f'{a:.0%}' for m, a in r['by_mode'].items()}) }  "
            f"fires when told not to {r['evasive_fires']}  predicted {r['predicted_counts']}"
        )


if __name__ == "__main__":
    main()
