"""Cut the EmbeddingGemma 2 showcase: intro, one chapter per instruction, results.

Each chapter is a complete recorded run (dashboard replay at game speed) behind
a title card. Numbers on the cards come from the runs' summary.json and the
perception probe report, never typed in by hand.

    uv run python edit_showcase.py --runs ../../runs/eg2-show-attack ../../runs/eg2-show-evade \\
        --probe ../../runs/embeddinggemma2/zero-shot-probe-gpu-v2.json --output ../../runs/eg2-showcase.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import textwrap
from pathlib import Path

from gemma_overlay import draw_eye, draw_hash
from PIL import Image, ImageDraw

from doom_bert.overlay import ACCENT, AMBER, BG, EDGE, MUTED, PANEL, TEXT, font

W, H, FPS = 1440, 900, 35
TITLE = "EmbeddingGemma 2 plays Doom, zero-shot"
MODE_TITLES = {"aggressive": "ATTACK", "evasive": "EVADE", "cautious": "CAUTIOUS"}


def card() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (W, H), BG)
    return image, ImageDraw.Draw(image)


def step(draw, box, icon, heading, lines):
    draw.rounded_rectangle(box, radius=10, fill=PANEL, outline=EDGE)
    if icon == "image":
        draw_eye(draw, box[0] + 22, box[1] + 24, AMBER)
    elif icon == "text":
        draw_hash(draw, box[0] + 22, box[1] + 23, ACCENT)
    draw.text((box[0] + (54 if icon else 22), box[1] + 20), heading, font=font(22, bold=True), fill=TEXT)
    for i, line in enumerate(lines):
        draw.text((box[0] + 22, box[1] + 64 + 26 * i), line, font=font(17), fill=MUTED)


def intro(probe: dict) -> Image.Image:
    image, draw = card()
    draw.text((80, 70), TITLE, font=font(46, bold=True), fill=TEXT)
    draw.text(
        (82, 136),
        "google/embeddinggemma-2 · frozen · no training · NVIDIA RTX 4090",
        font=font(20, mono=True),
        fill=ACCENT,
    )
    boxes = [(80, 220, 470, 520), (525, 220, 915, 520), (970, 220, 1360, 520)]
    step(
        draw,
        boxes[0],
        "image",
        "19 crops per frame",
        ["17 windows across the view", "the HEALTH box", "the AMMO box", "embedded in one batch"],
    )
    step(
        draw,
        boxes[1],
        "text",
        "cosine vs text prompts",
        [
            '"a monster" vs wall · floor · sky',
            '"HEALTH 60%"  "AMMO 14"',
            "the instruction vs 3 modes",
            "no fitted thresholds",
        ],
    )
    step(
        draw,
        boxes[2],
        None,
        "fixed rules → buttons",
        ["turn toward the best window", "fire when it is centred", "back off when hurt", "evade: its own contrasts"],
    )
    for x in (490, 935):
        draw.text((x, 352), "→", font=font(34, bold=True), fill=AMBER)
    stats = (
        f"Checked on {probe['frames']} logged frames:  ammo read {probe['ammo_exact']:.0%} exactly · "
        f"health within {probe['health_mean_abs_error']:.1f} points · "
        f"{probe['fire_precision_monster_in_aim_window']:.0%} of shots had a monster under the crosshair"
    )
    draw.text((80, 580), textwrap.fill(stats, 110), font=font(17), fill=TEXT)
    draw.text((80, 640), "Every number on screen is a live cosine similarity.", font=font(20, bold=True), fill=AMBER)
    return image


def chapter(index: int, summary: dict, behaviour: str) -> Image.Image:
    image, draw = card()
    draw.text(
        (80, 250), f"{index} · {MODE_TITLES.get(behaviour, behaviour.upper())}", font=font(30, bold=True), fill=ACCENT
    )
    draw.text((80, 310), f"“{summary['instruction']}”", font=font(40, bold=True), fill=TEXT)
    draw.text(
        (82, 390),
        f"instruction → mode by cosine: {behaviour}",
        font=font(20, mono=True),
        fill=MUTED,
    )
    result = (
        f"{summary['kills']} kills · {summary['deaths']} deaths · {summary['game_seconds']:.0f} game seconds · "
        f"{summary['mean_decision_ms']:.0f} ms per decision"
    )
    draw.text((82, 450), result, font=font(24, bold=True), fill=AMBER)
    draw.text(
        (82, 510),
        "Replay at normal game speed; the dashboard shows each decision's cosine readings.",
        font=font(17),
        fill=MUTED,
    )
    return image


def outro(runs: list[tuple[dict, str]], probe: dict) -> Image.Image:
    image, draw = card()
    draw.text((80, 90), "Results", font=font(40, bold=True), fill=TEXT)
    y = 170
    for summary, behaviour in runs:
        draw.text(
            (82, y), f"{MODE_TITLES.get(behaviour, behaviour):9s}", font=font(24, bold=True, mono=True), fill=ACCENT
        )
        draw.text(
            (260, y),
            f"{summary['kills']:>3} kills  {summary['deaths']:>2} deaths   “{summary['instruction']}”",
            font=font(22, mono=True),
            fill=TEXT,
        )
        y += 50
    lines = [
        "Zero-shot: no gradient step, no labels. Prompts + cosine similarity + fixed rules.",
        "The policy reads pixels only; game state is shown on the dashboard for checking.",
        f"Perception on {probe['frames']} logged frames: ammo {probe['ammo_exact']:.0%} exact, "
        f"health error {probe['health_mean_abs_error']:.1f}, instructions {probe['behaviour_accuracy']:.0%}.",
        "github.com/rozgo/doom-sandbox · experiments/embeddinggemma2",
    ]
    for i, line in enumerate(lines):
        draw.text((82, y + 40 + 34 * i), line, font=font(19), fill=MUTED if i < 3 else AMBER)
    return image


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *args], check=True)


def still(image: Image.Image, seconds: float, path: Path, work: Path, encoder: list[str]) -> None:
    png = work / (path.stem + ".png")
    image.save(png)
    ffmpeg("-loop", "1", "-framerate", str(FPS), "-t", str(seconds), "-i", str(png), *encoder, str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="Run directories in chapter order")
    parser.add_argument("--probe", type=Path, required=True, help="probe_zero_shot.py report")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder", default="libx264")
    args = parser.parse_args()
    probe = json.loads(args.probe.read_text())
    encoder = ["-c:v", args.encoder, "-pix_fmt", "yuv420p", "-r", str(FPS)]
    encoder += ["-crf", "18", "-preset", "medium"] if args.encoder == "libx264" else ["-cq", "19", "-preset", "p5"]

    runs = []
    for run in args.runs:
        summary = json.loads((run / "summary.json").read_text())
        first = next(
            json.loads(line)
            for line in (run / "states.jsonl").open()
            if '"policy_details"' in line and '"behaviour"' in line
        )
        runs.append((run, summary, first["policy_details"]["behaviour"]))

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        segments, chapters, clock = [], [], 0.0

        def add(path: Path, title: str, seconds: float) -> None:
            nonlocal clock
            segments.append(path)
            chapters.append((clock, clock + seconds, title))
            clock += seconds

        still(intro(probe), 7, work / "00-intro.mp4", work, encoder)
        add(work / "00-intro.mp4", "How it works", 7)
        for i, (run, summary, behaviour) in enumerate(runs, 1):
            still(chapter(i, summary, behaviour), 4, work / f"{i:02d}-card.mp4", work, encoder)
            clip = work / f"{i:02d}-play.mp4"
            ffmpeg("-i", str(run / "replay.mp4"), "-an", *encoder, str(clip))
            add(work / f"{i:02d}-card.mp4", f"{MODE_TITLES.get(behaviour, behaviour)}: {summary['instruction']}", 4)
            add(clip, f"{MODE_TITLES.get(behaviour, behaviour)} gameplay", summary["video_seconds"])
        still(outro([(s, b) for _, s, b in runs], probe), 8, work / "99-outro.mp4", work, encoder)
        add(work / "99-outro.mp4", "Results", 8)

        listing = work / "segments.txt"
        listing.write_text("".join(f"file '{p}'\n" for p in segments))
        meta = work / "chapters.txt"
        meta.write_text(
            f";FFMETADATA1\ntitle={TITLE}\n"
            + "".join(
                f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={round(a * 1000)}\nEND={round(b * 1000)}\ntitle={t}\n"
                for a, b, t in chapters
            )
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg(
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-i",
            str(meta),
            "-map_metadata",
            "1",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(args.output),
        )
    report = {
        "title": TITLE,
        "output": args.output.name,
        "chapters": [{"start_seconds": a, "end_seconds": b, "title": t} for a, b, t in chapters],
        "runs": [
            {
                "run": run.name,
                "behaviour": b,
                **{k: s[k] for k in ("instruction", "kills", "deaths", "game_seconds", "mean_decision_ms")},
            }
            for run, s, b in runs
        ],
        "probe": args.probe.name,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
