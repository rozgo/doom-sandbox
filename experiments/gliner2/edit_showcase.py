"""Cut the GLiNER2.5 showcase: intro, one chapter per instruction, results.

Each chapter is a complete recorded run (dashboard replay at game speed) behind
a title card. Numbers on the cards come from the runs' summary.json, the
evaluation report and the conditional probe report, never typed in by hand.

    uv run python edit_showcase.py --runs ../../runs/gliner2-show-attack ../../runs/gliner2-show-evade \\
        ../../runs/gliner2-show-cautious --report ../../runs/gliner2-v2/eval-base-full/report.json \\
        --conditionals ../../runs/gliner2-v2/conditionals-base-full.json --output ../../runs/gliner2-showcase/out.mp4
"""

import argparse
import json
import subprocess
import tempfile
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw

from doom_bert.overlay import ACCENT, AMBER, BG, EDGE, MUTED, PANEL, TEXT, font

W, H, FPS = 1440, 900, 35
TITLE = "GLiNER2.5 plays Doom"
TITLES = {
    "Attack enemies on sight.": "ATTACK",
    "Do not shoot. Evade the enemies.": "EVADE",
    "Shoot at enemies, but retreat when health falls below 40.": "CAUTIOUS",
}


def count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def scope(report: dict) -> str:
    n = report["doom_training_examples"]
    if not n:
        return "zero-shot, no Doom training"
    how = "fully fine-tuned" if "fully" in report["training_scope"] else "head adapted, encoder frozen"
    return f"{how} on {n:,} teacher-labelled Doom examples"


def card() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (W, H), BG)
    return image, ImageDraw.Draw(image)


def step(draw, box, heading, lines):
    draw.rounded_rectangle(box, radius=10, fill=PANEL, outline=EDGE)
    draw.text((box[0] + 22, box[1] + 20), heading, font=font(22, bold=True), fill=TEXT)
    for i, line in enumerate(lines):
        draw.text((box[0] + 22, box[1] + 64 + 26 * i), line, font=font(17), fill=MUTED)


def intro(report: dict, conditionals: dict) -> Image.Image:
    image, draw = card()
    draw.text((80, 70), TITLE, font=font(46, bold=True), fill=TEXT)
    draw.text(
        (82, 136),
        f"{report['checkpoint']} · {scope(report)}",
        font=font(18, mono=True),
        fill=ACCENT,
    )
    boxes = [(80, 220, 470, 520), (525, 220, 915, 520), (970, 220, 1360, 520)]
    step(
        draw,
        boxes[0],
        "instruction + game state",
        [
            "the player's instruction",
            "health and ammo as numbers",
            "nearest enemy: side and range",
            "read from the engine, as text",
        ],
    )
    step(
        draw,
        boxes[1],
        "7 label scores",
        [
            "one score per button description",
            '"Fire the equipped weapon…"',
            '"Walk backward, away…"',
            "all from one encoder pass",
        ],
    )
    step(
        draw,
        boxes[2],
        "valid button vector",
        ["best legal combination", "no opposite directions", "no firing on empty", "no gameplay rules added"],
    )
    for x in (490, 935):
        draw.text((x, 352), "→", font=font(34, bold=True), fill=AMBER)
    test = report["agreement"]["test"]["action"]["exact_teacher_agreement"]
    novel = report["agreement"]["novel_instruction_test"]["action"]["exact_teacher_agreement"]
    stats = (
        f"Held-out game states: {test:.0%} exact agreement with the teacher · new instruction phrasings: {novel:.0%} · "
        f"explicit conditional probes: {conditionals['attack_checks_passed']}/{conditionals['attack_checks_total']}"
    )
    draw.text((80, 580), textwrap.fill(stats, 110), font=font(17), fill=TEXT)
    draw.text(
        (80, 640),
        "The scores on the dashboard are the model's own sigmoid outputs.",
        font=font(20, bold=True),
        fill=AMBER,
    )
    return image


def chapter(index: int, summary: dict) -> Image.Image:
    image, draw = card()
    title = TITLES.get(summary["instruction"], "RUN")
    draw.text((80, 250), f"{index} · {title}", font=font(30, bold=True), fill=ACCENT)
    draw.text((80, 310), f"“{summary['instruction']}”", font=font(38, bold=True), fill=TEXT)
    result = (
        f"{count(summary['kills'], 'kill')} · {count(summary['deaths'], 'death')} · {summary['game_seconds']:.0f} game seconds · "
        f"{summary['mean_decision_ms']:.0f} ms per decision"
    )
    draw.text((82, 410), result, font=font(24, bold=True), fill=AMBER)
    draw.text(
        (82, 470),
        "Replay at normal game speed; the dashboard shows each decision's seven scores.",
        font=font(17),
        fill=MUTED,
    )
    return image


def outro(runs: list[dict], report: dict, conditionals: dict) -> Image.Image:
    image, draw = card()
    draw.text((80, 90), "Results", font=font(40, bold=True), fill=TEXT)
    y = 170
    for summary in runs:
        draw.text(
            (82, y), f"{TITLES.get(summary['instruction'], 'RUN'):9s}", font=font(24, bold=True, mono=True), fill=ACCENT
        )
        draw.text(
            (260, y),
            f"{count(summary['kills'], 'kill'):>8}  {count(summary['deaths'], 'death'):>8}   “{summary['instruction']}”",
            font=font(21, mono=True),
            fill=TEXT,
        )
        y += 50
    test = report["agreement"]["test"]["action"]["exact_teacher_agreement"]
    novel = report["agreement"]["novel_instruction_test"]["action"]["exact_teacher_agreement"]
    lines = [
        f"Held-out states {test:.0%}, new phrasings {novel:.0%}, conditional probes "
        f"{conditionals['attack_checks_passed']}/{conditionals['attack_checks_total']}, behaviour checks "
        f"{report['behavior']['passed']}/{report['behavior']['total']}.",
        f"GLiNER2.5 Base, {scope(report)}.",
        "The policy reads the game state as text; no gameplay rules are added after the model.",
        "github.com/rozgo/doom-sandbox · experiments/gliner2",
    ]
    for i, line in enumerate(lines):
        draw.text((82, y + 40 + 34 * i), line, font=font(19), fill=MUTED if i < 3 else AMBER)
    return image


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *args], check=True)


def still(image: Image.Image, seconds: float, path: Path, encoder: list[str]) -> None:
    png = path.with_suffix(".png")
    image.save(png)
    ffmpeg("-loop", "1", "-framerate", str(FPS), "-t", str(seconds), "-i", str(png), *encoder, str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--report", type=Path, required=True, help="compare.py report for the shown model")
    parser.add_argument("--conditionals", type=Path, required=True, help="check_conditionals.py report")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    conditionals = json.loads(args.conditionals.read_text())
    encoder = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS), "-crf", "18", "-preset", "medium"]
    runs = [(run, json.loads((run / "summary.json").read_text())) for run in args.runs]

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        segments, chapters, clock = [], [], 0.0

        def add(path: Path, title: str, seconds: float) -> None:
            nonlocal clock
            segments.append(path)
            chapters.append((clock, clock + seconds, title))
            clock += seconds

        still(intro(report, conditionals), 7, work / "00-intro.mp4", encoder)
        add(work / "00-intro.mp4", "How it works", 7)
        for i, (run, summary) in enumerate(runs, 1):
            title = TITLES.get(summary["instruction"], "RUN")
            still(chapter(i, summary), 4, work / f"{i:02d}-card.mp4", encoder)
            add(work / f"{i:02d}-card.mp4", f"{title}: {summary['instruction']}", 4)
            clip = work / f"{i:02d}-play.mp4"
            ffmpeg("-i", str(run / "replay.mp4"), "-an", *encoder, str(clip))
            add(clip, f"{title} gameplay", summary["video_seconds"])
        still(outro([s for _, s in runs], report, conditionals), 8, work / "99-outro.mp4", encoder)
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
    summary = {
        "title": TITLE,
        "output": args.output.name,
        "chapters": [{"start_seconds": a, "end_seconds": b, "title": t} for a, b, t in chapters],
        "runs": [
            {
                "run": run.name,
                **{k: s[k] for k in ("instruction", "kills", "deaths", "game_seconds", "mean_decision_ms")},
            }
            for run, s in runs
        ],
        "report": args.report.name,
        "conditionals": args.conditionals.name,
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
