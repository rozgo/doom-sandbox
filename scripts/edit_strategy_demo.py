"""Edit the two recorded strategies into a chaptered MP4 using FFmpeg."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

from doom_bert.overlay import ACCENT, AMBER, BG, EDGE, MUTED, PANEL, TEXT, font

ROOT = Path(__file__).resolve().parents[1]
SIZE = (1440, 900)
FPS = 30


def write_text(draw, xy, text, size=22, color=TEXT, bold=False, mono=False):
    draw.text(xy, text, font=font(size, bold=bold, mono=mono), fill=color)


def canvas():
    image = Image.new("RGB", SIZE, BG)
    draw = ImageDraw.Draw(image)
    write_text(draw, (72, 38), "DOOM / ModernBERT", 26, bold=True)
    write_text(draw, (1080, 47), "TRAINED CLASSIFIER", 16, ACCENT, mono=True)
    draw.line((72, 92, 1368, 92), fill=EDGE, width=1)
    return image, draw


def assets(work: Path, attack: dict, evade: dict, contrast: dict):
    image, draw = canvas()
    write_text(
        draw, (72, 154), "INSTRUCTION-CONDITIONED GAMEPLAY", 18, ACCENT, mono=True
    )
    write_text(draw, (68, 204), "One model.", 88, bold=True)
    write_text(draw, (68, 300), "Two strategies.", 88, bold=True)
    write_text(draw, (72, 433), "Attack on sight. Then evade without firing.", 30)
    write_text(
        draw, (72, 485), "Full ModernBERT inference on every decision.", 25, MUTED
    )
    for left, value, label in (
        (72, "~43", "DECISIONS / SECOND"),
        (512, "13.9 ms", "MEAN DECISION LATENCY"),
        (952, "MPS FP16", "APPLE METAL"),
    ):
        draw.rounded_rectangle(
            (left, 591, left + 408, 740), radius=12, fill=PANEL, outline=EDGE
        )
        write_text(draw, (left + 24, 607), value, 52, ACCENT, bold=True)
        write_text(draw, (left + 24, 692), label, 16, MUTED, mono=True)
    write_text(
        draw,
        (72, 800),
        "Controlled vocabulary / full encoder + trained action head",
        21,
        MUTED,
    )
    image.save(work / "intro.png")

    image, draw = canvas()
    write_text(draw, (72, 126), "Change the instruction.", 58, bold=True)
    write_text(
        draw, (72, 207), "Same checkpoint. Identical initial game state.", 27, MUTED
    )
    for left, mode, color, lines, action in (
        (72, "aggressive", AMBER, ["Attack enemies on sight."], "Advance + fire"),
        (
            756,
            "evasive",
            ACCENT,
            ["Do not shoot.", "Evade the enemies."],
            "Back off + strafe",
        ),
    ):
        draw.rounded_rectangle(
            (left, 280, left + 612, 668), radius=12, fill=PANEL, outline=EDGE
        )
        draw.rounded_rectangle((left + 24, 306, left + 30, 346), radius=3, fill=color)
        write_text(
            draw,
            (left + 48, 310),
            "ATTACK" if mode == "aggressive" else "EVADE",
            27,
            color,
            bold=True,
        )
        for index, line in enumerate(lines):
            write_text(draw, (left + 28, 367 + index * 29), line, 24)
        probability = contrast[mode]["scores"]["ATTACK"]
        write_text(draw, (left + 28, 446), f"{probability:.3f}", 84, color, bold=True)
        write_text(draw, (left + 330, 493), "ATTACK score", 22, MUTED)
        draw.rounded_rectangle((left + 28, 550, left + 584, 560), radius=5, fill=EDGE)
        draw.rounded_rectangle(
            (left + 28, 550, left + 28 + round(556 * probability), 560),
            radius=5,
            fill=color,
        )
        write_text(draw, (left + 28, 600), action, 29, color, bold=True)
    draw.line((700, 474, 740, 474), fill=MUTED, width=3)
    draw.polygon(((730, 466), (740, 474), (730, 482)), fill=MUTED)
    write_text(draw, (72, 712), "SHARED INPUT", 16, ACCENT, mono=True)
    write_text(draw, (72, 745), contrast["state_text"], 21, mono=True)
    write_text(
        draw,
        (72, 812),
        "Next: a separate seeded run with the evade instruction. Recorded playback speed is preserved.",
        20,
        MUTED,
    )
    image.save(work / "change.png")

    image, draw = canvas()
    write_text(
        draw, (72, 147), "Different instructions. Learned actions.", 54, bold=True
    )
    write_text(
        draw,
        (72, 224),
        "Both runs use the same trained ModernBERT checkpoint.",
        27,
        MUTED,
    )
    for left, title, color, summary, headline, detail in (
        (
            72,
            "ATTACK",
            AMBER,
            attack["summary"],
            "16 kills",
            "60 game seconds / 1 death",
        ),
        (
            756,
            "EVADE",
            ACCENT,
            evade["summary"],
            "0 shots",
            "30 game seconds / 26 rounds retained",
        ),
    ):
        draw.rounded_rectangle(
            (left, 327, left + 612, 689), radius=12, fill=PANEL, outline=EDGE
        )
        write_text(draw, (left + 28, 356), title, 26, color, bold=True)
        write_text(draw, (left + 24, 410), headline, 78, color, bold=True)
        write_text(draw, (left + 28, 517), detail, 24)
        write_text(
            draw,
            (left + 28, 582),
            f"{summary['decisions_per_wall_second']:.1f} decisions/s",
            31,
            color,
            bold=True,
        )
        write_text(
            draw,
            (left + 28, 635),
            f"{summary['mean_decision_ms']:.1f} ms per decision / includes full model",
            20,
            MUTED,
        )
    write_text(
        draw,
        (72, 745),
        "Gameplay uses model scores. Rules only supplied offline training labels.",
        23,
    )
    write_text(
        draw,
        (72, 806),
        "Scope: controlled state vocabulary and known instructions.",
        21,
        MUTED,
    )
    image.save(work / "results.png")

    for name, label, detail, color in (
        ("attack-tag", "01 / ATTACK", "Aim, fire, advance", AMBER),
        ("evade-tag", "02 / EVADE", "Back off, strafe, hold fire", ACCENT),
    ):
        image = Image.new("RGBA", (332, 74), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((0, 0, 331, 73), radius=8, fill=PANEL, outline=EDGE)
        draw.rounded_rectangle((13, 13, 17, 61), radius=2, fill=color)
        write_text(draw, (30, 9), label, 23, color, bold=True)
        write_text(draw, (30, 44), detail, 17, MUTED)
        image.save(work / f"{name}.png")


def probe(path: Path) -> dict:
    return json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-show_chapters",
                "-of",
                "json",
                str(path),
            ],
            text=True,
        )
    )


def edit(work: Path, output: Path, encoder: str):
    source_names = ("trained-modernbert-demo", "trained-modernbert-evasive")
    attack, evade = [
        json.loads((ROOT / "reports" / f"{name}.json").read_text())
        for name in source_names
    ]
    contrast = json.loads(
        (ROOT / "reports/trained-instruction-contrast.json").read_text()
    )
    assets(work, attack, evade, contrast)
    encoding = [
        "-c:v",
        encoder,
        "-b:v",
        "12M",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
        "-level:v",
        "4.2",
        "-r",
        str(FPS),
        "-fps_mode",
        "cfr",
        "-video_track_timescale",
        "30000",
        "-movflags",
        "+faststart",
    ]
    if encoder == "h264_videotoolbox":
        encoding += ["-allow_sw", "0"]
    else:
        encoding += ["-preset", "fast", "-crf", "18"]
    commands, segments, timeline = [], [], []
    start = 0.0
    plans = [
        ("intro", "One model, two strategies", 2.5, None),
        ("attack", "Strategy 1: attack", None, source_names[0]),
        ("change", "Change instruction: ATTACK 0.979 to 0.026", 5.0, None),
        ("evade", "Strategy 2: evade", None, source_names[1]),
        ("results", "Results and scope", 3.5, None),
    ]
    for name, title, duration, source in plans:
        segment = work / f"{name}.mp4"
        command = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin"]
        if source:
            command += [
                "-i",
                str(ROOT / "media" / f"{source}.mp4"),
                "-loop",
                "1",
                "-framerate",
                str(FPS),
                "-i",
                str(work / f"{name}-tag.png"),
                "-filter_complex",
                "[0:v]fps=30,setpts=PTS-STARTPTS,setsar=1[v];[v][1:v]overlay=x=654:y=12:shortest=1,format=yuv420p[out]",
                "-map",
                "[out]",
            ]
        else:
            command += [
                "-loop",
                "1",
                "-framerate",
                str(FPS),
                "-i",
                str(work / f"{name}.png"),
                "-t",
                str(duration),
                "-vf",
                f"format=yuv420p,setsar=1,fade=t=in:st=0:d=0.2,fade=t=out:st={duration - 0.2}:d=0.2",
            ]
        command += ["-an", *encoding, str(segment)]
        print(f"Rendering {title}...", flush=True)
        subprocess.run(command, check=True)
        commands.append(command)
        segment_duration = float(probe(segment)["format"]["duration"])
        timeline.append(
            {
                "title": title,
                "start_seconds": start,
                "end_seconds": start + segment_duration,
                "source": f"media/{source}.mp4" if source else "editorial title card",
                "playback_speed": 1.0,
            }
        )
        start += segment_duration
        segments.append(segment)

    concat = work / "segments.txt"
    concat.write_text("".join(f"file '{segment.name}'\n" for segment in segments))
    metadata = work / "chapters.ffmetadata"
    text = ";FFMETADATA1\ntitle=ModernBERT - One model, two strategies\ncomment=Two seeded runs; complete clips at recorded speed, with editorial title cards.\n"
    for chapter in timeline:
        text += f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={round(chapter['start_seconds'] * 1000)}\nEND={round(chapter['end_seconds'] * 1000)}\ntitle={chapter['title']}\n"
    metadata.write_text(text)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-nostdin",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat),
        "-i",
        str(metadata),
        "-map",
        "0:v:0",
        "-map_metadata",
        "1",
        "-map_chapters",
        "1",
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(output),
    ]
    subprocess.run(command, check=True)
    commands.append(command)
    report = {
        "output": str(output.relative_to(ROOT))
        if output.is_relative_to(ROOT)
        else str(output),
        "encoder": encoder,
        "fps": FPS,
        "resolution": list(SIZE),
        "timeline": timeline,
        "expected_duration_seconds": start,
        "source_sha256": {
            name: hashlib.sha256(
                (ROOT / "media" / f"{name}.mp4").read_bytes()
            ).hexdigest()
            for name in source_names
        },
        "editing_notes": [
            "Both source clips are included in full, with no speed multiplier.",
            "Sources resampled to 30 fps; original timing retained to video-frame precision.",
            "Attack and evade are separate runs with identical initial structured state, not a mid-game instruction switch.",
            "Chapter badges occupy unused header space; gameplay and the live stats remain visible.",
        ],
        "commands": commands,
    }
    report_path = ROOT / "reports" / f"{output.stem}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {output} ({start:.2f} seconds)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "media/trained-strategy-comparison.mp4"
    )
    parser.add_argument("--work-dir", type=Path, default=ROOT / "runs/strategy-edit")
    parser.add_argument(
        "--encoder",
        choices=("h264_videotoolbox", "libx264"),
        default="h264_videotoolbox",
    )
    args = parser.parse_args()
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        parser.error("Install FFmpeg with ffprobe to render this edit")
    output, work = args.output.resolve(), args.work_dir.resolve()
    if output.exists() or work.exists():
        parser.error("Use a new output filename and work directory")
    work.mkdir(parents=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    edit(work, output, args.encoder)


if __name__ == "__main__":
    main()
