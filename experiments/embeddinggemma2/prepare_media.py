"""Build a small labeled multimodal test set from local macOS assets and Doom runs.

Everything is generated locally: no media is downloaded. Output goes under the
gitignored ``runs/`` directory because the stock photos and voices are Apple's.

    uv run python prepare_media.py --output ../../runs/embeddinggemma2/media
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
USER_PICTURES = Path("/Library/User Pictures")

# File stem -> caption. Captions describe the visible object, not the file name.
IMAGES = {
    "Animals/Zebra": "a zebra with black and white stripes",
    "Animals/Penguin": "a penguin",
    "Animals/Eagle": "an eagle, a bird of prey",
    "Animals/Owl": "an owl",
    "Animals/Parrot": "a colorful parrot",
    "Flowers/Sunflower": "a sunflower",
    "Flowers/Red Rose": "a red rose",
    "Flowers/Dandelion": "a dandelion seed head",
    "Flowers/Lotus": "a lotus flower",
    "Instruments/Guitar": "a guitar",
    "Instruments/Piano": "piano keys",
    "Instruments/Violin": "a violin",
    "Instruments/Drum": "a drum",
    "Instruments/Turntable": "a record turntable",
    "Sports/Basketball": "a basketball",
    "Sports/Baseball": "a baseball",
    "Sports/Soccer": "a soccer ball",
    "Sports/Golf": "a golf ball",
    "Sports/8ball": "a billiards eight ball",
    "Sports/Bowling": "a bowling ball and pins",
    "Nature/Cactus": "a cactus",
    "Nature/Earth": "planet Earth seen from space",
    "Nature/Lightning": "a lightning bolt in a stormy sky",
    "Nature/Snowflake": "a snowflake",
    "Nature/Leaf": "a green leaf",
    "Fun/Fortune Cookie": "a fortune cookie",
    "Fun/Gingerbread Man": "a gingerbread man cookie",
}

# (id, voice, spoken text, English paraphrase that shares few words with the speech)
SPEECH = [
    ("lights_en", "Samantha", "Please turn off the lights in the kitchen.", "switch off the kitchen lamps"),
    ("weather_en", "Samantha", "It is going to rain heavily tomorrow afternoon.", "storm forecast for the next day"),
    ("train_en", "Daniel", "The next train to Boston leaves from platform four.", "railway departure announcement"),
    ("doctor_en", "Daniel", "I need to schedule an appointment with my doctor.", "booking a medical visit"),
    ("pizza_en", "Samantha", "Can I order a large pepperoni pizza for delivery?", "ordering food to be delivered"),
    (
        "goal_en",
        "Daniel",
        "He scores! What an incredible goal in the final minute!",
        "sports commentator excited about a match",
    ),
    ("lights_es", "Mónica", "Por favor, apaga las luces de la cocina.", "switch off the kitchen lamps"),
    ("weather_es", "Mónica", "Mañana por la tarde va a llover mucho.", "storm forecast for the next day"),
]

# Short spoken descriptions of photos, used for audio -> image retrieval.
SPOKEN_IMAGE_QUERIES = {
    "say_zebra": ("Samantha", "A zebra.", "Animals/Zebra"),
    "say_guitar": ("Daniel", "A guitar.", "Instruments/Guitar"),
    "say_sunflower": ("Samantha", "A sunflower.", "Flowers/Sunflower"),
    "say_basketball": ("Daniel", "A basketball.", "Sports/Basketball"),
    "say_lightning": ("Samantha", "Lightning in a storm.", "Nature/Lightning"),
    "say_penguin": ("Daniel", "A penguin.", "Animals/Penguin"),
}

SR = 16_000


def run(*cmd: str) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


def write_wav(path: Path, samples: np.ndarray) -> None:
    import wave

    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SR)
        wav.writeframes(pcm.tobytes())


def tts(path: Path, voice: str, text: str) -> None:
    aiff = path.with_suffix(".aiff")
    run("say", "-v", voice, "-o", str(aiff), text)
    run("afconvert", "-f", "WAVE", "-d", f"LEI16@{SR}", "-c", "1", str(aiff), str(path))
    aiff.unlink()


def build_images(out: Path) -> dict:
    images = {}
    (out / "images").mkdir(parents=True, exist_ok=True)
    for stem, caption in IMAGES.items():
        dst = out / "images" / (stem.split("/")[1].replace(" ", "_").lower() + ".png")
        run("sips", "-s", "format", "png", "-Z", "512", str(USER_PICTURES / f"{stem}.heic"), "--out", str(dst))
        images[stem] = {"path": str(dst), "caption": caption}
    return images


def build_audio(out: Path) -> dict:
    (out / "audio").mkdir(parents=True, exist_ok=True)
    speech = []
    for sid, voice, text, paraphrase in SPEECH:
        dst = out / "audio" / f"{sid}.wav"
        tts(dst, voice, text)
        speech.append({"id": sid, "path": str(dst), "voice": voice, "transcript": text, "paraphrase": paraphrase})
    spoken_queries = []
    for sid, (voice, text, image) in SPOKEN_IMAGE_QUERIES.items():
        dst = out / "audio" / f"{sid}.wav"
        tts(dst, voice, text)
        spoken_queries.append({"id": sid, "path": str(dst), "transcript": text, "image": image})

    rng = np.random.default_rng(0)
    t = np.arange(3 * SR) / SR
    events = {
        "beep": (0.4 * np.sin(2 * math.pi * 1000 * t), "a steady high-pitched electronic beep tone"),
        "noise": (0.3 * rng.standard_normal(t.size), "loud white noise static hiss"),
        "silence": (np.zeros_like(t), "complete silence"),
        "speech": (None, "a person talking"),
    }
    sound_events = []
    for name, (samples, label) in events.items():
        if samples is None:
            path = out / "audio" / "train_en.wav"
        else:
            path = out / "audio" / f"event_{name}.wav"
            write_wav(path, samples)
        sound_events.append({"id": name, "path": str(path), "label": label})
    return {"speech": speech, "spoken_image_queries": spoken_queries, "sound_events": sound_events}


def frames_to_video(frames: list[Path], dst: Path, fps: int = 1) -> None:
    listing = dst.with_suffix(".txt")
    listing.write_text("".join(f"file '{f}'\nduration {1 / fps}\n" for f in frames) + f"file '{frames[-1]}'\n")
    run(
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(listing),
        "-vf",
        f"fps={fps},format=yuv420p",
        "-c:v",
        "libx264",
        str(dst),
    )
    listing.unlink()


def moving_ball(dst: Path, left_to_right: bool, seconds: int = 6, fps: int = 4) -> None:
    frames_dir = dst.parent / dst.stem
    frames_dir.mkdir(exist_ok=True)
    frames = []
    n = seconds * fps
    for i in range(n):
        x = 60 + (520 * i / (n - 1))
        if not left_to_right:
            x = 640 - x
        im = Image.new("RGB", (640, 360), (235, 235, 235))
        ImageDraw.Draw(im).ellipse([x - 40, 140, x + 40, 220], fill=(220, 30, 30))
        frames.append(frames_dir / f"{i:03d}.png")
        im.save(frames[-1])
    frames_to_video(frames, dst, fps=fps)


def build_video(out: Path, images: dict) -> list[dict]:
    vdir = out / "video"
    vdir.mkdir(parents=True, exist_ok=True)
    videos = []

    def add(vid: str, path: Path, label: str, group: str) -> None:
        videos.append({"id": vid, "path": str(path), "label": label, "group": group})

    doom = sorted((REPO / "runs/trained-modernbert-demo/frames").glob("*.png"))[:16]
    frames_to_video(doom, vdir / "doom.mp4")
    add("doom", vdir / "doom.mp4", "first-person shooter video game with monsters and a gun", "content")

    lavfi = {
        "mandelbrot": ("mandelbrot=size=640x360:rate=4", "zooming into a colorful fractal"),
        "life": (
            "life=size=640x360:rate=4:mold=10:ratio=0.3:life_color=white:death_color=black",
            "Conway's game of life cellular automaton pixels",
        ),
        "colorbars": ("smptebars=size=640x360:rate=4", "a television color bars test pattern"),
    }
    for vid, (src, label) in lavfi.items():
        dst = vdir / f"{vid}.mp4"
        run("ffmpeg", "-y", "-f", "lavfi", "-i", src, "-t", "8", "-pix_fmt", "yuv420p", "-c:v", "libx264", str(dst))
        add(vid, dst, label, "content")

    zebra = images["Animals/Zebra"]["path"]
    dst = vdir / "zebra_zoom.mp4"
    run(
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-i",
        zebra,
        "-vf",
        "scale=1024:1024,zoompan=z='1+0.02*on':d=32:s=512x512:fps=4,format=yuv420p",
        "-t",
        "8",
        "-c:v",
        "libx264",
        str(dst),
    )
    add("zebra_zoom", dst, "a slow zoom on a zebra", "content")

    moving_ball(vdir / "ball_ltr.mp4", left_to_right=True)
    add("ball_ltr", vdir / "ball_ltr.mp4", "a red ball moving from left to right", "direction")
    moving_ball(vdir / "ball_rtl.mp4", left_to_right=False)
    add("ball_rtl", vdir / "ball_rtl.mp4", "a red ball moving from right to left", "direction")
    return videos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "runs/embeddinggemma2/media")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    images = build_images(out)
    manifest = {"images": images, "audio": build_audio(out), "video": build_video(out, images)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"{len(images)} images, {len(manifest['audio']['speech'])} speech clips, "
        f"{len(manifest['video'])} videos -> {out}"
    )


if __name__ == "__main__":
    main()
