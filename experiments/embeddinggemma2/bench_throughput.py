"""Measure EmbeddingGemma 2 throughput: texts, images, and images with text per second.

Rates are end to end through sentence-transformers ``encode`` (CPU preprocessing
included), repeated for about four seconds per cell after one warm-up call.
Images are 640x480 Doom frames; the vision token budget sets their cost.

    uv run python bench_throughput.py --output ../../reports/embeddinggemma2/throughput.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch
from gemma_doom import default_dtype, load_encoder, select_device
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
SHORT = "a monster in a dark corridor"
PASSAGE = " ".join(
    ["The northern lights are caused by charged particles from the sun colliding with the atmosphere."] * 8
)


def rate(model, inputs: list, batch: int, seconds: float) -> float | None:
    try:
        model.encode(inputs[:batch], batch_size=batch)
        sync()
        count, start = 0, time.perf_counter()
        while time.perf_counter() - start < seconds:
            model.encode(inputs, batch_size=batch)
            count += len(inputs)
        sync()
        return round(count / (time.perf_counter() - start), 1)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--output", type=Path, default=REPO / "reports/embeddinggemma2/throughput.json")
    args = parser.parse_args()
    device = select_device(args.device)
    dtype = default_dtype(device)
    frame = Image.open(Path(__file__).with_name("sample_frame.png")).convert("RGB")
    rows = []

    model = load_encoder(device, dtype, 140)
    for name, text in (("short text", SHORT), ("passage", PASSAGE)):
        tokens = len(model.preprocess([text])["input_ids"][0])
        for batch in (1, 64, 256, 1024):
            rows.append(
                {
                    "input": name,
                    "text_tokens": tokens,
                    "batch": batch,
                    "per_second": rate(model, [text] * batch, batch, args.seconds),
                }
            )
            print(rows[-1], flush=True)
    del model

    for vision_tokens in (70, 140, 280):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model = load_encoder(device, dtype, vision_tokens)
        inputs = {
            "image": frame,
            "image + short text": {"text": f"A Doom screenshot: {SHORT}. <|image|>", "image": [frame]},
            "image + passage": {"text": f"{PASSAGE} <|image|>", "image": [frame]},
        }
        for name, item in inputs.items():
            for batch in (1, 8, 19, 64):
                rows.append(
                    {
                        "input": name,
                        "vision_tokens": vision_tokens,
                        "batch": batch,
                        "per_second": rate(model, [item] * batch, batch, args.seconds),
                    }
                )
                print(rows[-1], flush=True)
        del model

    report = {
        "model": "google/embeddinggemma-2",
        "device": torch.cuda.get_device_name(0) if device == "cuda" else device,
        "dtype": dtype,
        "torch": torch.__version__,
        "machine": platform.machine(),
        "seconds_per_cell": args.seconds,
        "image": "640x480 Doom frame",
        "method": "sentence-transformers encode, CPU preprocessing included; null = out of memory",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
