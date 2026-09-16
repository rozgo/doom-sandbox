"""Measure batch-one ModernBERT decisions, including tokenization and GPU transfer."""

import argparse
import gc
import json
import math
import platform
import statistics
import time
from pathlib import Path

import torch
import transformers

from doom_bert.model import BASE_MODEL, ModernBertPolicy


def sample_observation(index: int) -> dict:
    return {
        "variables": {
            "HEALTH": 100 - index % 80,
            "SELECTED_WEAPON_AMMO": 26,
            "POSITION_X": 0,
            "POSITION_Y": 0,
        },
        "screen_width": 640,
        "damage_since_observation": index % 2,
        "visible_objects": [
            {
                "id": enemy,
                "name": "MarineChainsawVzd",
                "category": "Monster",
                "box": [100 + enemy * 140 + index % 10, 150, 40, 90],
                "position": [120 + enemy * 80, 50, 0],
            }
            for enemy in range(3)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=BASE_MODEL)
    parser.add_argument(
        "--devices",
        nargs="+",
        choices=("auto", "mps", "cpu", "cuda"),
        default=["cpu", "auto"],
    )
    parser.add_argument(
        "--dtype", choices=("auto", "float16", "float32", "bfloat16"), default="auto"
    )
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--output", type=Path, default=Path("runs/modernbert-benchmark.json")
    )
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 1:
        parser.error("--iterations and --warmup must be positive")
    report = {
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "checkpoint": args.checkpoint,
        "batch_size": 1,
        "max_tokens": 256,
        "attention": "sdpa",
        "untrained_action_head": args.checkpoint == BASE_MODEL,
        "timing_scope": "serialization + tokenization + transfer + inference + CPU action scores",
        "results": [],
    }
    instruction = "Shoot at enemies, but retreat when health falls below 40."
    for device in args.devices:
        print(f"Loading {args.checkpoint} for {device}...", flush=True)
        torch.manual_seed(7)
        policy = ModernBertPolicy(
            args.checkpoint,
            device=device,
            dtype=args.dtype,
            allow_untrained_head=report["untrained_action_head"],
        )
        for index in range(args.warmup):
            policy(instruction, sample_observation(index))
        policy.synchronize()
        timings = []
        for index in range(args.iterations):
            started = time.perf_counter()
            scores = policy(instruction, sample_observation(index))
            policy.synchronize()
            timings.append((time.perf_counter() - started) * 1000)
            if not all(
                math.isfinite(value) and 0 <= value <= 1 for value in scores.values()
            ):
                raise RuntimeError("Model produced invalid scores")
        result = {
            "device": policy.device,
            "dtype": policy.dtype,
            "iterations": args.iterations,
            "mean_ms": statistics.mean(timings),
            "median_ms": statistics.median(timings),
            "p95_ms": sorted(timings)[math.ceil(0.95 * len(timings)) - 1],
            "decisions_per_second": 1000 / statistics.mean(timings),
        }
        report["results"].append(result)
        print(json.dumps(result, indent=2), flush=True)
        del policy
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
