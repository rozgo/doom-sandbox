"""Check CPU float32 speed and score agreement with recorded MPS float16 probes."""

import argparse
import gc
import json
import random
from pathlib import Path

import torch
from compare import behavior_checks, evaluate_one, latency, probes
from gliner_policy import GLiNERPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    torch.set_num_threads(4)
    results = []
    for model in ("small", "base"):
        policy = GLiNERPolicy(
            f"fastino/gliner2.5-{model}-v1", device="cpu", dtype="float32"
        )
        cases = list(probes())
        for _, instruction, obs in cases:
            evaluate_one(policy, instruction, obs)
        measured = []
        order = cases * 5
        random.Random(7).shuffle(order)
        for name, instruction, obs in order:
            measured.append({"id": name, **evaluate_one(policy, instruction, obs)})
        reference = json.loads(
            (args.reference_root / f"gliner2-{model}-mps-fp16/report.json").read_text()
        )
        old = {p["id"]: p["scores"] for p in reference["probes"]}
        result = {
            "model": model,
            "device": "cpu",
            "dtype": "float32",
            "cpu_threads": 4,
            "latency": latency([r["classification_ms"] for r in measured]),
            "behavior": behavior_checks(measured),
            "max_absolute_score_difference_vs_mps_fp16": max(
                abs(value - old[r["id"]][button])
                for r in measured
                for button, value in r["scores"].items()
            ),
            "probes": measured[:],
        }
        results.append(result)
        print(
            json.dumps({k: v for k, v in result.items() if k != "probes"}, indent=2),
            flush=True,
        )
        del policy
        gc.collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "results": results,
                "note": "55 batch-one classifications per model after all 11 probe shapes warmed; CPU timings include the same preprocessing and action decoding as MPS.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
