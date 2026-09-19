"""Small imitation experiment: freeze GLiNER's encoder and adapt its shared head.

Features are cached for offline training only. Head checkpoints are selected on
validation states; test states and novel phrasings never select weights.
"""

import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from compare import ROOT, observation
from gliner_policy import GLiNERPolicy
from safetensors.torch import save_file

from doom_bert.policy import BUTTONS


def metrics(logits, targets):
    predicted = logits.sigmoid() > 0.5
    return {
        "exact_teacher_agreement": (predicted == targets.bool())
        .all(1)
        .float()
        .mean()
        .item(),
        "button_teacher_agreement": (predicted == targets.bool()).float().mean().item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("small", "base"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[32, 128, 372])
    parser.add_argument("--steps", type=int, default=600)
    args = parser.parse_args()
    if args.steps <= 0 or any(n < 1 or n > 372 for n in args.budgets):
        parser.error("Require positive steps and budgets between 1 and 372")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(7)
    torch.set_num_threads(4)
    started = time.perf_counter()
    checkpoint = f"fastino/gliner2.5-{args.model}-v1"
    policy = GLiNERPolicy(checkpoint)
    policy.classifier.model.requires_grad_(False)
    rows = [
        json.loads(line)
        for line in (ROOT / "data/controlled-demo.jsonl").read_text().splitlines()
    ]
    features = []
    original_scores = []

    def capture(module, inputs):
        features.append(inputs[0].detach().float().cpu())

    hook = policy.classifier.model.classifier.register_forward_pre_hook(capture)
    cache_started = time.perf_counter()
    for index, row in enumerate(rows):
        scores = policy(row["instruction"], observation(row["state"]))
        original_scores.append([scores[b] for b in BUTTONS])
        if (index + 1) % 100 == 0:
            print(
                f"Cached training/evaluation features {index + 1}/{len(rows)}",
                flush=True,
            )
    policy.synchronize()
    cache_seconds = time.perf_counter() - cache_started
    hook.remove()
    if len(features) != len(rows) or policy.encoder_forward_calls != len(rows):
        raise RuntimeError(
            "Feature extraction must execute the complete encoder per row"
        )
    features = torch.stack(features).clone()
    labels = torch.tensor([r["labels"] for r in rows], dtype=torch.float32)
    indices = {
        split: [i for i, r in enumerate(rows) if r["split"] == split]
        for split in ("train", "validation", "test")
    }
    rng = random.Random(7)
    groups = [
        [i for i in indices["train"] if rows[i]["mode"] == mode]
        for mode in ("aggressive", "cautious", "evasive")
    ]
    for group in groups:
        rng.shuffle(group)
    ordered_train = [i for triplet in zip(*groups, strict=True) for i in triplet]
    original = copy.deepcopy(policy.classifier.model.classifier).cpu().float()
    with torch.no_grad():
        reconstruction_error = (
            (original(features).squeeze(-1).sigmoid() - torch.tensor(original_scores))
            .abs()
            .max()
            .item()
        )
    if reconstruction_error > 0.02:
        raise RuntimeError(
            "Cached feature/head scores do not reproduce label-aligned inference"
        )
    results = []
    for budget in args.budgets:
        torch.manual_seed(7)
        head = copy.deepcopy(original).requires_grad_(True).train()
        chosen = torch.tensor(ordered_train[:budget])
        positives = labels[chosen].sum(0)
        loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=((budget - positives) / positives.clamp(min=1)).clamp(
                min=1, max=20
            )
        )
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.0005, weight_decay=0.01)
        best_key, best_weights, best_step = (-1, float("-inf")), None, 0
        history = []
        optimize_started = time.perf_counter()
        for step in range(1, args.steps + 1):
            head.train()
            batch = chosen[torch.randint(budget, (min(64, budget),))]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(head(features[batch]).squeeze(-1), labels[batch])
            loss.backward()
            optimizer.step()
            if step % 25 == 0 or step == args.steps:
                head.eval()
                with torch.no_grad():
                    logits = head(features[indices["validation"]]).squeeze(-1)
                    validation = metrics(logits, labels[indices["validation"]])
                    validation_loss = loss_fn(
                        logits, labels[indices["validation"]]
                    ).item()
                key = validation["exact_teacher_agreement"], -validation_loss
                history.append(
                    {"step": step, **validation, "validation_loss": validation_loss}
                )
                if key > best_key:
                    best_key, best_weights, best_step = (
                        key,
                        copy.deepcopy(head.state_dict()),
                        step,
                    )
        optimize_seconds = time.perf_counter() - optimize_started
        head.load_state_dict(best_weights)
        head.eval()
        with torch.no_grad():
            evaluation = {
                name: metrics(head(features[selection]).squeeze(-1), labels[selection])
                for name, selection in {
                    "selected_train": chosen,
                    "validation": indices["validation"],
                    "test": indices["test"],
                }.items()
            }
        path = args.output / f"head-{budget}"
        path.mkdir()
        save_file(
            {k: v.contiguous() for k, v in head.state_dict().items()},
            path / "head.safetensors",
        )
        metadata = {
            "base_model": checkpoint,
            "revision": policy.revision,
            "schema": policy.schema.build(),
            "training_examples": budget,
            "dataset_sha256": hashlib.sha256(
                (ROOT / "data/controlled-demo.jsonl").read_bytes()
            ).hexdigest(),
            "optimizer": "AdamW",
            "learning_rate": 0.0005,
            "weight_decay": 0.01,
            "batch_size": min(64, budget),
            "loss": "BCEWithLogitsLoss; per-button positive weight clipped to [1,20]",
            "state_format": policy.state_format,
            "trainable_parameters": sum(p.numel() for p in head.parameters()),
            "selected_dataset_rows": chosen.tolist(),
            "positive_training_examples": dict(
                zip(BUTTONS, positives.tolist(), strict=True)
            ),
            "seed": 7,
            "selected_step": best_step,
            "completed_steps": args.steps,
            "encoder_frozen": True,
            "runtime_feature_cache": False,
            "head_training_device": "cpu",
            "head_training_dtype": "float32",
            "encoder_feature_device": "mps",
            "encoder_feature_dtype": "float16",
            "optimization_seconds": optimize_seconds,
            "selection": "Best validation exact vector agreement, then validation BCE; test unused for selection",
            "metrics_cached_features": evaluation,
            "history": history,
        }
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        results.append(
            {
                "budget": budget,
                "head": str(path),
                "selected_step": best_step,
                "optimization_seconds": optimize_seconds,
                "metrics_cached_features": evaluation,
            }
        )
        print(json.dumps(results[-1], indent=2), flush=True)
    report = {
        "model": checkpoint,
        "revision": policy.revision,
        "feature_extraction_examples": len(rows),
        "feature_extraction_seconds": cache_seconds,
        "max_original_head_reconstruction_error": reconstruction_error,
        "wall_seconds_including_load_features_and_all_budgets": time.perf_counter()
        - started,
        "timing_note": "Optimization times exclude shared model loading and feature extraction; not directly comparable with the earlier 3.6-second ModernBERT training timer.",
        "budgets": results,
        "evaluation_note": "Cached-feature metrics only. Reload adapted heads and run compare.py for full-forward inference and live gameplay verification.",
    }
    (args.output / "training-report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
