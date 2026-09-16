"""Train a small ModernBERT action head on an explicitly controlled vocabulary."""

import argparse
import copy
import json
import random
import time
from itertools import product
from pathlib import Path

import torch
import transformers

from doom_bert.model import BASE_MODEL, ModernBertPolicy
from doom_bert.policy import BUTTONS, serialize_categorical_observation
from doom_bert.teacher import TEACHERS, teacher_decision

INSTRUCTIONS = {
    "aggressive": (
        TEACHERS["aggressive"].instruction,
        "Shoot enemies whenever you can.",
    ),
    "cautious": (
        TEACHERS["cautious"].instruction,
        "Fight, but stop shooting and retreat below 40 health.",
    ),
    "evasive": (TEACHERS["evasive"].instruction, "Never fire. Keep away from enemies."),
}


def make_dataset(seed: int = 7) -> list[dict]:
    states = []
    targets = [(None, None), *product((-0.5, 0, 0.5), (100, 200, 350, 700))]
    for health, ammo, (bearing, distance) in product((10, 30, 100), (0, 10), targets):
        states.append(
            {
                "variables": {
                    "HEALTH": health,
                    "SELECTED_WEAPON_AMMO": ammo,
                    "POSITION_X": 0,
                    "POSITION_Y": 0,
                },
                "screen_width": 640,
                "visible_objects": []
                if bearing is None
                else [
                    {
                        "id": 1,
                        "name": "Enemy",
                        "category": "Monster",
                        "box": [300 + bearing * 320, 100, 40, 80],
                        "position": [distance, 0, 0],
                    }
                ],
            }
        )
    order = list(range(len(states)))
    random.Random(seed).shuffle(order)
    train_ids, validation_ids = set(order[:62]), set(order[62:70])
    rows = []
    for state_id, state in enumerate(states):
        split = (
            "train"
            if state_id in train_ids
            else ("validation" if state_id in validation_ids else "test")
        )
        for mode, instructions in INSTRUCTIONS.items():
            scores, _ = teacher_decision(state, TEACHERS[mode])
            labels = [float(scores[button] > 0.5) for button in BUTTONS]
            for instruction in instructions:
                rows.append(
                    {
                        "state_id": state_id,
                        "split": split,
                        "mode": mode,
                        "instruction": instruction,
                        "state": serialize_categorical_observation(state),
                        "labels": labels,
                    }
                )
    return rows


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    predicted = logits.sigmoid() > 0.5
    target = labels.bool()
    per_button = {}
    for index, name in enumerate(BUTTONS):
        p, t = predicted[:, index], target[:, index]
        tp, fp, fn = (p & t).sum().item(), (p & ~t).sum().item(), (~p & t).sum().item()
        per_button[name] = {
            "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "positive_examples": t.sum().item(),
        }
    return {
        "examples": len(labels),
        "exact_vector_accuracy": (predicted == target).all(dim=1).float().mean().item(),
        "button_accuracy": (predicted == target).float().mean().item(),
        "per_button": per_button,
    }


def cache_features(
    policy: ModernBertPolicy, rows: list[dict], batch_size: int
) -> torch.Tensor:
    features = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            encoded = policy.tokenizer(
                [r["instruction"] for r in batch],
                [r["state"] for r in batch],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
                return_token_type_ids=False,
            ).to(policy.device)
            hidden = policy.model.model(**encoded).last_hidden_state
            if policy.model.config.classifier_pooling == "cls":
                pooled = hidden[:, 0]
            else:
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            features.append(pooled.float().cpu())
            if start % (batch_size * 8) == 0:
                print(
                    f"Cached frozen encoder features: {min(start + batch_size, len(rows))}/{len(rows)}",
                    flush=True,
                )
    return torch.cat(features)


def train(
    *,
    output: Path,
    dataset_path: Path,
    seed: int = 7,
    steps: int = 1200,
    batch_size: int = 16,
    device: str = "auto",
) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    rows = make_dataset(seed)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    policy = ModernBertPolicy(BASE_MODEL, device=device, allow_untrained_head=True)
    policy.model.config.doom_bert_state_format = "categorical-v1"
    policy.model.config.doom_bert_training_scope = (
        "frozen encoder; trained prediction head and seven-button classifier"
    )
    policy.model.model.requires_grad_(False)
    features = cache_features(policy, rows, batch_size)
    # Caching is only a training optimization. Gameplay always runs the encoder.
    head = torch.nn.Sequential(
        policy.model.head, policy.model.drop, policy.model.classifier
    ).to(device="cpu", dtype=torch.float32)
    labels = torch.tensor([r["labels"] for r in rows], dtype=torch.float32)
    indices = {
        split: torch.tensor([i for i, r in enumerate(rows) if r["split"] == split])
        for split in ("train", "validation", "test")
    }
    train_indices = indices["train"]
    positives = labels[train_indices].sum(dim=0)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=((len(train_indices) - positives) / positives.clamp(min=1)).clamp(
            max=20
        )
    )
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.002, weight_decay=0.01)

    def evaluate(split):
        head.eval()
        with torch.no_grad():
            selected = indices[split]
            return classification_metrics(head(features[selected]), labels[selected])

    initial_validation = evaluate("validation")
    best_accuracy, best_loss, best_step, best = -1, float("inf"), 0, None
    history = []
    for step in range(1, steps + 1):
        head.train()
        batch = train_indices[torch.randint(len(train_indices), (64,))]
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(head(features[batch]), labels[batch])
        loss.backward()
        optimizer.step()
        if step % 50 == 0 or step == steps:
            train_metrics, validation_metrics = (
                evaluate("train"),
                evaluate("validation"),
            )
            with torch.no_grad():
                validation_loss = loss_fn(
                    head(features[indices["validation"]]), labels[indices["validation"]]
                ).item()
            accuracy = validation_metrics["exact_vector_accuracy"]
            history.append(
                {
                    "step": step,
                    "train_exact": train_metrics["exact_vector_accuracy"],
                    "validation_exact": accuracy,
                    "validation_loss": validation_loss,
                }
            )
            print(json.dumps(history[-1]), flush=True)
            if (accuracy, -validation_loss) > (best_accuracy, -best_loss):
                best_accuracy, best_loss, best_step = accuracy, validation_loss, step
                best = copy.deepcopy(head.state_dict())
            if (
                step >= 200
                and accuracy == 1
                and train_metrics["exact_vector_accuracy"] == 1
            ):
                break
    head.load_state_dict(best)
    metrics = {split: evaluate(split) for split in indices}
    head.eval()
    with torch.no_grad():
        scores = head(features).sigmoid()
    flip_results = []
    # Same observations, opposite known instructions; these are test-set states.
    for state_id in sorted({r["state_id"] for r in rows if r["split"] == "test"}):
        pair = {
            mode: next(
                i
                for i, r in enumerate(rows)
                if r["state_id"] == state_id and r["mode"] == mode
            )
            for mode in ("aggressive", "evasive")
        }
        a, e = pair["aggressive"], pair["evasive"]
        if labels[a, 0] == 1:
            flip_results.append(
                {
                    "state": rows[a]["state"],
                    "aggressive_attack": scores[a, 0].item(),
                    "evasive_attack": scores[e, 0].item(),
                }
            )
    report = {
        "base_model": BASE_MODEL,
        "seed": seed,
        "state_format": "categorical-v1",
        "training_scope": policy.model.config.doom_bert_training_scope,
        "encoder_device": policy.device,
        "encoder_dtype": policy.dtype,
        "head_training_device": "cpu",
        "head_training_dtype": "float32",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "trainable_parameters": sum(p.numel() for p in head.parameters()),
        "total_parameters": sum(p.numel() for p in policy.model.parameters()),
        "selected_step": best_step,
        "completed_steps": step,
        "initial_validation": initial_validation,
        "metrics": metrics,
        "test_instruction_flips": flip_results,
        "history": history,
        "split_unit": "whole categorical state, across every instruction",
        "dataset": str(dataset_path),
        "instructions": INSTRUCTIONS,
        "runtime_encoder_cache": False,
        "runtime_teacher": False,
        "training_wall_seconds": time.monotonic() - started,
    }
    policy.model.to(device="cpu", dtype=torch.float16).eval()
    policy.model.save_pretrained(output)
    policy.tokenizer.save_pretrained(output)
    (output / "training-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "checkpoint": str(output),
                "selected_step": best_step,
                "metrics": metrics,
                "test_instruction_flips": flip_results,
            },
            indent=2,
        ),
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/controlled-demo"))
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/controlled-demo.jsonl")
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device", choices=("auto", "mps", "cuda", "cpu"), default="auto"
    )
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error("--steps and --batch-size must be positive")
    if args.output.exists():
        parser.error(f"Checkpoint directory already exists: {args.output}")
    train(
        output=args.output,
        dataset_path=args.dataset,
        seed=args.seed,
        steps=args.steps,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
