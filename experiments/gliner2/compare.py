"""Evaluate zero-shot Doom classification and uncapped, unrecorded gameplay.

Run separately in each model's locked uv environment. Teacher actions are only
evaluation references; policy implementations never import the teacher.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from doom_bert.play import legal_actions, play
from doom_bert.policy import BUTTONS, decode_scores

ROOT = Path(__file__).resolve().parents[2]
NOVEL_INSTRUCTIONS = {
    "aggressive": [
        "Open fire on the monsters you encounter.",
        "Take down any enemies in your sights.",
    ],
    "evasive": [
        "Keep your finger off the trigger and dodge the monsters.",
        "Avoid enemies without using your weapon.",
    ],
    "cautious": [
        "Engage enemies while healthy; stop firing and fall back below 40 health.",
        "Once your health is under 40, retreat instead of shooting. Otherwise fight.",
    ],
}
GAME_INSTRUCTIONS = {
    "aggressive": "Attack enemies on sight.",
    "evasive": "Do not shoot. Evade the enemies.",
    "cautious": "Shoot at enemies, but retreat when health falls below 40.",
}


def observation(state, *, health=None, ammo=None):
    fields = dict(item.split("=", 1) for item in state.split())
    health = (
        health
        if health is not None
        else {"critical": 10, "wounded": 30, "healthy": 100}[fields["health"]]
    )
    ammo = ammo if ammo is not None else (10 if fields["ammo"] == "loaded" else 0)
    side = fields["nearest_enemy"]
    objects = []
    if side != "none":
        bearing = {"left": -0.5, "center": 0, "right": 0.5}[side]
        distance = {"touching": 100, "near": 200, "medium": 350, "distant": 700}[
            fields["range"]
        ]
        objects.append(
            {
                "id": 1,
                "name": "Enemy",
                "category": "Monster",
                "box": [300 + bearing * 320, 100, 40, 80],
                "position": [distance, 0, 0],
            }
        )
    return {
        "variables": {
            "HEALTH": health,
            "SELECTED_WEAPON_AMMO": ammo,
            "SELECTED_WEAPON": 2,
            "DEAD": 0,
            "POSITION_X": 0,
            "POSITION_Y": 0,
        },
        "screen_width": 640,
        "damage_since_observation": 0,
        "visible_objects": objects,
    }


def evaluate_one(policy, instruction, obs):
    candidates = legal_actions(list(BUTTONS), obs["variables"])
    started = time.perf_counter()
    scores = policy(instruction, obs)
    action = decode_scores(scores, list(BUTTONS), candidates)
    policy.synchronize()
    elapsed = (time.perf_counter() - started) * 1000
    return {
        "scores": scores,
        "action": action,
        "raw_action": [int(scores[b] > 0.5) for b in BUTTONS],
        "classification_ms": elapsed,
        "input_tokens": policy.last_token_count,
        "model_state_text": policy.last_state_text,
    }


def latency(values):
    return {
        "count": len(values),
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p95_ms": sorted(values)[math.ceil(len(values) * 0.95) - 1],
        "classifications_per_second": 1000 / statistics.mean(values),
    }


def agreement(rows):
    output = {"examples": len(rows)}
    for field in ("raw_action", "action"):
        output[field] = {
            "exact_teacher_agreement": sum(r[field] == r["labels"] for r in rows)
            / len(rows),
            "button_teacher_agreement": sum(
                a == b for r in rows for a, b in zip(r[field], r["labels"], strict=True)
            )
            / (len(rows) * len(BUTTONS)),
            "positive_button_f1": {},
            "all_buttons_released_fraction": sum(not any(r[field]) for r in rows)
            / len(rows),
        }
        for i, button in enumerate(BUTTONS):
            tp = sum(r[field][i] and r["labels"][i] for r in rows)
            fp = sum(r[field][i] and not r["labels"][i] for r in rows)
            fn = sum(not r[field][i] and r["labels"][i] for r in rows)
            output[field]["positive_button_f1"][button] = {
                "f1": 2 * tp / max(1, 2 * tp + fp + fn),
                "target_positives": sum(r["labels"][i] for r in rows),
                "predicted_positives": sum(r[field][i] for r in rows),
            }
    return output


def probes():
    center = "health=healthy ammo=loaded nearest_enemy=center range=medium"
    for mode, instruction in GAME_INSTRUCTIONS.items():
        yield f"center_{mode}", instruction, observation(center)
    for i, instruction in enumerate(NOVEL_INSTRUCTIONS["evasive"]):
        yield f"no_fire_paraphrase_{i}", instruction, observation(center)
    for hp in (39, 40):
        yield (
            f"cautious_health_{hp}",
            GAME_INSTRUCTIONS["cautious"],
            observation(center, health=hp),
        )
    for side in ("left", "right"):
        yield (
            f"enemy_{side}",
            GAME_INSTRUCTIONS["aggressive"],
            observation(center.replace("center", side)),
        )
    yield "no_ammo", GAME_INSTRUCTIONS["aggressive"], observation(center, ammo=0)
    yield (
        "no_enemy",
        GAME_INSTRUCTIONS["aggressive"],
        observation("health=healthy ammo=loaded nearest_enemy=none range=unknown"),
    )


def behavior_checks(results):
    raw = {r["id"]: r["scores"] for r in results}
    checks = {
        "fires_at_aligned_enemy_when_aggressive": raw["center_aggressive"]["ATTACK"]
        > 0.5,
        "does_not_fire_when_evasive": raw["center_evasive"]["ATTACK"] <= 0.5,
        "evades_with_movement": any(
            raw["center_evasive"][b] > 0.5
            for b in ("MOVE_LEFT", "MOVE_RIGHT", "MOVE_BACKWARD")
        ),
        "no_fire_paraphrase_0": raw["no_fire_paraphrase_0"]["ATTACK"] <= 0.5,
        "no_fire_paraphrase_1": raw["no_fire_paraphrase_1"]["ATTACK"] <= 0.5,
        "cautious_stops_firing_below_40": raw["cautious_health_39"]["ATTACK"] <= 0.5,
        "cautious_fires_at_40": raw["cautious_health_40"]["ATTACK"] > 0.5,
        "cautious_retreats_below_40": raw["cautious_health_39"]["MOVE_BACKWARD"] > 0.5,
        "turns_left_for_enemy_on_left": raw["enemy_left"]["TURN_LEFT"] > 0.5
        and raw["enemy_left"]["TURN_LEFT"] > raw["enemy_left"]["TURN_RIGHT"],
        "turns_right_for_enemy_on_right": raw["enemy_right"]["TURN_RIGHT"] > 0.5
        and raw["enemy_right"]["TURN_RIGHT"] > raw["enemy_right"]["TURN_LEFT"],
        "raw_no_fire_without_ammo": raw["no_ammo"]["ATTACK"] <= 0.5,
        "raw_no_fire_without_enemy": raw["no_enemy"]["ATTACK"] <= 0.5,
    }
    return {
        "checks": checks,
        "passed": sum(checks.values()),
        "total": len(checks),
        "attack_instruction_score_delta": raw["center_aggressive"]["ATTACK"]
        - raw["center_evasive"]["ATTACK"],
        "scope": "Targeted smoke checks, not a broad generalization benchmark; evaluated before legal masking.",
    }


def gameplay_metrics(path):
    rows = [
        json.loads(line) for line in (path / "states.jsonl").read_text().splitlines()
    ]
    observations = [r for r in rows if r["type"] == "observation"]
    decisions = [r for r in observations if r["action"] is not None]
    ammo_used = sum(
        max(
            0,
            a["variables"]["SELECTED_WEAPON_AMMO"]
            - b["variables"]["SELECTED_WEAPON_AMMO"],
        )
        for a, b in zip(observations, observations[1:], strict=False)
        if a["episode"] == b["episode"]
    )
    mismatches = sum(
        r["action"]
        != decode_scores(
            r["action_scores"],
            r["buttons"],
            legal_actions(r["buttons"], r["variables"]),
        )
        for r in decisions
    )
    return {
        "attack_button_decisions": sum("ATTACK" in r["pressed"] for r in decisions),
        "raw_attack_above_half_decisions": sum(
            r["action_scores"]["ATTACK"] > 0.5 for r in decisions
        ),
        "ammo_consumed_observed": ammo_used,
        "damage_taken_observed": sum(
            r["damage_since_observation"] for r in observations
        ),
        "final_health": observations[-1]["variables"]["HEALTH"]
        if observations
        else None,
        "noop_decisions": sum(not r["pressed"] for r in decisions),
        "model_action_decode_mismatches": mismatches,
        "state_log_sha256": hashlib.sha256(
            (path / "states.jsonl").read_bytes()
        ).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=("small", "base", "modernbert"), required=True
    )
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="float16"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--head", type=Path, help="Optional adapted GLiNER head directory"
    )
    parser.add_argument(
        "--weights", type=Path, help="Optional fine-tuned GLiNER weights (finetune.py)"
    )
    parser.add_argument("--game-seconds", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 19])
    args = parser.parse_args()
    if args.game_seconds < 0:
        parser.error("--game-seconds must be non-negative")
    if (args.head or args.weights) and args.model == "modernbert":
        parser.error("--head and --weights apply only to GLiNER")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(7)
    torch.set_num_threads(4)
    if args.model == "modernbert":
        from doom_bert.model import ModernBertPolicy

        checkpoint = "models/controlled-demo"
        policy = ModernBertPolicy(
            str(ROOT / checkpoint), device=args.device, dtype=args.dtype
        )
        policy.encoder_forward_calls = 0

        def count_forward(module, inputs, output):
            policy.encoder_forward_calls += 1

        policy.model.model.register_forward_hook(count_forward)
    else:
        from gliner_policy import ACTION_DESCRIPTIONS, TASK_INSTRUCTION, GLiNERPolicy

        checkpoint = f"fastino/gliner2.5-{args.model}-v1"
        policy = GLiNERPolicy(
            checkpoint,
            device=args.device,
            dtype=args.dtype,
            head=args.head,
            weights=args.weights,
        )
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "model": args.model,
        "checkpoint": checkpoint,
        "device": args.device,
        "dtype": args.dtype,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "hardware": torch.cuda.get_device_name(0)
        if args.device == "cuda"
        else subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        if platform.system() == "Darwin"
        else platform.processor(),
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("torch", "transformers", "vizdoom")
        },
        "torch_cpu_threads": torch.get_num_threads(),
        "doom_training_examples": 372
        if args.model == "modernbert"
        else policy.doom_training_examples,
        "adapted_head": str(args.head) if args.head else None,
        "fine_tuned_weights": str(args.weights) if args.weights else None,
        "training_scope": policy.training_scope,
        "batch_size": 1,
        "warmup_calls": 10,
        "classification_timing_scope": "input serialization + tokenization + encoder + scores to CPU + identical legal action decoding; warmup excluded",
        "gameplay_conditions": "headless, no video or dashboard; existing state logs and periodic PNGs; 2 game tics per decision; uncapped",
        "teacher_agreement_note": "Agreement with existing scripted training targets, not an objective measure of optimal play. GLiNER receives button meanings, not the teacher rules.",
        "new_instruction_note": "Fixed paraphrases declared before evaluation; same held-out state categories as the original test split.",
        "runtime_teacher": False,
        "runtime_embedding_or_action_cache": False,
    }
    if args.model != "modernbert":
        report.update(
            {
                "action_descriptions": ACTION_DESCRIPTIONS,
                "task_instruction": TASK_INSTRUCTION,
                "gliner2_version": importlib.metadata.version("gliner2"),
                "schema": policy.schema.build(),
                "model_revision": policy.revision,
                "encoder_attention": policy.classifier.model.encoder.config._attn_implementation,
                "total_parameters": sum(
                    p.numel() for p in policy.classifier.model.parameters()
                ),
            }
        )
    dataset_path = ROOT / "data/controlled-demo.jsonl"
    report["dataset_sha256"] = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    rows = [json.loads(line) for line in dataset_path.read_text().splitlines()]
    novel = []
    seen = set()
    for row in rows:
        key = row["state_id"], row["mode"]
        if row["split"] != "test" or key in seen:
            continue
        seen.add(key)
        for instruction in NOVEL_INSTRUCTIONS[row["mode"]]:
            novel.append(
                {**row, "instruction": instruction, "split": "novel_instruction_test"}
            )
    for _, instruction, obs in list(probes())[:1] * 10:
        evaluate_one(policy, instruction, obs)
    rows_out = []
    start_calls = policy.encoder_forward_calls
    with (args.output / "predictions.jsonl").open("w") as log:
        for index, row in enumerate(rows + novel):
            result = {
                **row,
                **evaluate_one(policy, row["instruction"], observation(row["state"])),
            }
            rows_out.append(result)
            log.write(json.dumps(result) + "\n")
            if (index + 1) % 100 == 0:
                print(
                    f"{args.model}: evaluated {index + 1}/{len(rows) + len(novel)}",
                    flush=True,
                )
    report["classification_latency"] = latency(
        [r["classification_ms"] for r in rows_out]
    )
    report["input_tokens"] = {
        "min": min(r["input_tokens"] for r in rows_out),
        "max": max(r["input_tokens"] for r in rows_out),
    }
    report["agreement"] = {
        split: agreement([r for r in rows_out if r["split"] == split])
        for split in ("train", "validation", "test", "novel_instruction_test")
    }
    report["probes"] = [
        {
            "id": name,
            "instruction": instruction,
            "observation": obs,
            **evaluate_one(policy, instruction, obs),
        }
        for name, instruction, obs in probes()
    ]
    report["behavior"] = behavior_checks(report["probes"])
    report["evaluation_encoder_forwards"] = policy.encoder_forward_calls - start_calls
    report["expected_evaluation_encoder_forwards"] = len(rows_out) + len(
        report["probes"]
    )
    if (
        report["evaluation_encoder_forwards"]
        != report["expected_evaluation_encoder_forwards"]
    ):
        raise RuntimeError("Each evaluation must execute exactly one full encoder pass")
    report["gameplay"] = []

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    print(
        json.dumps(
            {
                "model": args.model,
                "latency": report["classification_latency"],
                "behavior": report["behavior"],
            },
            indent=2,
        ),
        flush=True,
    )
    if args.game_seconds:
        for seed in args.seeds:
            for mode, instruction in GAME_INSTRUCTIONS.items():
                path = args.output / f"game-{seed}-{mode}"
                calls = policy.encoder_forward_calls
                summary = play(
                    scenario="defend_the_center",
                    seconds=args.game_seconds,
                    seed=seed,
                    output=path,
                    headless=True,
                    video=False,
                    stats=False,
                    action_tics=2,
                    instruction=instruction,
                    policy=policy,
                )
                summary.update(gameplay_metrics(path))
                summary["mode"] = mode
                summary["encoder_forwards"] = policy.encoder_forward_calls - calls
                if summary["encoder_forwards"] != summary["decisions"]:
                    raise RuntimeError(
                        "Each gameplay decision must execute the encoder"
                    )
                report["gameplay"].append(summary)
                save()
    print(f"Saved {args.output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
