"""Disambiguate 'retreat when hurt' with explicit conditional no-fire instructions."""

import argparse
import json
from pathlib import Path

import torch
from compare import ROOT, evaluate_one, observation

INSTRUCTIONS = [
    "Fight, but stop shooting and retreat below 40 health.",
    "Fire only when your health is at least 40. If lower, do not fire and move backward.",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=("small", "base", "modernbert"), required=True
    )
    parser.add_argument("--head", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    torch.set_num_threads(4)
    if args.model == "modernbert":
        from doom_bert.model import ModernBertPolicy

        policy = ModernBertPolicy(str(ROOT / "models/controlled-demo"))
    else:
        from gliner_policy import GLiNERPolicy

        policy = GLiNERPolicy(f"fastino/gliner2.5-{args.model}-v1", head=args.head)
    cases = []
    for instruction in INSTRUCTIONS:
        for hp in (20, 39, 40, 80):
            obs = observation(
                "health=healthy ammo=loaded nearest_enemy=center range=medium",
                health=hp,
            )
            result = evaluate_one(policy, instruction, obs)
            cases.append(
                {
                    "instruction": instruction,
                    "health": hp,
                    **result,
                    "expected_attack": hp >= 40,
                    "correct_raw_attack": (result["scores"]["ATTACK"] > 0.5)
                    == (hp >= 40),
                    "retreats_if_hurt": result["scores"]["MOVE_BACKWARD"] > 0.5
                    if hp < 40
                    else None,
                }
            )
    report = {
        "model": args.model,
        "adapted_head": str(args.head) if args.head else None,
        "cases": cases,
        "attack_checks_passed": sum(c["correct_raw_attack"] for c in cases),
        "attack_checks_total": len(cases),
        "note": "Follow-up diagnostic: the original 'shoot, but retreat below 40' wording did not explicitly forbid firing while retreating. These instructions do. No weights, schemas or thresholds are changed for this check. Health categories still use the existing perception bins; this is not proof of arbitrary numeric reasoning.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
