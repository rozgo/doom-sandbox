"""Compare the fine-tuned showcase runs' live decisions with the scripted teacher.

Reads the showcase game logs from reports/gliner2-v2/raw-evidence.tar.gz and
writes reports/gliner2-v2/live-agreement.json: per run, how often the raw
seven-button vector equals the teacher's, which buttons disagree, and how the
cautious run's low-health decisions split by ammo.

    uv run python live_agreement.py
"""

import json
import tarfile
from collections import Counter

from compare import ROOT

from doom_bert.policy import BUTTONS
from doom_bert.teacher import TEACHERS, teacher_decision

RUNS = {"gliner2-show-attack": "aggressive", "gliner2-show-evade": "evasive", "gliner2-show-cautious": "cautious"}


def main() -> None:
    evidence = ROOT / "reports/gliner2-v2/raw-evidence.tar.gz"
    report = {}
    with tarfile.open(evidence) as archive:
        for run, mode in RUNS.items():
            lines = archive.extractfile(f"{run}/states.jsonl").read().decode().splitlines()
            decisions = [
                r for r in map(json.loads, lines) if r.get("type") == "observation" and r.get("action") is not None
            ]
            agree, buttons = 0, Counter()
            for r in decisions:
                scores, _ = teacher_decision(r, TEACHERS[mode])
                teacher = [int(scores[b] > 0.5) for b in BUTTONS]
                model = [int(r["action_scores"][b] > 0.5) for b in BUTTONS]
                agree += teacher == model
                for b, t, m in zip(BUTTONS, teacher, model, strict=True):
                    if t != m:
                        buttons[f"{b} ({'teacher' if t else 'model'} pressed it)"] += 1
            entry = {
                "mode": mode,
                "decisions": len(decisions),
                "raw_vector_equals_teacher": agree,
                "disagreeing_buttons": dict(buttons),
            }
            if mode == "cautious":
                low = [
                    r
                    for r in decisions
                    if r["variables"]["HEALTH"] < 40
                    and any(o.get("category") == "Monster" for o in r["visible_objects"])
                ]
                loaded = [r for r in low if r["variables"]["SELECTED_WEAPON_AMMO"] > 0]
                entry["below_40_health_with_enemy_visible"] = {
                    "decisions": len(low),
                    "with_ammo": len(loaded),
                    "attack_pressed": sum("ATTACK" in r["pressed"] for r in low),
                    "move_backward_pressed": sum("MOVE_BACKWARD" in r["pressed"] for r in low),
                    "note": "In this run health fell below 40 only after the ammo ran out, so these decisions do not "
                    "isolate the 40-health rule; check_conditionals.py tests it with ammo loaded.",
                }
            report[run] = entry
    out = ROOT / "reports/gliner2-v2/live-agreement.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
