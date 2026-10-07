"""Archive the second GLiNER2.5 attempt into reports/gliner2-v2 (run where runs/ lives).

Same metric definitions as archive_results.py, plus training scope, behaviour
checks and explicit conditional probes. Fine-tuned weights are not committed:
finetune.py rebuilds them from data/gliner2-teacher-v2.jsonl in minutes.
"""

import gzip
import hashlib
import io
import json
import shutil
import tarfile

from compare import ROOT

RUNS = {
    "modernbert": ("eval-modernbert", "conditionals-modernbert.json", None),
    "small-zero-shot": ("eval-small-zero-shot", "conditionals-small-zero-shot.json", None),
    "base-zero-shot": ("eval-base-zero-shot", "conditionals-base-zero-shot.json", None),
    "small-head": ("eval-small-head", "conditionals-small-head.json", "small-head"),
    "base-head": ("eval-base-head", "conditionals-base-head.json", "base-head"),
    "small-full": ("eval-small-full", "conditionals-small-full.json", "small-full"),
    "base-full": ("eval-base-full", "conditionals-base-full.json", "base-full"),
}
SHOWCASE = ("gliner2-show-attack", "gliner2-show-evade", "gliner2-show-cautious")


def main() -> None:
    source = ROOT / "runs/gliner2-v2"
    destination = ROOT / "reports/gliner2-v2"
    destination.mkdir(exist_ok=False)
    manifest, summaries = {}, []
    with (destination / "raw-evidence.tar.gz").open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:

                def add(path, arcname):
                    data = path.read_bytes()
                    info = tarfile.TarInfo(arcname)
                    info.size, info.mode = len(data), 0o644
                    archive.addfile(info, io.BytesIO(data))
                    manifest[arcname] = hashlib.sha256(data).hexdigest()

                for name, (folder, conditional, training) in RUNS.items():
                    path = source / folder
                    report = json.loads((path / "report.json").read_text())
                    if len(report["gameplay"]) != 6:
                        raise RuntimeError(f"Incomplete gameplay: {name}")
                    if report["evaluation_encoder_forwards"] != report["expected_evaluation_encoder_forwards"]:
                        raise RuntimeError(f"Incomplete model evaluation: {name}")
                    for game in report["gameplay"]:
                        if game["encoder_forwards"] != game["decisions"] or game["model_action_decode_mismatches"]:
                            raise RuntimeError(f"Gameplay verification failed: {name}")
                    shutil.copyfile(path / "report.json", destination / f"{name}.json")
                    conditionals = json.loads((source / conditional).read_text())
                    shutil.copyfile(source / conditional, destination / f"conditional-{name}.json")
                    meta = None
                    if training:
                        meta = json.loads((source / training / "metadata.json").read_text())
                        shutil.copyfile(source / training / "metadata.json", destination / f"training-{name}.json")
                    for item in [path / "predictions.jsonl", *sorted(path.glob("game-*/*.json*"))]:
                        add(item, f"{name}/{item.relative_to(path)}")
                    games = report["gameplay"]
                    n = sum(g["decisions"] for g in games)
                    summaries.append(
                        {
                            "id": name,
                            "doom_training_examples": report["doom_training_examples"],
                            "training_scope": report.get("training_scope", "trained head on a frozen encoder"),
                            "trainable_parameters": meta["trainable_parameters"] if meta else None,
                            "train_seconds": meta["train_seconds"] if meta else None,
                            "classification_mean_ms": report["classification_latency"]["mean_ms"],
                            "classification_median_ms": report["classification_latency"]["median_ms"],
                            "gameplay_classification_mean_ms": sum(
                                g["mean_decision_ms"] * g["decisions"] for g in games
                            )
                            / n,
                            "headless_decisions_per_second": n / sum(g["wall_seconds"] for g in games),
                            "test_decoded_teacher_agreement": report["agreement"]["test"]["action"][
                                "exact_teacher_agreement"
                            ],
                            "new_instruction_decoded_teacher_agreement": report["agreement"]["novel_instruction_test"][
                                "action"
                            ]["exact_teacher_agreement"],
                            "behavior_checks_passed": report["behavior"]["passed"],
                            "behavior_checks_total": report["behavior"]["total"],
                            "conditional_checks_passed": conditionals["attack_checks_passed"],
                            "conditional_checks_total": conditionals["attack_checks_total"],
                            "gameplay_decisions": n,
                            "gameplay": [
                                {
                                    k: g[k]
                                    for k in (
                                        "seed",
                                        "mode",
                                        "kills",
                                        "deaths",
                                        "ammo_consumed_observed",
                                        "damage_taken_observed",
                                        "noop_decisions",
                                    )
                                }
                                for g in games
                            ],
                        }
                    )
                for run in SHOWCASE:
                    folder = ROOT / "runs" / run
                    if folder.exists():
                        shutil.copyfile(folder / "summary.json", destination / f"{run}-summary.json")
                        add(folder / "states.jsonl", f"{run}/states.jsonl")
    (destination / "evidence-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (destination / "summary.json").write_text(
        json.dumps(
            {
                "policies": summaries,
                "scope": "Second attempt on one RTX 4090: seven pipelines, each 516 evaluation pairs, 11 probes with 12 "
                "behaviour checks, 8 explicit conditional probes, and six 20-game-second headless runs.",
                "caution": "Fine-tuned models trained on 12,000 teacher-labelled examples; ModernBERT keeps its "
                "372-example head. The GPU was shared with another inference service, so timings are indicative.",
            },
            indent=2,
        )
        + "\n"
    )
    print(destination)


if __name__ == "__main__":
    main()
