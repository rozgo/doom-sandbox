"""Preserve this experiment's reports, raw decisions, and adapted heads in Git."""

import gzip
import hashlib
import io
import json
import shutil
import tarfile

from compare import ROOT

RUNS = {
    "modernbert": "gliner2-modernbert-mps-fp16",
    "small-zero-shot": "gliner2-small-mps-fp16",
    "base-zero-shot": "gliner2-base-mps-fp16",
    "small-128": "gliner2-small-trained128-mps-fp16",
    "base-32": "gliner2-base-trained32-mps-fp16",
}


def main():
    destination = ROOT / "reports/gliner2-experiment"
    destination.mkdir(exist_ok=False)
    manifest = {}
    summaries = []
    # Stable gzip timestamp and tar ownership; preserve exact uncompressed bytes.
    with (destination / "raw-evidence.tar.gz").open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for name, folder in RUNS.items():
                    path = ROOT / "runs" / folder
                    report = json.loads((path / "report.json").read_text())
                    if len(report["gameplay"]) != 6:
                        raise RuntimeError(f"Incomplete gameplay: {name}")
                    if (
                        report["evaluation_encoder_forwards"]
                        != report["expected_evaluation_encoder_forwards"]
                    ):
                        raise RuntimeError(f"Incomplete model evaluation: {name}")
                    for game in report["gameplay"]:
                        if (
                            game["encoder_forwards"] != game["decisions"]
                            or game["model_action_decode_mismatches"]
                        ):
                            raise RuntimeError(f"Gameplay verification failed: {name}")
                    shutil.copyfile(path / "report.json", destination / f"{name}.json")
                    evidence = [
                        path / "predictions.jsonl",
                        *sorted(path.glob("game-*/*.json*")),
                    ]
                    for source in evidence:
                        data = source.read_bytes()
                        arcname = f"{name}/{source.relative_to(path)}"
                        info = tarfile.TarInfo(arcname)
                        info.size = len(data)
                        info.mode = 0o644
                        archive.addfile(info, io.BytesIO(data))
                        manifest[arcname] = hashlib.sha256(data).hexdigest()
                    games = report["gameplay"]
                    n = sum(g["decisions"] for g in games)
                    summaries.append(
                        {
                            "id": name,
                            "doom_training_examples": report["doom_training_examples"],
                            "classification_mean_ms": report["classification_latency"][
                                "mean_ms"
                            ],
                            "classification_median_ms": report[
                                "classification_latency"
                            ]["median_ms"],
                            "gameplay_classification_mean_ms": sum(
                                g["mean_decision_ms"] * g["decisions"] for g in games
                            )
                            / n,
                            "headless_decisions_per_second": n
                            / sum(g["wall_seconds"] for g in games),
                            "test_decoded_teacher_agreement": report["agreement"][
                                "test"
                            ]["action"]["exact_teacher_agreement"],
                            "new_instruction_decoded_teacher_agreement": report[
                                "agreement"
                            ]["novel_instruction_test"]["action"][
                                "exact_teacher_agreement"
                            ],
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
    for model in ("small", "base"):
        source = ROOT / "runs" / f"gliner2-{model}-heads"
        target = ROOT / "models" / f"gliner2.5-{model}-heads"
        target.mkdir(exist_ok=False)
        for file in sorted(source.glob("head-*/*")):
            out = target / file.relative_to(source)
            out.parent.mkdir(exist_ok=True)
            shutil.copyfile(file, out)
        shutil.copyfile(
            source / "training-report.json", destination / f"{model}-training.json"
        )
    shutil.copyfile(
        ROOT / "runs/gliner2-cpu-check.json", destination / "cpu-check.json"
    )
    for name in RUNS:
        shutil.copyfile(
            ROOT / "runs" / f"gliner2-conditional-{name}.json",
            destination / f"conditional-{name}.json",
        )
    (destination / "evidence-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (destination / "summary.json").write_text(
        json.dumps(
            {
                "policies": summaries,
                "scope": "Five pipelines, each 516 evaluation pairs, 11 initial probes, and six 20-game-second headless runs; plus explicit conditional follow-up and CPU precision check.",
                "caution": "Small exploratory comparison. Same underlying observations, model-appropriate text formats. Fixed 600-step head adaptation is not an optimized training benchmark. Earlier 44 decisions/s video includes display and encoding overhead and is not directly comparable.",
            },
            indent=2,
        )
        + "\n"
    )
    print(destination)


if __name__ == "__main__":
    main()
