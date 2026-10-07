"""Local multimodal checks for google/embeddinggemma-2.

Every test has a known answer, so each reports top-1 accuracy and mean
reciprocal rank (MRR) rather than impressions. Embeddings are saved to compare
devices and dtypes against a CPU float32 reference.

    uv run python prepare_media.py
    uv run python run_tests.py --device mps --dtype float32
    uv run python run_tests.py --compare ref.npz other.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import wave
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODEL = "google/embeddinggemma-2"
REPO = Path(__file__).resolve().parents[2]
DEFAULT_MEDIA = REPO / "runs/embeddinggemma2/media"
MONSTERS = {"MarineChainsawVzd", "Demon"}

TEXT_PAIRS = [
    (
        "What causes the northern lights?",
        "Auroras",
        "Charged particles from the sun collide with gases in the upper atmosphere, producing glowing curtains of light near the poles.",
    ),
    (
        "how do I undo my last git commit but keep the changes",
        "Git",
        "Run git reset --soft HEAD~1 to move the branch back one commit while leaving your changes staged.",
    ),
    (
        "best temperature to bake sourdough bread",
        "Baking",
        "Preheat the oven with a Dutch oven inside to about 250 C, then lower it to 230 C after removing the lid.",
    ),
    (
        "symptoms of dehydration in adults",
        "Health",
        "Thirst, dark urine, dizziness, fatigue and dry mouth are common signs that the body lacks water.",
    ),
    (
        "why do cats purr",
        "Cats",
        "Felines produce a low vibrating sound with their laryngeal muscles when content, and sometimes when stressed or healing.",
    ),
    (
        "¿Cuál es la capital de Australia?",
        "Australia",
        "Canberra, not Sydney, is the capital city of Australia and home to Parliament House.",
    ),
    (
        "Wie funktioniert eine Wärmepumpe?",
        "Heat pumps",
        "A heat pump moves heat from outside air or ground into a building using a refrigerant compression cycle.",
    ),
    ("python function to reverse a string", "reverse.py", "def reverse(s: str) -> str:\n    return s[::-1]"),
]


def as_unit(x: np.ndarray, dim: int | None = None) -> np.ndarray:
    x = x[..., :dim] if dim else x
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def retrieval(queries: np.ndarray, docs: np.ndarray, truth: list[int], dim: int | None = None) -> dict:
    sim = as_unit(queries, dim) @ as_unit(docs, dim).T
    ranks = [int((row > row[t]).sum()) + 1 for row, t in zip(sim, truth, strict=True)]
    return {
        "top1": float(np.mean([r == 1 for r in ranks])),
        "mrr": float(np.mean([1 / r for r in ranks])),
        "n_queries": len(truth),
        "n_candidates": docs.shape[0],
        "ranks": ranks,
    }


def read_wav(path: str) -> dict:
    with wave.open(path) as wav:
        assert wav.getnchannels() == 1 and wav.getsampwidth() == 2
        rate = wav.getframerate()
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
    return {"array": samples, "sampling_rate": rate}


def load_rgb(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def doom_frames(limit: int | None) -> list[dict]:
    """Unique gameplay screenshots labelled with the controlled-perception bearing bins."""
    frames, seen = [], set()
    for states in sorted(REPO.glob("runs/*/states.jsonl")):
        for line in states.open():
            record = json.loads(line)
            if record.get("type") != "observation" or not record.get("screen"):
                continue
            path = states.parent / record["screen"]
            if not path.exists():
                continue
            digest = hashlib.md5(path.read_bytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            v = record["variables"]
            width = record.get("screen_width", 640)
            enemies = []
            for obj in record["visible_objects"]:
                if obj.get("category", "Monster" if obj["name"] in MONSTERS else None) != "Monster":
                    continue
                x, _, w, _ = obj["box"]
                distance = float(np.hypot(obj["position"][0] - v["POSITION_X"], obj["position"][1] - v["POSITION_Y"]))
                enemies.append((distance, (x + w / 2 - width / 2) / (width / 2)))
            side = "none"
            if enemies:
                bearing = min(enemies)[1]
                side = "center" if abs(bearing) <= 0.12 else ("left" if bearing < 0 else "right")
            frames.append({"path": str(path), "run": states.parent.name, "side": side, "enemies": len(enemies)})
    return frames[:limit] if limit else frames


class Runner:
    def __init__(self, device: str, dtype: torch.dtype, vision_tokens: int):
        from sentence_transformers import SentenceTransformer

        start = time.perf_counter()
        self.model = SentenceTransformer(MODEL, device=device, model_kwargs={"dtype": dtype})
        self.load_seconds = time.perf_counter() - start
        # Soft tokens per image (70-1120). Video frames keep their own 140-token default.
        self.model[0].processor.image_processor.max_soft_tokens = vision_tokens
        self.device = device
        self.saved: dict[str, np.ndarray] = {}

    def encode(self, key: str, inputs, **kwargs) -> np.ndarray:
        emb = self.model.encode(inputs, convert_to_numpy=True, batch_size=8, **kwargs).astype(np.float32)
        self.saved[key] = emb
        return emb

    def latency(self, inputs, repeats: int = 7, **kwargs) -> dict:
        self.model.encode(inputs, **kwargs)  # warm
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            self.model.encode(inputs, **kwargs)
            times.append((time.perf_counter() - start) * 1e3)
        return {"median_ms": float(np.median(times)), "min_ms": float(np.min(times))}


def test_text(r: Runner) -> dict:
    queries = [q for q, _, _ in TEXT_PAIRS]
    docs = [f"title: {t} | text: {d}" for _, t, d in TEXT_PAIRS]
    q = r.encode("text/queries", queries, prompt_name="SearchQuery")
    d = r.encode("text/docs", docs)
    truth = list(range(len(TEXT_PAIRS)))
    return {
        "retrieval_by_dim": {dim: retrieval(q, d, truth, dim) for dim in (768, 512, 256, 128)},
        "no_prompt": retrieval(r.encode("text/queries_raw", queries), d, truth),
    }


def test_images(r: Runner, manifest: dict) -> dict:
    items = list(manifest["images"].values())
    images = [load_rgb(i["path"]) for i in items]
    captions = [i["caption"] for i in items]
    img = r.encode("image/images", images)
    truth = list(range(len(items)))
    out = {}
    for name, kwargs in {
        "SearchQuery": {"prompt_name": "SearchQuery"},
        "Document": {"prompt_name": "Document"},
        "no_prompt": {},
    }.items():
        txt = r.encode(f"image/captions_{name}", captions, **kwargs)
        out[name] = {
            "text_to_image": retrieval(txt, img, truth),
            "image_to_text": retrieval(img, txt, truth),
        }
    txt = r.saved["image/captions_SearchQuery"]
    out["text_to_image_by_dim"] = {dim: retrieval(txt, img, truth, dim)["top1"] for dim in (768, 512, 256, 128)}
    names = [Path(i["path"]).stem for i in items]
    sim = as_unit(img) @ as_unit(txt).T
    out["image_to_text_misses"] = [
        {"image": names[k], "predicted_caption": captions[int(sim[k].argmax())]}
        for k in range(len(items))
        if int(sim[k].argmax()) != k
    ]
    return out


def test_audio(r: Runner, manifest: dict) -> dict:
    speech = manifest["audio"]["speech"]
    clips = [read_wav(s["path"]) for s in speech]
    audio = r.encode("audio/speech", clips)
    transcripts = r.encode("audio/transcripts", [s["transcript"] for s in speech], prompt_name="SearchQuery")
    paraphrases = sorted({s["paraphrase"] for s in speech})
    para = r.encode("audio/paraphrases", paraphrases, prompt_name="SearchQuery")
    para_truth = [paraphrases.index(s["paraphrase"]) for s in speech]
    out = {
        "speech_to_transcript": retrieval(audio, transcripts, list(range(len(speech)))),
        "transcript_to_speech": retrieval(transcripts, audio, list(range(len(speech)))),
        "speech_to_paraphrase": retrieval(audio, para, para_truth),
        "clips": [s["id"] for s in speech],
    }

    events = manifest["audio"]["sound_events"]
    ev = r.encode("audio/events", [read_wav(e["path"]) for e in events])
    labels = r.encode("audio/event_labels", [e["label"] for e in events], prompt_name="SearchQuery")
    out["sound_event_to_label"] = retrieval(ev, labels, list(range(len(events))))
    out["sound_event_ids"] = [e["id"] for e in events]

    image_items = list(manifest["images"])
    images = r.saved["image/images"]
    spoken = manifest["audio"]["spoken_image_queries"]
    truth = [image_items.index(s["image"]) for s in spoken]
    spoken_emb = r.encode("audio/spoken_image_queries", [read_wav(s["path"]) for s in spoken])
    typed_emb = r.encode("audio/typed_image_queries", [s["transcript"] for s in spoken], prompt_name="SearchQuery")
    out["spoken_query_to_image"] = retrieval(spoken_emb, images, truth)
    out["typed_query_to_image"] = retrieval(typed_emb, images, truth)
    out["spoken_queries"] = [s["transcript"] for s in spoken]
    return out


def test_video(r: Runner, manifest: dict) -> dict:
    out = {}
    for group in ("content", "direction"):
        items = [v for v in manifest["video"] if v["group"] == group]
        vid = r.encode(f"video/{group}", [v["path"] for v in items])
        txt = r.encode(f"video/{group}_labels", [v["label"] for v in items], prompt_name="SearchQuery")
        truth = list(range(len(items)))
        sim = as_unit(vid) @ as_unit(txt).T
        out[group] = {
            "video_to_text": retrieval(vid, txt, truth),
            "text_to_video": retrieval(txt, vid, truth),
            "ids": [v["id"] for v in items],
            "similarity": np.round(sim, 4).tolist(),
        }
    return out


def test_interleaved(r: Runner, manifest: dict) -> dict:
    """Two images in one embedding: can a text query find the right pair?"""
    images = manifest["images"]
    animals = {"zebra": "Animals/Zebra", "penguin": "Animals/Penguin", "parrot": "Animals/Parrot"}
    instruments = {"guitar": "Instruments/Guitar", "piano": "Instruments/Piano", "violin": "Instruments/Violin"}
    pairs = [(a, i) for a in animals for i in instruments]
    docs = [
        {
            "text": "<|image|> <|image|>",
            "image": [load_rgb(images[animals[a]]["path"]), load_rgb(images[instruments[i]]["path"])],
        }
        for a, i in pairs
    ]
    pair_emb = r.encode("interleaved/pairs", docs)
    queries = r.encode("interleaved/queries", [f"a {a} and a {i}" for a, i in pairs], prompt_name="SearchQuery")
    truth = list(range(len(pairs)))
    # Text + image in one input: does the text change what the image embedding means?
    rose = load_rgb(images["Flowers/Red Rose"]["path"])
    framed = r.encode(
        "interleaved/rose_contexts",
        [
            rose,
            {"text": "title: Valentine's Day gift ideas | text: <|image|>", "image": [rose]},
            {"text": "title: Garden pest control | text: aphids on this plant <|image|>", "image": [rose]},
        ],
    )
    probes = r.encode(
        "interleaved/rose_probes", ["romantic gift for a partner", "insects damaging plants"], prompt_name="SearchQuery"
    )
    return {
        "text_to_image_pair": retrieval(queries, pair_emb, truth),
        "pairs": [f"{a}+{i}" for a, i in pairs],
        "rose_context_similarity": {
            "rows": ["image only", "image + valentine text", "image + pest text"],
            "cols": ["romantic gift for a partner", "insects damaging plants"],
            "cosine": np.round(as_unit(framed) @ as_unit(probes).T, 4).tolist(),
        },
    }


def test_doom(r: Runner, limit: int | None) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    frames = doom_frames(limit)
    emb = r.encode("doom/frames", [load_rgb(f["path"]) for f in frames])
    sides = ["left", "center", "right", "none"]
    y = np.array([sides.index(f["side"]) for f in frames])
    groups = np.array([f["run"] for f in frames])
    counts = Counter(f["side"] for f in frames)
    majority = max(counts.values()) / len(frames)

    prompts = [
        "Doom gameplay: a monster on the left side of the screen",
        "Doom gameplay: a monster directly ahead in the center of the screen",
        "Doom gameplay: a monster on the right side of the screen",
        "Doom gameplay: no monsters visible",
    ]
    zs = r.encode("doom/zero_shot_labels", prompts, prompt_name="SearchQuery")
    zero_shot_pred = (as_unit(emb) @ as_unit(zs).T).argmax(1)

    has_enemy = (y != 3).astype(int)
    seen_prompts = r.encode(
        "doom/zero_shot_presence",
        ["Doom gameplay with a monster visible", "Doom gameplay, an empty room with no monsters"],
        prompt_name="SearchQuery",
    )
    presence_pred = ((as_unit(emb) @ as_unit(seen_prompts).T).argmax(1) == 0).astype(int)

    # Linear probes, cross-validated by run so a trajectory never appears in both train and test.
    folds = GroupKFold(n_splits=min(5, len(set(groups))))
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.5))
    emb_pred = cross_val_predict(probe, as_unit(emb), y, cv=folds, groups=groups)
    pixels = np.stack(
        [np.asarray(load_rgb(f["path"]).convert("L").resize((32, 24)), dtype=np.float32).ravel() / 255 for f in frames]
    )
    pix_pred = cross_val_predict(probe, pixels, y, cv=folds, groups=groups)
    return {
        "frames": len(frames),
        "runs": len(set(groups)),
        "label_counts": dict(counts),
        "majority_baseline": majority,
        "zero_shot_side_accuracy": float((zero_shot_pred == y).mean()),
        "zero_shot_side_predictions": dict(Counter(sides[k] for k in zero_shot_pred)),
        "zero_shot_presence_accuracy": float((presence_pred == has_enemy).mean()),
        "presence_majority_baseline": float(max(has_enemy.mean(), 1 - has_enemy.mean())),
        "probe_side_accuracy_embedding": float((emb_pred == y).mean()),
        "probe_side_accuracy_32x24_pixels": float((pix_pred == y).mean()),
    }


def test_latency(r: Runner, manifest: dict) -> dict:
    images = manifest["images"]
    speech = {s["id"]: s for s in manifest["audio"]["speech"]}
    clip = read_wav(speech["train_en"]["path"])
    video = next(v["path"] for v in manifest["video"] if v["id"] == "doom")
    zebra, guitar = load_rgb(images["Animals/Zebra"]["path"]), load_rgb(images["Instruments/Guitar"]["path"])
    return {
        "text_short": r.latency("health=healthy ammo=loaded nearest_enemy=center range=medium"),
        "image_1": r.latency(zebra),
        "image_batch_8": r.latency([zebra] * 8, batch_size=8),
        "audio_speech": {**r.latency(clip), "seconds": len(clip["array"]) / clip["sampling_rate"]},
        "video_doom_16s_at_1fps": r.latency(video, repeats=3),
        "interleaved_2_images": r.latency({"text": "<|image|> <|image|>", "image": [zebra, guitar]}),
    }


def compare(paths: list[Path]) -> None:
    ref = np.load(paths[0])
    for path in paths[1:]:
        other = np.load(path)
        print(f"\n{path.name} vs {paths[0].name}")
        worst = {}
        for key in ref.files:
            a, b = ref[key], other[key]
            nans = int(np.isnan(b).sum())
            cos = (as_unit(a) * as_unit(b)).sum(-1)
            group = key.split("/")[0]
            entry = worst.setdefault(group, [1.0, 0])
            entry[0] = min(entry[0], float(np.nanmin(cos)) if nans < b.size else float("nan"))
            entry[1] += nans
        for group, (cos, nans) in worst.items():
            print(f"  {group:12s} min cosine {cos:.5f}   NaN values {nans}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--media", type=Path, default=DEFAULT_MEDIA)
    parser.add_argument("--output", type=Path, default=REPO / "runs/embeddinggemma2")
    parser.add_argument("--vision-tokens", type=int, default=280, choices=[70, 140, 280, 560, 1120])
    parser.add_argument("--doom-limit", type=int)
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--compare", type=Path, nargs="+")
    args = parser.parse_args()
    if args.compare:
        compare(args.compare)
        return

    manifest = json.loads((args.media / "manifest.json").read_text())
    r = Runner(args.device, getattr(torch, args.dtype), args.vision_tokens)
    report = {
        "model": MODEL,
        "device": args.device,
        "dtype": args.dtype,
        "vision_tokens": args.vision_tokens,
        "torch": torch.__version__,
        "machine": platform.machine(),
        "load_seconds": r.load_seconds,
    }
    # Latency runs first: after hundreds of encodes MPS timings drift upward.
    for name, fn in ([] if args.skip_latency else [("latency", lambda: test_latency(r, manifest))]) + [
        ("text", lambda: test_text(r)),
        ("image", lambda: test_images(r, manifest)),
        ("audio", lambda: test_audio(r, manifest)),
        ("video", lambda: test_video(r, manifest)),
        ("interleaved", lambda: test_interleaved(r, manifest)),
        ("doom", lambda: test_doom(r, args.doom_limit)),
    ]:
        start = time.perf_counter()
        report[name] = fn()
        report[name]["_seconds"] = time.perf_counter() - start
        print(f"{name:12s} done in {report[name]['_seconds']:.1f}s", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    stem = f"{args.device}-{args.dtype}-v{args.vision_tokens}"
    (args.output / f"report-{stem}.json").write_text(json.dumps(report, indent=2, default=str))
    np.savez(args.output / f"embeddings-{stem}.npz", **r.saved)
    print(f"wrote {args.output / f'report-{stem}.json'}")


if __name__ == "__main__":
    main()
