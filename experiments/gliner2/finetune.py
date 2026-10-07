"""Fine-tune GLiNER2.5 on teacher-labelled Doom data through its own scoring path.

Training runs the same calls as inference (collate, encoder, label embeddings,
classifier) without inference mode, so the trained model scores exactly what it
was trained on. ``--scope full`` updates the encoder and head; ``--scope head``
freezes the encoder and adapts only the classifier, as an ablation that
separates the effect of more data from the effect of fine-tuning.

Checkpoints are selected on held-out validation state combinations; the
original test combinations and the novel test phrasings are never seen.

    uv run python finetune.py --model base --scope full --output ../../runs/gliner2-v2/base-full
"""

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from compare import ROOT, observation
from gliner2.classification.scoring import _label_names
from gliner_policy import GLiNERPolicy, describe_state, input_text
from safetensors.torch import save_file

from doom_bert.policy import BUTTONS


def logits_for(policy: GLiNERPolicy, texts: list[str]) -> torch.Tensor:
    """Differentiable twin of ClassificationScorer.batch_score, labels in BUTTONS order."""
    model, processor = policy.classifier.model, policy.classifier.model.processor
    batch = processor.collate_fn_inference(
        [(t, policy.schema.build()) for t in texts], max_len=512
    )
    batch = batch.to(policy.device, None)
    encoded = model.encoder(
        input_ids=batch.input_ids, attention_mask=batch.attention_mask
    ).last_hidden_state
    _, schema_embs = processor.extract_embeddings_from_batch(
        encoded, batch.input_ids, batch
    )
    rows = []
    for i in range(len(texts)):
        names = _label_names(batch.schema_tokens_list[i][0])
        logits = model.classifier(torch.stack(list(schema_embs[i][0][1:]))).squeeze(-1)
        rows.append(logits[[names.index(b) for b in BUTTONS]])
    return torch.stack(rows).float()


@torch.no_grad()
def evaluate(
    policy: GLiNERPolicy, texts: list[str], labels: torch.Tensor, batch_size: int = 64
) -> dict:
    policy.classifier.model.eval()
    logits = torch.cat(
        [
            logits_for(policy, texts[i : i + batch_size])
            for i in range(0, len(texts), batch_size)
        ]
    )
    predicted = (logits.sigmoid() > 0.5).cpu()
    target = labels.bool()
    return {
        "examples": len(texts),
        "exact_teacher_agreement": (predicted == target).all(1).float().mean().item(),
        "button_teacher_agreement": (predicted == target).float().mean().item(),
        "loss": torch.nn.functional.binary_cross_entropy_with_logits(
            logits.cpu(), labels
        ).item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", choices=("small", "base"), required=True)
    parser.add_argument("--scope", choices=("full", "head"), required=True)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data/gliner2-teacher-v2.jsonl"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Split each batch into this many micro-batches",
    )
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=150)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.perf_counter()

    checkpoint = f"fastino/gliner2.5-{args.model}-v1"
    policy = GLiNERPolicy(checkpoint, device="cuda", dtype="float32")
    model = policy.classifier.model
    model.processor.change_mode(
        is_training=False
    )  # inference collation: fixed label order, no sampling
    rows = [json.loads(line) for line in args.data.open()]
    train = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "validation"]
    original = [
        json.loads(line) for line in (ROOT / "data/controlled-demo.jsonl").open()
    ]
    original_validation = [r for r in original if r["split"] == "validation"]

    def texts_labels(items, state_key):
        texts = [
            input_text(
                r["instruction"],
                r["state_text"]
                if state_key
                else describe_state(observation(r["state"])),
            )
            for r in items
        ]
        return texts, torch.tensor([r["labels"] for r in items], dtype=torch.float32)

    train_texts, train_labels = texts_labels(train, True)
    val_texts, val_labels = texts_labels(validation, True)
    orig_val_texts, orig_val_labels = texts_labels(original_validation, False)

    if args.scope == "head":
        model.requires_grad_(False)
        model.classifier.requires_grad_(True)
        groups = [
            {"params": list(model.classifier.parameters()), "lr": args.head_lr * 5}
        ]
    else:
        model.requires_grad_(True)
        head = list(model.classifier.parameters())
        head_ids = {id(p) for p in head}
        groups = [
            {
                "params": [p for p in model.parameters() if id(p) not in head_ids],
                "lr": args.encoder_lr,
            },
            {"params": head, "lr": args.head_lr},
        ]
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    steps = math.ceil(len(train) * args.epochs / args.batch_size)
    warmup = max(1, int(0.06 * steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (
            min(1.0, (s + 1) / warmup) * max(0.0, (steps - s) / max(1, steps - warmup))
        ),
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()

    history = [{"step": 0, "validation": evaluate(policy, val_texts, val_labels)}]
    best = (
        history[0]["validation"]["exact_teacher_agreement"],
        -history[0]["validation"]["loss"],
    )
    best_state, best_step = (
        {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        0,
    )
    order: list[int] = []
    train_started = time.perf_counter()
    for step in range(1, steps + 1):
        if len(order) < args.batch_size:
            fresh = list(range(len(train)))
            random.shuffle(fresh)
            order += fresh
        batch, order = order[: args.batch_size], order[args.batch_size :]
        model.train()
        if args.scope == "head":
            model.encoder.eval()
        optimizer.zero_grad(set_to_none=True)
        # Micro-batches give the same update as one batch, with less activation memory.
        size = math.ceil(len(batch) / args.grad_accum)
        loss_total = 0.0
        for start in range(0, len(batch), size):
            micro = batch[start : start + size]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = logits_for(policy, [train_texts[i] for i in micro])
            loss = (
                loss_fn(logits, train_labels[micro].to("cuda"))
                * len(micro)
                / len(batch)
            )
            loss.backward()
            loss_total += loss.item()
        loss = torch.tensor(loss_total)
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        optimizer.step()
        scheduler.step()
        if step % args.eval_every == 0 or step == steps:
            val = evaluate(policy, val_texts, val_labels)
            history.append({"step": step, "train_loss": loss.item(), "validation": val})
            print(
                f"step {step}/{steps} loss {loss.item():.4f} val exact {val['exact_teacher_agreement']:.3f}",
                flush=True,
            )
            key = (val["exact_teacher_agreement"], -val["loss"])
            if key > best:
                best, best_step = key, step
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
    train_seconds = time.perf_counter() - train_started

    model.load_state_dict(best_state)
    metrics = {
        "validation": evaluate(policy, val_texts, val_labels),
        "original_validation_48": evaluate(policy, orig_val_texts, orig_val_labels),
    }
    saved = (
        best_state
        if args.scope == "full"
        else {k: v for k, v in model.classifier.state_dict().items()}
    )
    save_file(
        {k: v.detach().to(torch.float16).contiguous().cpu() for k, v in saved.items()},
        args.output / "model.safetensors",
    )
    metadata = {
        "base_model": checkpoint,
        "revision": policy.revision,
        "schema": policy.schema.build(),
        "scope": args.scope,
        "training_examples": len(train),
        "dataset": str(args.data.relative_to(ROOT)),
        "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "trainable_parameters": sum(p.numel() for g in groups for p in g["params"]),
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "epochs": args.epochs,
        "steps": steps,
        "batch_size": args.batch_size,
        "gradient_accumulation_micro_batches": args.grad_accum,
        "optimizer": "AdamW, weight decay 0.01, 6% linear warm-up then linear decay, gradient clip 1.0",
        "learning_rates": {"head": args.head_lr * 5}
        if args.scope == "head"
        else {"encoder": args.encoder_lr, "head": args.head_lr},
        "loss": "BCEWithLogitsLoss, unweighted",
        "precision": "bfloat16 autocast on CUDA; weights saved as float16",
        "selected_step": best_step,
        "selection": "Best validation exact agreement, then validation loss; validation states are held-out combinations",
        "train_seconds": train_seconds,
        "wall_seconds": time.perf_counter() - started,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "metrics": metrics,
        "history": history,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: metadata[k]
                for k in ("scope", "selected_step", "steps", "train_seconds", "metrics")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
