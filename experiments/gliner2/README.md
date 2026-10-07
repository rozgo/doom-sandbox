# GLiNER2.5 Doom experiment

Compare schema-conditioned GLiNER2.5 classification with the existing trained
ModernBERT policy. The first attempt produced no video; the second attempt (below) does. The experiment measures actual model
scores, targeted behavior checks, and short ViZDoom rollouts.

## Second attempt: fine-tuning on a CUDA GPU

Run from `experiments/gliner2` on a machine with an NVIDIA GPU (the reported run
used one RTX 4090):

```sh
uv sync
./run_v2.sh data      # data/gliner2-teacher-v2.jsonl: 12,000 + 1,200 teacher-labelled examples
./run_v2.sh train     # Small and Base, full fine-tuning and head-only ablations
./run_v2.sh eval      # all seven pipelines, conditional probes, six games each
./run_v2.sh videos base base-full && ./run_v2.sh edit base-full
uv run python archive_v2.py && uv run python live_agreement.py   # reports/gliner2-v2
```

`finetune.py` trains through the same calls as `Classifier.score` (checked to give
identical logits), selects on held-out validation states, and saves weights that
`compare.py --weights`, `check_conditionals.py --weights` and `play_gliner.py`
load. Weights are not committed. Results: [RESULTS.md](RESULTS.md).

## Reproduce (first attempt)

Run from the repository root. GLiNER2 2.0.0 requires Transformers 4; ModernBERT
uses the existing Transformers 5 environment. The nested uv project has its own
lockfile and Python 3.12 environment. Both use PyTorch 2.14.0.

```sh
uv sync --project experiments/gliner2 --locked
uv run --project experiments/gliner2 python experiments/gliner2/compare.py \
  --model small --output runs/gliner2-small-new
uv run --project experiments/gliner2 python experiments/gliner2/compare.py \
  --model base --output runs/gliner2-base-new
uv run --extra model python experiments/gliner2/compare.py \
  --model modernbert --output runs/gliner2-modernbert-new
```

Each command refuses an existing output directory. Defaults are MPS float16,
four CPU threads, 10 warmup calls, 516 evaluation calls, 11 counterfactual probe
inputs with 12 checks, then six 20-game-second rollouts: three instructions,
seeds 7 and 19, two tics per decision. All runs are synchronous, batch one, and
uncapped. Video and dashboard are disabled for every model. The existing runner
still logs decisions and saves a PNG approximately once per game second.

The separate CPU float32 check measures 55 calls after warming all probe inputs
and compares its scores with the MPS float16 results:

```sh
uv run --project experiments/gliner2 python experiments/gliner2/check_backends.py \
  --output runs/gliner2-cpu-check.json
```

## Inputs and scoring

GLiNER receives a fixed schema describing what the seven buttons do. The schema
contains no conditional gameplay rules. Each input contains the user's
instruction and a short factual state description. Health and ammo are numerical;
enemy bearing and range use the same perception categories as the ModernBERT
demo. ModernBERT retains its trained categorical serializer. Consequently the
inputs describe the same observations but differ in wording and token length;
this compares usable pipelines, not equal-length encoder kernels.

`Classifier.score` invokes the full encoder and pretrained classification head,
returning seven sigmoid scores. The unchanged shared decoder chooses the most
likely legal button vector, blocking contradictory directions and attacks
without ammo. There is no instruction-specific action override. The report
evaluates raw scores as well as decoded actions, so the ammo mask cannot hide
whether the model itself understood an empty weapon.

All encoder calls are counted. Every scored input and every gameplay decision
must execute a full encoder forward pass. No features or actions are cached in
evaluation or gameplay. Compiling the fixed schema only caches its definition.

## Evaluation scope

- The 468 original synthetic pairs retain their original whole-state train,
  validation and test splits. The test set contains only eight state categories
  and 48 pairs; it has two positive ATTACK examples and no positive MOVE_FORWARD
  examples. Exact vector agreement and positive-button F1 are both reported.
- Another 48 pairs use six fixed new instruction phrasings on those eight test
  states. These are an exploratory extension, not a large independent benchmark.
- Eleven probes check opposite instructions, two additional no-fire phrasings,
  health 39 versus 40, left versus right, empty ammo and no visible enemy. The
  12 checks use raw scores before masking. Passing all checks is not a guarantee
  of general instruction following.
- The original cautious instruction says to retreat below 40 but does not
  explicitly prohibit shooting while retreating. Its stop-firing check measures
  teacher-policy agreement. `check_conditionals.py` separately checks two
  explicit conditional no-fire instructions at health 20, 39, 40 and 80 without
  changing any weights, schema or thresholds. The second phrasing is new.
- Teacher agreement measures imitation of the existing scripted policy, not
  optimal gameplay. Zero-shot models are never supplied those teacher rules.
- Short gameplay runs report kills, deaths, raw/selected attacks, observed ammo
  consumption, observed damage, no-ops, and throughput. Damage and ammo totals
  are observed between logged states and can omit terminal transitions.
- Fresh headless throughput is comparable across these runs. It should not be
  directly compared with the older **44 decisions/s video demo**, which also ran
  the live dashboard and recorder.

## Small training experiment

```sh
uv run --project experiments/gliner2 python experiments/gliner2/train_heads.py \
  --model small --output runs/gliner2-small-heads-new
uv run --project experiments/gliner2 python experiments/gliner2/train_heads.py \
  --model base --output runs/gliner2-base-heads-new
uv run --project experiments/gliner2 python experiments/gliner2/compare.py \
  --model small --head runs/gliner2-small-heads-new/head-128 \
  --output runs/gliner2-small-adapted-new
```

The encoder stays frozen. Only the existing shared classification MLP is
adapted, starting from the published weights. Budgets 32, 128 and 372 are nested,
deterministic, behavior-balanced subsets of the original training split. Each
gets 600 AdamW steps with identical settings; weights are selected by validation
exact agreement, with validation BCE breaking ties. Test data and new instruction
phrasings never select weights. For gameplay, use the smallest budget achieving
the best validation exact agreement, regardless of its test result.

Encoder features are cached only for offline training and cached-feature
evaluation. A reconstruction check verifies their label order and agreement
with the original model. Reload a saved head through `compare.py` to obtain
full-forward evaluation and gameplay results; cached-feature results alone do
not establish live behavior. Head optimization time is reported separately from
feature extraction and model loading, so it is not equivalent to the earlier
3.6-second ModernBERT training measurement.

## Dependencies and provenance

Published checkpoints are pinned to revisions in `gliner_policy.py`. Downloaded
base weights remain in the Hugging Face cache; adapted heads are small standalone
Safetensors files layered over those exact checkpoints. `sentencepiece` and
`protobuf` are explicit dependencies: protobuf allows the upstream tokenizer
compatibility handler to recover from legacy special-token metadata.

The GLiNER DeBERTa encoder falls back from requested SDPA to eager attention in
this Transformers version. It still runs on Apple MPS. FlashDeBERTa's NVIDIA
kernels are not used. The ModernBERT baseline retains its existing SDPA path.

Sources: [GLiNER2 repository](https://github.com/fastino-ai/GLiNER2),
[Small checkpoint](https://huggingface.co/fastino/gliner2.5-small-v1),
[Base checkpoint](https://huggingface.co/fastino/gliner2.5-base-v1).
Upstream code and checkpoints are Apache-2.0 licensed.
