# Doom Bert

ViZDoom with an **uncapped observe → decide → advance one tic** loop, a random
baseline, and a ModernBERT policy adapter accelerated by Apple Metal.
Instructions and structured state enter the model; button probabilities come out.
Pixels are used only for screenshots and video.

The [GLiNER2.5 comparison](experiments/gliner2/RESULTS.md) tests zero-shot action
classification and small imitation-training runs against this ModernBERT policy
on the same Mac. [Experiment setup and reproduction](experiments/gliner2/README.md).

**A small trained checkpoint is included:** `models/controlled-demo`. It learns
seven button scores from known instructions and a controlled description of live
state. The full ModernBERT encoder and trained head run on every decision. The
CLI defaults to random actions; `--checkpoint` selects learned gameplay and
`--benchmark-demo` explicitly selects the earlier untrained-head speed demo.

## Controlled learned demo

[ModernBERT controls Doom at 44 decisions/s (MP4)](media/modernbert-44-decisions-per-second.mp4)

[![ModernBERT controls Doom at 44 decisions per second](media/modernbert-44-decisions-per-second.png)](media/modernbert-44-decisions-per-second.mp4)

The 51-second edit shows the classifier driving game actions from instructions
and structured state. It distinguishes three measurements:

- **About 44 decisions/s:** the complete gameplay loop, including dashboard and
  recording overhead; the attack run measured 43.7 decisions/s.
- **13.9 ms per classification:** mean time for text preparation, tokenization,
  model inference and transfer, and action decoding. This is part of each loop.
- **30 FPS video:** the recording's frame rate, independent of the decision rate.

Both complete gameplay recordings are included. At **0:29**, changing the
instruction changes the classifier's scores for the same initial state; the
second run starts at **0:34**. These are separate seeded runs. Gameplay keeps its
recorded wall-clock speed, resampled to 30 fps.
See the [edit timeline and verification](reports/modernbert-44-decisions-per-second.json).

Original clips: [attack](media/trained-modernbert-demo.mp4) ·
[evade](media/trained-modernbert-evasive.mp4).

| Instruction | Game time | Actual playback | Decisions/s | Mean decision | Result |
| --- | --- | --- | --- | --- | --- |
| Attack enemies on sight. | 60 s | 24.0 s | 43.7 | 13.9 ms | 16 kills, 1 death |
| Do not shoot. Evade the enemies. | 30 s | 12.1 s | 43.3 | 13.9 ms | No shots, no deaths |

Both use the same checkpoint on MPS float16 with the live window and hardware
MP4 recorder enabled. The initial structured game state is identical: changing
only the instruction changes ATTACK from **0.979** to **0.026**. In the evasive
run, ammo remains at 26 and ATTACK never exceeds 0.036; the no-fire behavior comes
from model classification. Reports include [gameplay and video verification](reports/trained-modernbert-demo.json),
[evasive verification](reports/trained-modernbert-evasive.json), and the
[initial-state instruction contrast](reports/trained-instruction-contrast.json).

```sh
git lfs pull
uv sync --locked --extra model
uv run --extra model doom-bert --checkpoint models/controlled-demo \
  --instruction "Attack enemies on sight." --action-tics 2 --seconds 60

# Same trained model, opposite instruction
uv run --extra model doom-bert --checkpoint models/controlled-demo \
  --instruction "Do not shoot. Evade the enemies." --action-tics 2 --seconds 30

# Reproduce the small training experiment in a new directory
uv run --extra model doom-bert-train --output models/my-experiment \
  --dataset runs/my-experiment-data.jsonl
```

To reproduce the video edit with new output filenames, install FFmpeg (including
`ffprobe`) and run:

```sh
uv run --extra model python scripts/edit_strategy_demo.py \
  --output media/my-classifier-edit.mp4 --work-dir runs/my-classifier-edit
```

The editor uses Pillow for charcoal title cards and FFmpeg for overlays, video
encoding, concatenation, and chapter metadata. Encoding defaults to macOS
VideoToolbox; `--encoder libx264` selects software encoding.

Code turns the nearest visible monster's geometry into `left`, `center`, or
`right`, bins distance and health, and reports whether ammo is available. An
example input is `health=healthy ammo=loaded nearest_enemy=center range=medium`.
These are perception features; the text contains no button commands. The
checkpoint selects this serializer through its saved configuration.

Offline rules generated [468 labeled examples](data/controlled-demo.jsonl) for
three behaviors and two known phrasings per behavior. We trained **595,975 head
parameters** and froze the encoder, preserving the complete **149.6M parameter**
inference architecture. Encoder outputs were cached only during training.
Gameplay imports no teacher, uses no embedding or action cache, and executes the
highest-scoring legal vector from the model's seven sigmoid outputs.

Whole state combinations are split across training (62 combinations / 372
examples), validation (8 / 48), and test (8 / 48). The selected head gets **81.25%
exact vector accuracy** and **97.02% individual-button accuracy** on the test set.
The saved float16 checkpoint was reloaded and verified with 48 complete encoder
forwards. For one held-out aligned, wounded state, ATTACK scores **0.7565** under
the aggressive instruction and **0.0029** under the evasive instruction.

This deliberately small demonstration covers a finite vocabulary and six known
instructions. It does not establish general Doom skill or arbitrary language
understanding. The test set has only two positive ATTACK examples and no positive
MOVE_FORWARD examples. See the [training report](models/controlled-demo/training-report.json),
[reload verification](reports/trained-model-verification.json), and
[checkpoint notes](models/controlled-demo/README.md) for the complete results.

The two-tic action interval requires **17.5 decisions/s** to match the engine's
35 tics/s. Decisions remain uncapped and each receives a fresh observation. Use
`--action-tics 1` to decide on every game tic instead. Training changes weights;
the full inference work still happens on each call.

## Live decision dashboard

[Watch the earlier untrained dashboard demo (MP4)](media/modernbert-charcoal-demo.mp4).

[![ModernBERT decision dashboard with live scores and selection explanations](media/modernbert-charcoal-demo.png)](media/modernbert-charcoal-demo.mp4)

This recording contains **2,100 decisions across 60 game seconds**, played back
in **86.5 seconds of actual wall time**. On the M3 Max using MPS float16, mean
decision latency was **30.8 ms** and the complete loop averaged **24.3 decisions/s**
with the live window and video capture enabled. This run fell below Doom's native
35 tics/s; the visible clocks and measured throughput retain that difference.
[Run report and video verification](reports/modernbert-charcoal-demo.json).

```sh
uv run --extra model doom-bert --benchmark-demo --seconds 60 \
  --instruction "Shoot at enemies, but retreat when health falls below 40."
```

The charcoal window puts Doom on the left and instrumentation on the right:

- Median and p95 decision latency over the last 60 decisions, including text
  preparation, tokenization, model inference, transfer, and action decoding.
- Actual full-loop decisions per wall-clock second, including rendering and
  recording, measured between completed decisions.
- All seven sigmoid button scores, a 0.5 marker, and executed-button indicators.
  Cyan marks executed buttons; amber marks above-threshold buttons suppressed by
  the legal-action constraints.
- Legal combination count, the chosen combination's product score, and explicit
  explanations for conflicting directions or unavailable firing.
- Instruction, input token count, Metal/dtype, game and wall clocks, and the
  untrained action-head status.

The same layout is recorded into the MP4. The default `--video-clock wall` uses
actual monotonic capture timestamps, so playback demonstrates measured speed.
Video/preview capture is capped at 30 Hz without sleeping or capping decisions;
wall-clock drawing and encoding run in a background worker with one pending
frame. If display work falls behind, stale queued frames are discarded while
their timestamps preserve actual playback timing. The model never waits for
encoding. Panel text refreshes at 10 Hz for readability. `--no-stats` records just
the game.
`--headless` hides the window while retaining the recorded dashboard.

## Setup and play

Install [uv](https://docs.astral.sh/uv/) and [Git LFS](https://git-lfs.com/):

```sh
git lfs install --local
git lfs pull
uv sync --locked --extra model
uv run doom-bert --seconds 60
```

Python 3.12, ViZDoom, video encoding, and model dependencies are managed by uv.
ViZDoom supplies the engine, scenarios, and Freedoom assets. The `model` extra
adds PyTorch and Transformers; `uv sync --locked` alone runs the random baseline.

The game runs as fast as decisions, rendering, logging, and recording allow.
`--seconds` means **game seconds**, not wall time. One action is applied for one
game tic; the next decision always sees a fresh state. The engine waits during
inference, so there is no queue of outdated model actions. Default MP4 playback
matches actual wall time. Use `--video-clock game` for a fixed 35 fps replay at
normal game speed, independent of inference speed.

```sh
# Maximum throughput without a window or video encoding
uv run doom-bert --headless --no-video --seconds 60

# Watch gameplay at normal wall-clock speed
uv run doom-bert --realtime --seconds 60

# Hold each action for four tics
uv run doom-bert --action-tics 4 --seconds 60

# Reproduce the original one-decision-per-second behavior
uv run doom-bert --realtime --action-tics 35 --video-clock game --seconds 60 --seed 7

# Movement arena
uv run doom-bert --scenario deadly_corridor --seconds 60
```

`--fast` remains an alias for the default uncapped mode. Ctrl+C preserves captures
and finalizes the video. Each run needs a new output directory; `--output` can
choose one explicitly.

## ModernBERT on macOS

```sh
# Benchmark the pretrained encoder + untrained action head, not gameplay ability
uv run --extra model doom-bert-benchmark --devices cpu mps

# Use your own fine-tuned checkpoint:
uv run --extra model doom-bert --checkpoint models/my-policy \
  --instruction "shoot at enemies but retreat below 40 health" \
  --device auto --seconds 60
```

Device selection prefers MPS (Apple Metal), then CUDA, then CPU. Force it with
`--device`. MPS/CUDA default to float16; CPU defaults to float32. Override with
`--dtype float32`, `float16`, or `bfloat16`. Inference uses SDPA attention and
`torch.inference_mode()`, without CUDA-only FlashAttention or an assumed
compilation backend. GPU results synchronize when scores return to CPU.
Benchmark timings include serialization, tokenization, transfer, and inference;
initial model download/loading and warmup are excluded.

Measured on this M3 Max (128 GB unified memory): CPU float32 averaged **43.3 ms**
per decision, MPS float16 **13.3 ms**, and MPS bfloat16 **34.2 ms** on the same
batch-one synthetic observation workload. Float16 is the default based on these
results. The complete MPS game loop with hardware MP4 encoding completed 175
decisions in 3.72 seconds (**47 decisions/s**), including first-call overhead.
These are hardware checks with an untrained action head, not policy-quality
results: [CPU/MPS benchmark](reports/modernbert-mac-benchmark.json),
[bfloat16 comparison](reports/modernbert-mps-bfloat16.json),
[game loop measurement](reports/modernbert-mps-smoke.json).

The checkpoint must use `problem_type="multi_label_classification"` and this
`id2label` order: `ATTACK`, `MOVE_LEFT`, `MOVE_RIGHT`, `MOVE_FORWARD`,
`MOVE_BACKWARD`, `TURN_LEFT`, `TURN_RIGHT`. These seven controls are enabled for
model gameplay. The original numeric adapter serializes health, ammo, recent
damage, enemy count, and the three nearest visible monsters' screen positions
and distances. The included trained checkpoint uses the categorical format
described above. Both filter by object category. No screen buffer enters the
tokenizer.

The decoder chooses the highest-probability legal combination, excluding opposing
directions and ammo-dependent firing without ammo. Instruction following is
learned; the decoder does not independently interpret language prohibitions.

Apple VideoToolbox performs hardware H.264 encoding on macOS when available;
other platforms or unavailable hardware use libx264. The selected encoder is
recorded in the summary. Audio is disabled. No separate FFmpeg install is needed.

## Captures and measurements

Every run is stored under `runs/<UTC timestamp>/`:

| File | Contents |
| --- | --- |
| `states.jsonl` | Every decision's observation, raw scores, buttons, selection explanation, rolling metrics, rewards, and episode-end events |
| `frames/<tic>.png` | RGB screenshot approximately once per game second, plus the final active observation |
| `config.json` | Scenario, seed, instruction, policy device, dtype, and control interval |
| `summary.json` | Decisions/second, mean decision latency, game/wall duration, kills, deaths, video frames and encoder |
| `replay.mp4` | 1440×900 dashboard (640×480 with `--no-stats`), H.264, actual wall-clock timing by default |

The `second` field is game time; `wall_seconds` records actual elapsed time.
`screen` is null between screenshot captures. Frame paths are relative to the run
directory. Logs are buffered and flushed at screenshot boundaries. Screenshots
and terminal output do not run on every inference. Rendering, disk I/O, and
encoding still consume time; `--headless --no-video` removes window/video overhead.
`mean_decision_ms` measures policy selection and decoding;
`decisions_per_wall_second` includes the rest of the gameplay loop.

Episodes restart at the next decision after death or completion. With multi-tic
actions, the final image is held until that action interval ends. A final
observation or terminal event is logged without a further action. A partial
final action is shortened to preserve the requested game duration. With
`--video-clock game`, this also preserves the fixed-rate video duration.

The original 1 Hz random demo recorded **10 kills, 5 deaths, and 61 observations**
in 60 seconds: [MP4](media/first-demo.mp4), [live metrics](media/first-demo.json),
[video verification](media/first-video.json). Its seed was 7; changing the decision
interval changes the trajectory.

## Development

```sh
uv run --extra model ruff check src tests
uv run --extra model ruff format --check src tests
uv run --extra model pytest -q
git lfs ls-files
```

Tests launch the real engine and check fresh one-tic policy observations without
sleeping, legacy 35-tic spacing, seed reproducibility, action masks, object
filtering, screenshots, episode restart, output preservation, and decoded MP4
frame count and duration, including partial action intervals. Additional tests
check rolling throughput arithmetic, selection explanations, preservation of
irregular wall-clock video timestamps, and that a blocked encoder cannot block
frame submission and retains only the latest pending frame.
Training tests check that whole state combinations remain disjoint across data
splits and that live numeric observations map to the controlled perception
vocabulary.

Media, model weights, and checkpoint tokenizer data use Git LFS. Code,
configuration, labeled examples, reports, and `uv.lock` stay in Git.
`runs/`, `.venv/`, caches, and local environment files are ignored. Copy a
capture worth retaining into `media/` and add it normally to track it with LFS.

References: [ViZDoom game control](https://vizdoom.farama.org/api/python/doom_game/),
[ModernBERT](https://huggingface.co/docs/transformers/model_doc/modernbert),
[PyTorch MPS](https://docs.pytorch.org/docs/2.14/notes/mps.html).
