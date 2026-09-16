# Doom Bert

ViZDoom with an **uncapped observe → decide → advance one tic** loop, a random
baseline, and a ModernBERT policy adapter accelerated by Apple Metal.
Instructions and structured state enter the model; button probabilities come out.
Pixels are used only for screenshots and video.

**No trained Doom policy checkpoint is included yet.** The pretrained ModernBERT
encoder is downloaded and its seven-button head is initialized only for hardware
benchmarking. Normal model gameplay requires saved action-head weights and named
action labels. The CLI defaults to random actions; `--checkpoint` selects a trained
policy and `--benchmark-demo` explicitly selects the untrained-head speed demo.

## Live decision dashboard

[Watch the charcoal dashboard demo (MP4)](media/modernbert-charcoal-demo.mp4).

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

# Once a fine-tuned checkpoint has been produced:
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
model gameplay. The adapter serializes health, ammo, recent damage, enemy count,
and the three nearest visible monsters' screen positions and distances. It
filters by object category. No screen buffer enters the tokenizer.

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

Media extensions use Git LFS. Code, configuration, reports, and `uv.lock` stay in
Git. `runs/`, `.venv/`, caches, and local environment files are ignored. Copy a
capture worth retaining into `media/` and add it normally to track it with LFS.

References: [ViZDoom game control](https://vizdoom.farama.org/api/python/doom_game/),
[ModernBERT](https://huggingface.co/docs/transformers/model_doc/modernbert),
[PyTorch MPS](https://docs.pytorch.org/docs/2.14/notes/mps.html).
