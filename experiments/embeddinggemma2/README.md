# EmbeddingGemma 2 plays Doom zero-shot

[`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2)
(released 2026-10-06, Apache 2.0) embeds text, images, audio and video into one
768-d space. This experiment uses it to play ViZDoom **with no training**: every
decision is a cosine-similarity comparison between crops of the game frame and
fixed text prompts. A small rule controller turns those comparisons into buttons.
No head is trained and the controller reads no game state.

## How a decision is made

Each decision cuts the 640×480 frame into 22 crops and embeds them in one batch:

| Crops | Compared with | Decides |
| --- | --- | --- |
| Left / front / right thirds of the monster band (y 120–245) | monster prompts vs their negations ("a monster" vs "not a monster") | Is any enemy visible? (some third has margin > 0) |
| 17 overlapping 128-px windows, 32 px apart, same band | monster prompts vs scene prompts (wall, floor, sky) | Where is it? Best window plus a parabola through its neighbours gives the bearing |
| HEALTH box of the status bar | "HEALTH 0%" … "HEALTH 100%" | Health |
| AMMO box of the status bar | "AMMO 0" … "AMMO 50" | Ammo |

The instruction is matched once per run against three behaviour descriptions
(aggressive / cautious / evasive). Each side of a contrast is the normalized mean
of its prompts. Every decision is either the sign of a cosine difference or an
argmax across crops; no decision threshold is fitted. The controller has the
same shape as the scripted teacher in `src/doom_bert/teacher.py`: turn toward the
enemy, fire when its bearing is within ±0.12 of the crosshair, and back off
when hurt, out of ammo, or in evasive mode.

The band was chosen from the logged monster boxes (tops ≈ y 191, bottoms ≈ y 233).
It also excludes the pistol, which bobs up to y ≈ 249 and was being shot at as a
"monster".

## Results

All numbers are from an Apple M3 Max (MPS, float32) and are small samples.

**Perception against logged game state**: 150 unique screenshots from earlier
runs, [report](../../reports/embeddinggemma2/zero-shot-probe.json):

| Reading | Result |
| --- | --- |
| AMMO from pixels | 100% exact (43 empty-ammo frames) |
| HEALTH from pixels | mean error 0.5 points; below-40 correct 100% (14 low frames) |
| Instruction → mode | 9/9, including 3 phrasings never written into the prompts |
| Enemy visible | 84.0%, vs 81.3% for always answering yes |
| Bearing error, to the closest monster | median 0.028 (aim window is ±0.12) |
| Turn direction vs teacher | agrees 62.7%, opposite 2.7% |
| Fire decisions | 55.7% of shots have a monster in the aim window; 49.3% of chances taken |

**Gameplay smoke tests**: 10 game seconds each, seed 7, before the band fix
(strips then reached down to y 300):

| Instruction | Kills / deaths | Notes |
| --- | --- | --- |
| Attack enemies on sight. | 2 / 1 | Locked onto and killed the first marine; then fired at the bobbing pistol in an empty room |
| Do not shoot. Evade the enemies. | 0 / 0 | No shots fired (ammo stayed 26); backed away and strafed |

[Attack summary](../../reports/embeddinggemma2/local-attack-summary.json) ·
[evade summary](../../reports/embeddinggemma2/local-evade-summary.json). Decisions
took about 2.2 s each on MPS with 22 crops, so videos use the game clock.

### What did not work

- **Negated prompts alone.** "is a monster" vs "is not a monster" separated monster
  strips from empty ones with AUC 0.62. The model barely distinguishes the
  negation. Contrasting against what is actually in the scene (wall, floor, sky)
  reached AUC 0.96. Ensembles of 3–4 prompts beat single prompts in every set.
- **One fused "situation" embedding.** Interleaving the instruction, game state
  text (`health 60%, ammo 14`) and the view into one input, then picking the
  closest of five command prompts, collapsed: every variant predicted the same
  command for all 300 situations (15.7% vs a 45.7% majority baseline).
  The shared text dominates the embedding. [Report](../../reports/embeddinggemma2/command-probe.json).
- **Whole-frame zero-shot bearing.** Mean-pooling a full frame averages away
  position: left/centre/right/none was right 26% of the time, below the 38%
  majority. Crops fix this.
- **Video direction.** A ball moving left→right and right→left got the same
  similarity to both captions (0.739 vs 0.735).

Exploratory comparisons that led to this layout (7 strips vs thirds vs windows of
128/160/213 px, interpolation) were run as one-off scripts on the same 150
frames and are not committed. They are summarized here: thirds made the negation
contrast usable for presence (90.7% vs 85.3% baseline on that set). 128-px
windows localized best (median error 0.048, 0.038 with interpolation).

## General multimodal checks

`prepare_media.py` builds a labelled set from local macOS assets (stock photos,
`say` speech in English and Spanish, ffmpeg-generated videos), and `run_tests.py`
scores retrieval with known answers. The first MPS float32 run in this session
gave the following (rerun `run_tests.py` to regenerate; that JSON was not kept):

| Test | Result |
| --- | --- |
| Text query → document, incl. Spanish and German queries | 8/8, also at 128 d |
| Caption → photo / photo → caption | 27/27 / 26/27 (cactus → "a green leaf") |
| Speech → transcript, speech → paraphrase (incl. Spanish audio) | 8/8, 8/8 |
| Spoken "A zebra." → photo, among 27 | 6/6 |
| Video → label / label → video | 4/5 / 5/5 |
| Text → correct two-image interleaved pair | 9/9 |

Precision: MPS float32 matched CPU float32 (cosine 1.00000), bfloat16 reached
0.99996. The model card forbids float16 because activations overflow.

## Setup

```sh
cd experiments/embeddinggemma2
uv sync                                 # transformers 5.19.0, sentence-transformers 6.1.0, torch 2.14.1
uv run python probe_zero_shot.py        # perception vs logged state (needs runs/*/states.jsonl)
uv run python play_gemma.py --headless --seconds 60 --instruction "Attack enemies on sight."
```

`transformers` 5.19.0 is the first release with `embedding_gemma2`. Video file
inputs need `torchcodec` with FFmpeg shared libraries. Defaults: CUDA bfloat16,
otherwise float32. The audio encoder is not loaded for Doom.

### Remote GPU

`remote.sh` syncs the package, this directory and the logged screenshots to a
CUDA host, installs with uv, and runs everything headless over SSH:

```sh
./remote.sh sync                                        # HOST=gpu-host by default
./remote.sh probe
./remote.sh play "Attack enemies on sight." 60
./remote.sh play "Do not shoot. Evade the enemies." 60
./remote.sh fetch                                       # copies runs/eg2-* back
```

## Files

| File | Purpose |
| --- | --- |
| `gemma_doom.py` | Crops, prompts, perception, controller, `EmbeddingGemmaPolicy` |
| `play_gemma.py` | Plays through `doom_bert.play` with the dashboard and video |
| `probe_zero_shot.py` | Scores perception against logged game state |
| `probe_commands.py` | The fused situation → command experiment |
| `prepare_media.py`, `run_tests.py` | General multimodal retrieval checks (macOS media) |
| `remote.sh` | Sync, run and fetch on a remote CUDA machine |
