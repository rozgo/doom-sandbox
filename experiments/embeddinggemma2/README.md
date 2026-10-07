# EmbeddingGemma 2 plays Doom zero-shot

[`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2)
(released 2026-10-06, Apache 2.0) embeds text, images, audio and video into one
768-d space. This experiment uses it to play ViZDoom **with no training**: every
decision is a cosine-similarity comparison between crops of the game frame and
fixed text prompts. A small rule controller turns those comparisons into buttons.
No head is trained, no threshold is fitted, and the policy reads no game state.

[![EmbeddingGemma 2 plays Doom with live cosine readings](../../media/embeddinggemma2-plays-doom.png)](../../media/embeddinggemma2-plays-doom.mp4)

[Showcase video (MP4, 2:23)](../../media/embeddinggemma2-plays-doom.mp4): how it works, a
full 60-second attack run and a full 60-second evade run with every cosine reading
on screen, then results. Recorded on an RTX 4090 and played back at normal game
speed. [Chapters and sources](../../reports/embeddinggemma2/showcase.json).

## How a decision is made

Each decision cuts the 640×480 frame into 19 crops and embeds them in one batch
(👁 = image input, # = text prompt):

| 👁 Crops | # Contrast (each side: mean of its prompts) | Decides |
| --- | --- | --- |
| 17 overlapping 128-px windows, 32 px apart, across the band where monsters stand (y 120–245) | "a monster", "a pink demon", "a zombie soldier" vs wall, floor, sky | Visible if any window's margin > 0; the best window, refined by a parabola through its neighbours, gives the bearing |
| HEALTH box of the status bar | "HEALTH 0%" … "HEALTH 100%" | Health (argmax) |
| AMMO box of the status bar | "AMMO 0" … "AMMO 50" | Ammo (argmax) |

The instruction is matched once per run against three behaviour descriptions
(text vs text cosine). **Attack/cautious** follow the scripted teacher's shape in
`src/doom_bert/teacher.py`: turn toward the enemy, fire when its bearing is within
±0.12 of the crosshair, back off when hurt or out of ammo. **Evade** is its own
policy with its own contrasts on the same windows:

| Evade contrast | Positive prompts | Negative prompts | Used for |
| --- | --- | --- | --- |
| danger (best threat window) | "a monster right in front of you, up close", … | "a small monster far away", … | close: face it, back off, strafe away |
| escape (every window) | "an empty open floor", "a clear path …", … | the monster prompts | far: turn toward open space |

The player's weapon is not rendered (`set_render_weapon(False)`), and the policy
never fires in evade mode. The HUD and the game are otherwise unchanged.

## Results

RTX 4090, CUDA bfloat16, seed 7, `defend_the_center`, two tics per decision,
60 game seconds per run. Small samples.

| Instruction | Kills | Deaths | ms / decision |
| --- | ---: | ---: | ---: |
| Attack enemies on sight. | 37 | 2 | 287 |
| Do not shoot. Evade the enemies. | 0 | 0 | 275 |
| Shoot at enemies, but retreat when health falls below 40. | 37 | 2 | 269 |

The evade run survived all 60 seconds without firing. For reference, the trained ModernBERT checkpoint, which reads structured state
text rather than pixels, scored 16 kills and 1 death in its 60-second attack run.
The cautious run matched an earlier attack run exactly (37 / 2): with this seed
its 40-health rule never changed a decision. [Summaries](../../reports/embeddinggemma2/).

**Perception against logged game state**: 150 unique screenshots,
[report](../../reports/embeddinggemma2/zero-shot-probe-gpu.json):

| Reading | Result |
| --- | --- |
| AMMO from pixels | 99.3% exact |
| HEALTH from pixels | mean error 1.0 point |
| Instruction → mode | 9/9, including 3 phrasings never written into the prompts |
| Enemy visible | 80.7%, vs 75.3% for always answering yes; 94.6% of empty views rejected |
| Bearing error, to the closest monster | median 0.025 (aim window is ±0.12) |
| Fire decisions | 89.3% of shots have a monster in the aim window |
| Turn direction vs teacher | agrees 60.7%, opposite 9.3% |

Each decision takes about 0.27 s on the 4090 (19 crops, bfloat16); videos use the
game clock, so they play at normal game speed.

### How it got here, and what did not work

- **Negated prompts.** "is a monster" vs "is not a monster" separated monster
  crops from empty ones with AUC 0.62. Even with ensembles, the negation
  contrast answered "monster" for nearly every empty view. Contrasting against
  what is actually in the scene (wall, floor, sky) works.
- **Firing at its own gun.** With the negation presence rule, the first 60-second
  attack run scored 12 kills and 7 deaths: no monster was in view for 868 of
  1,050 decisions, the policy saw one anyway in 860, and it held ATTACK 96% of
  the time. Each shot raised the pistol and its muzzle flash into the centre
  crops, which matched "a monster". Hiding the weapon and the scene-contrast
  presence rule fixed it (37 kills, 2 deaths).
- **Evade as attack-without-firing.** With the new presence rule, an evade
  policy that only backed off when it saw a monster died 5 times in 60 seconds.
  It now circles backwards while scanning and uses its own danger/escape
  contrasts.
- **One fused "situation" embedding.** Interleaving the instruction, game state
  text (`health 60%, ammo 14`) and the view into one input, then picking the
  closest of five command prompts, collapsed: every variant predicted the same
  command for all 300 situations (15.7% vs a 45.7% majority baseline). The shared
  text dominates the embedding. [Report](../../reports/embeddinggemma2/command-probe.json).
- **Whole-frame bearing.** Mean-pooling a full frame averages away position
  (26% on left/centre/right/none, below the 38% majority). Crops fix this.
- **Denser windows.** A 16-px stride (38 crops) did not beat 32 px.
- **Video direction.** A ball moving left→right and right→left got the same
  similarity to both captions (0.739 vs 0.735).

The earlier Apple-silicon runs (MPS float32, 2.2 s per decision) and the
negation-presence probes are kept in `reports/embeddinggemma2/` for comparison.

## General multimodal checks

`prepare_media.py` builds a labelled set from local macOS assets (stock photos,
`say` speech in English and Spanish, ffmpeg-generated videos), and `run_tests.py`
scores retrieval with known answers. The first MPS float32 run gave the following
(rerun `run_tests.py` to regenerate; that JSON was not kept):

| Test | Result |
| --- | --- |
| Text query → document, incl. Spanish and German queries | 8/8, also at 128 d |
| Caption → photo / photo → caption | 27/27 / 26/27 (cactus → "a green leaf") |
| Speech → transcript, speech → paraphrase (incl. Spanish audio) | 8/8, 8/8 |
| Spoken "A zebra." → photo, among 27 | 6/6 |
| Video → label / label → video | 4/5 / 5/5 |
| Text → correct two-image interleaved pair | 9/9 |

MPS float32 matched CPU float32 (cosine 1.00000), bfloat16 reached 0.99996. The
model card forbids float16 because activations overflow.

## Setup

```sh
cd experiments/embeddinggemma2
uv sync                                 # transformers 5.19.0, sentence-transformers 6.1.0, torch 2.14.1
uv run python probe_zero_shot.py        # perception vs logged state (needs runs/*/states.jsonl)
uv run python play_gemma.py --headless --seconds 60 --instruction "Attack enemies on sight."
```

`transformers` 5.19.0 is the first release with `embedding_gemma2`. Defaults:
CUDA bfloat16, otherwise float32; the audio encoder is not loaded for Doom.
`--show-weapon` renders the pistol again.

### Remote GPU

`remote.sh` syncs the package, this directory and the logged screenshots to a
CUDA host, installs with uv, and runs everything headless over SSH. Set `HOST`
(an SSH destination) and, if needed, `SSH_PORT` in your own environment:

```sh
./remote.sh sync
./remote.sh probe
./remote.sh play "Attack enemies on sight." 60 --output ../../runs/eg2-show-attack
./remote.sh fetch                                       # copies runs/eg2-* back
```

`edit_showcase.py` cuts the showcase from run directories and a probe report;
every number on its cards is read from those files.

## Files

| File | Purpose |
| --- | --- |
| `gemma_doom.py` | Crops, prompts, perception, attack and evade controllers, `EmbeddingGemmaPolicy` |
| `gemma_overlay.py` | Showcase dashboard: window margins, HUD prompt matches, mode cosines |
| `play_gemma.py` | Plays through `doom_bert.play` with the dashboard and video |
| `edit_showcase.py` | Title cards, chapters and concatenation for the showcase MP4 |
| `probe_zero_shot.py` | Scores perception against logged game state |
| `probe_commands.py` | The fused situation → command experiment |
| `prepare_media.py`, `run_tests.py` | General multimodal retrieval checks (macOS media) |
| `remote.sh` | Sync, run and fetch on a remote CUDA machine |
