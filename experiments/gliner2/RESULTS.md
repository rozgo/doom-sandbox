# GLiNER2.5 versus ModernBERT in Doom

GLiNER2.5 Base can drive basic attack/evade behavior with **zero Doom-specific
training**. It is a credible subject for a zero-shot demonstration, but this
experiment does **not** establish an upgrade over the trained ModernBERT policy.
ModernBERT was faster and more reliable on its trained state/action task.

GLiNER2.5 Small needed training to produce useful evasive movement. Adapting only
its classification head on 128 training examples substantially improved it,
including better agreement than ModernBERT on our small new-phrasing test. Both
adapted GLiNER heads still struggled with conditional firing instructions.

## Measured speed

Apple M3 Max, Python 3.12, PyTorch 2.14, MPS float16, batch one. Each pipeline
scored 516 evaluation pairs and completed six 20-game-second rollouts across
three instructions and two seeds: 2,100 live decisions per pipeline.

| Pipeline | Doom training examples | Mean classification, evaluation set | Complete headless game loop |
| --- | ---: | ---: | ---: |
| ModernBERT | 372 | 12.0 ms | 53.5 decisions/s |
| GLiNER2.5 Small | 0 | 15.8 ms | 46.1 decisions/s |
| GLiNER2.5 Base | 0 | 24.4 ms | 31.3 decisions/s |
| GLiNER2.5 Small, adapted head | 128 | 14.6 ms | 42.8 decisions/s |
| GLiNER2.5 Base, adapted head | 32 | 23.5 ms | 27.5 decisions/s |

Classification includes serialization, tokenization, the complete encoder,
scores transferred to CPU, and legal action decoding. It excludes model loading
and 10 warmup calls. Headless throughput includes the game, JSON logs, and
periodic screenshots. Throughput is total decisions divided by total wall time,
not an average of individual rates. Game-loop classification latency is also
preserved separately in the reports because gameplay inputs differ from the
evaluation set. Individual runs varied; these are measurements from this run,
not latency guarantees.

**The previous 44 decisions/s ModernBERT video also ran the dashboard and video
encoder.** Its number is not the comparison baseline here. No video was created
for this experiment. Two game tics per decision require 17.5 decisions/s to
advance Doom at native game speed; all pipeline averages exceeded that rate,
although one adapted-Base cautious run fell below it.

GLiNER inputs are 227–253 tokens, including action descriptions, versus
ModernBERT's 25–36 tokens. This compares complete usable pipelines on the same
observations, not equal-length encoder kernels. GLiNER's DeBERTa backbone uses
eager attention with the supported Transformers 4 environment; ModernBERT uses
SDPA with Transformers 5.

CPU float32, four threads, averaged 36.6 ms for Small and 74.9 ms for Base on
55 probe classifications each. CPU/MPS scores differed by at most 0.00062 and
0.00223 respectively, and produced identical initial behavior-check outcomes.
The observed weaknesses therefore also occur without MPS float16.

## What it actually did

On an identical healthy state with ammo and an aligned enemy, zero-shot Base's
ATTACK score changed from **0.976 for “Attack enemies on sight.” to 0.022 for
“Do not shoot. Evade the enemies.”** Its evasion runs consumed no ammo. It also
reacted to enemy bearing, though right-turn selection was inconsistent.

Zero-shot Small changed firing scores with the instruction, but its movement
scores stayed too low: it stood still for every decision in both evasion runs.
No-fire compliance alone would have hidden this failure.

The following totals combine two seeds, each run for 20 game seconds per
instruction. Entries are **kills / deaths**, not a comprehensive skill score.

| Pipeline | Attack | Evade | Cautious |
| --- | ---: | ---: | ---: |
| ModernBERT | 18 / 0 | 0 / 0 | 9 / 1 |
| Small, zero-shot | 9 / 4 | 0 / 4 | 0 / 4 |
| Base, zero-shot | 8 / 3 | 0 / 1 | 21 / 1 |
| Small, 128 examples | 14 / 1 | 0 / 0 | 0 / 0 |
| Base, 32 examples | 19 / 1 | 0 / 0 | 0 / 0 |

Both adapted heads produced movement and survived both evasion runs without
firing. Both also **never fired in their cautious runs**, even while healthy.
Zero-shot Base scored more cautious kills than ModernBERT in these short runs,
but that does not demonstrate reliable conditional instruction following.

## Training and generalization

The original dataset's whole-state splits were retained. The following metric
is **exact agreement of the decoded seven-button vector with the scripted
teacher**, not correctness under every reasonable gameplay strategy.

| Pipeline | Original held-out states, 48 pairs | New instruction phrasings, 48 pairs |
| --- | ---: | ---: |
| ModernBERT | 81.25% | 56.25% |
| Small, zero-shot | 2.08% | 2.08% |
| Base, zero-shot | 4.17% | 4.17% |
| Small, 128 examples | 75.00% | 70.83% |
| Base, 32 examples | 70.83% | 41.67% |

The held-out set has only eight distinct state categories, two positive ATTACK
examples and no positive MOVE_FORWARD examples. Low teacher agreement is not
equivalent to inability to play, as Base's zero-shot gameplay demonstrates.
Conversely, Small's new-phrasing result is promising evidence for this narrow
test, not proof of superior general instruction understanding.

Both GLiNER sizes were tried with 32, 128 and 372 training examples, freezing
the encoder and adapting only the existing classification MLP. Each budget
used 600 steps and the same hyperparameters. The smallest budget attaining
the best validation exact agreement was selected for full inference/gameplay:
128 for Small and 32 for Base. All budgets also used the same 48 validation
examples for checkpoint selection. Test results did not select weights or
the gameplay budget. These training settings were not exhaustively tuned;
several selected checkpoints were at the final step.

Head optimization took 0.75–1.19 seconds per Small budget and 1.79–2.95 seconds
per Base budget. Shared feature extraction took another 8.46 and 13.89 seconds
respectively; loading, feature extraction, all three budgets and saves took
13.46 and 25.07 seconds. These timings have a different scope from the earlier
3.6-second ModernBERT training run and should not be presented as a speedup.

## Explicit conditional-instruction follow-up

“Shoot at enemies, but retreat when health falls below 40” does not explicitly
forbid firing while retreating. To remove that ambiguity, a follow-up tested
“Fight, but stop shooting and retreat below 40 health” and one new, explicit
“fire only when health is at least 40” phrasing, at health 20, 39, 40 and 80.
No model, schema or threshold was changed for this diagnostic.

Correct raw ATTACK decisions out of eight: ModernBERT **6**, zero-shot Small
**4**, zero-shot Base **4**, adapted Small **4**, adapted Base **3**.
ModernBERT passed all four cases for the known phrasing, but missed firing
above the threshold under the new phrasing. GLiNER results also changed
substantially with wording. Neither approach establishes arbitrary instruction
following. Existing health categories already use the 40-health boundary, so
these probes are not proof of general numerical reasoning either.

## Evidence and next decision

The useful video opportunity is **GLiNER2.5 Base selecting game actions without
Doom-specific training**, showing both its responsiveness and its limitations.
The current evidence does not support a “faster/better than ModernBERT” claim.
Any video headline should use throughput measured with its actual recording
and dashboard enabled.

- [Aggregate results](../../reports/gliner2-experiment/summary.json)
- [All detailed reports](../../reports/gliner2-experiment/)
- [Raw predictions and game logs, Git LFS archive](../../reports/gliner2-experiment/raw-evidence.tar.gz)
- [Evidence SHA-256 manifest](../../reports/gliner2-experiment/evidence-manifest.json)
- [Small adapted heads](../../models/gliner2.5-small-heads/)
- [Base adapted heads](../../models/gliner2.5-base-heads/)
- [Setup, methodology and reproduction](README.md)

All 10,500 live decisions used one full encoder forward each. Re-decoding the
recorded scores reproduced every executed action. The scripted teacher was
used only to supply offline training/evaluation targets. No instruction-specific
override, runtime teacher, embedding cache or action cache was used.
