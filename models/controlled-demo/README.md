---
base_model: answerdotai/ModernBERT-base
library_name: transformers
pipeline_tag: text-classification
license: apache-2.0
---

# Controlled Doom classification demo

Derived from [Answer.AI / LightOn ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base).
The encoder is frozen. The prediction head and seven-button classifier were
trained on synthetic, rule-labeled instruction/state pairs. Those rules are used
offline for labels, never to choose actions during model gameplay.

Modified weights: the prediction head and classifier were trained; the saved
checkpoint uses float16. The upstream Apache 2.0 license is included in LICENSE.

Run with the repository's `ModernBertPolicy` adapter. It reads the checkpoint's
`doom_bert_state_format="categorical-v1"` setting and serializes actual game state
into health, ammo, nearest-enemy direction, and distance categories. The
preprocessor emits no action labels. Inference computes the full transformer,
prediction head, and seven sigmoid outputs every time.

Supported demonstration instructions:

- Attack enemies on sight.
- Shoot enemies whenever you can.
- Shoot at enemies, but retreat when health falls below 40.
- Fight, but stop shooting and retreat below 40 health.
- Do not shoot. Evade the enemies.
- Never fire. Keep away from enemies.

The intended scope is a controlled classification and throughput demonstration.
The test set contains 8 held-out categorical states paired with the same 6 known
instructions used for training. It is not a test of unseen instruction wording.
Test exact action-vector accuracy is 81.25%; per-button accuracy is 97.02%.
The test set has only 2 positive ATTACK examples and no positive MOVE_FORWARD
examples. Validation selected step 150; test data was not used for optimization
or checkpoint selection. All intermediate validation results are retained in
training-report.json.

The 149,014,272 encoder parameters were frozen; 595,975 parameters in the
prediction head and final classifier were trained. Total: 149,610,247.
Training uses cached frozen encoder outputs, then discards the cache. Runtime
has no teacher, lookup table, or embedding cache. Code only masks illegal button
combinations and ammo-dependent firing after the model emits its scores.
