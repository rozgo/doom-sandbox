---
license: apache-2.0
base_model: fastino/gliner2.5-small-v1
---

# Experimental GLiNER2.5 Small Doom classification heads

Derived from [Fastino GLiNER2.5 Small](https://huggingface.co/fastino/gliner2.5-small-v1),
revision `f1e4d8fdd6fe328f45dee6aca3e6a07c9db4296e`. The pretrained encoder remains
frozen. These Safetensors files contain only the adapted shared classification
MLP; they require that exact upstream checkpoint and the experiment's schema.

Budgets are 32, 128 and 372 synthetic Doom training examples. Metadata records
the selected rows, validation-based checkpoint selection, hyperparameters,
training metrics and schema. All three use 48 additional validation examples.
The 128-example head was selected for gameplay by validation exact agreement,
with the smaller training budget breaking a tie.

Load via `GLiNERPolicy(..., head=path)` in
[the experiment](../../experiments/gliner2/README.md). This downloads the pinned
base model, loads its complete weights, then replaces only the classification
head. The complete encoder executes for each decision.

These heads demonstrate limited imitation learning, not a general Doom agent.
See [measured results and limitations](../../experiments/gliner2/RESULTS.md).
Upstream attribution: Fastino AI, GLiNER2 / GLiNER2.5. Distributed under Apache-2.0;
see [LICENSE](LICENSE). No affiliation with or endorsement by Fastino is implied.
