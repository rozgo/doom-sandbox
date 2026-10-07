#!/usr/bin/env bash
# Second GLiNER2.5 attempt, run on one CUDA GPU from experiments/gliner2.
#   ./run_v2.sh data | train | eval | videos | edit
# Outputs go to ../../runs/gliner2-v2 (ignored); reports worth keeping are copied
# to reports/gliner2-v2/ by hand after review.
set -euo pipefail
cd "$(dirname "$0")"
OUT=../../runs/gliner2-v2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"

case "${1:-}" in
data)
  uv run python teacher_data.py --output ../../data/gliner2-teacher-v2.jsonl
  ;;
train)
  # Same effective batch of 32 everywhere; Base splits it in two to fit beside other GPU work.
  uv run python finetune.py --model small --scope full --output "$OUT/small-full"
  uv run python finetune.py --model base --scope full --batch-size 32 --grad-accum 2 --output "$OUT/base-full"
  uv run python finetune.py --model small --scope head --output "$OUT/small-head"
  uv run python finetune.py --model base --scope head --output "$OUT/base-head"
  ;;
eval)
  for model in small base; do
    uv run python compare.py --model "$model" --device cuda --output "$OUT/eval-$model-zero-shot"
    uv run python check_conditionals.py --model "$model" --device cuda --output "$OUT/conditionals-$model-zero-shot.json"
  done
  for run in small-full base-full small-head base-head; do
    model=${run%%-*}
    uv run python compare.py --model "$model" --device cuda --weights "$OUT/$run" --output "$OUT/eval-$run"
    uv run python check_conditionals.py --model "$model" --device cuda --weights "$OUT/$run" \
      --output "$OUT/conditionals-$run.json"
  done
  (cd ../.. && uv run --extra model python experiments/gliner2/compare.py --model modernbert --device cuda \
    --output runs/gliner2-v2/eval-modernbert)
  (cd ../.. && uv run --extra model python experiments/gliner2/check_conditionals.py --model modernbert --device cuda \
    --output runs/gliner2-v2/conditionals-modernbert.json)
  ;;
videos)
  model=${2:?model, e.g. base}; run=${3:?weights run, e.g. base-full}
  uv run python play_gliner.py --model "$model" --weights "$OUT/$run" --instruction "Attack enemies on sight." \
    --output "../../runs/gliner2-show-attack"
  uv run python play_gliner.py --model "$model" --weights "$OUT/$run" --instruction "Do not shoot. Evade the enemies." \
    --output "../../runs/gliner2-show-evade"
  uv run python play_gliner.py --model "$model" --weights "$OUT/$run" \
    --instruction "Shoot at enemies, but retreat when health falls below 40." --output "../../runs/gliner2-show-cautious"
  ;;
edit)
  run=${2:?weights run, e.g. base-full}
  uv run python edit_showcase.py --runs ../../runs/gliner2-show-attack ../../runs/gliner2-show-evade \
    ../../runs/gliner2-show-cautious --report "$OUT/eval-$run/report.json" \
    --conditionals "$OUT/conditionals-$run.json" --output ../../runs/gliner2-showcase/gliner2-plays-doom.mp4
  ;;
*)
  sed -n '2,5p' "$0"
  exit 1
  ;;
esac
