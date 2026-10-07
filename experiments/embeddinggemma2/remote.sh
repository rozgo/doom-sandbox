#!/usr/bin/env bash
# Run the zero-shot EmbeddingGemma 2 Doom player on a remote CUDA machine over SSH.
#
#   ./remote.sh sync                       copy code, install with uv, check the GPU
#   ./remote.sh play "Attack enemies on sight." 60 [extra play_gemma.py args]
#   ./remote.sh probe [extra probe_zero_shot.py args]
#   ./remote.sh fetch                      copy remote runs/eg2-* back here
#
# Set HOST (an SSH destination) and optionally SSH_PORT and REMOTE_DIR (default
# doom-sandbox) in your environment; nothing about the machine is stored here.
set -euo pipefail

HOST=${HOST:?set HOST to an SSH destination}
REMOTE_DIR=${REMOTE_DIR:-doom-sandbox}
SSH="ssh${SSH_PORT:+ -p $SSH_PORT}"
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
EXP=experiments/embeddinggemma2

case "${1:-}" in
sync)
  # Only what the experiment needs: the doom-bert package and this directory.
  # The probe also reads labelled screenshots from runs/, so those are copied too.
  $SSH "$HOST" "mkdir -p $REMOTE_DIR/$EXP $REMOTE_DIR/runs"
  rsync -az -e "$SSH" --exclude .venv --exclude __pycache__ \
    "$ROOT/pyproject.toml" "$ROOT/uv.lock" "$ROOT/README.md" "$ROOT/.python-version" "$ROOT/src" \
    "$HOST:$REMOTE_DIR/"
  rsync -az -e "$SSH" --exclude .venv --exclude __pycache__ "$ROOT/$EXP/" "$HOST:$REMOTE_DIR/$EXP/"
  rsync -az -e "$SSH" --prune-empty-dirs --exclude 'eg2-*/' --exclude 'embeddinggemma2/' \
    --include '*/' --include 'states.jsonl' --include 'frames/*.png' --exclude '*' \
    "$ROOT/runs/" "$HOST:$REMOTE_DIR/runs/"
  $SSH "$HOST" "command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh"
  $SSH "$HOST" "export PATH=\$HOME/.local/bin:\$PATH; cd $REMOTE_DIR/$EXP && uv sync && \
    uv run python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())'"
  ;;
play)
  instruction=${2:-Attack enemies on sight.}
  seconds=${3:-60}
  shift $(($# < 3 ? $# : 3))
  $SSH "$HOST" "export PATH=\$HOME/.local/bin:\$PATH; cd $REMOTE_DIR/$EXP && \
    uv run python play_gemma.py --headless --instruction $(printf '%q' "$instruction") --seconds $seconds $*"
  ;;
probe)
  shift
  $SSH "$HOST" "export PATH=\$HOME/.local/bin:\$PATH; cd $REMOTE_DIR/$EXP && uv run python probe_zero_shot.py $*"
  ;;
fetch)
  rsync -az -e "$SSH" "$HOST:$REMOTE_DIR/runs/eg2-*" "$ROOT/runs/"
  rsync -az -e "$SSH" "$HOST:$REMOTE_DIR/runs/embeddinggemma2/zero-shot-probe.json" "$ROOT/runs/embeddinggemma2/zero-shot-probe-remote.json" || true
  ;;
*)
  sed -n '2,10p' "$0"
  exit 1
  ;;
esac
