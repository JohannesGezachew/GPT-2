#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
CKPT="${1:-runs/pretrained}"
NUM="${2:-500}"
python -m eval.rouge_eval --ckpt "$CKPT" --config configs/pretrained.yaml --num_samples "$NUM"
