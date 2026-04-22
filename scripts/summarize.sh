#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
CKPT="${1:-runs/pretrained}"
shift || true
python -m eval.generate --ckpt "$CKPT" --config configs/pretrained.yaml "$@"
