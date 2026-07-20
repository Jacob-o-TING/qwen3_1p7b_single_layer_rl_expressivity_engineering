#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
python -m qwen_single_layer_rl.training.dry_run --config configs/smoke_tiny.yaml --out runs/smoke
python -m unittest discover -s tests
