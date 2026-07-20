#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/layer10_whole_layer_shs.yaml}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
RAY_BIN="${RAY_BIN:-$(dirname "$PYTHON_BIN")/ray}"
VERL_ROOT="${VERL_ROOT:-/root/autodl-tmp/verl-v0.6.1-qwenpatch}"
MODEL_PATH="${MODEL_PATH:-$ROOT/models/Qwen3-1.7B-Base}"
CANON_DATA_DIR="${CANON_DATA_DIR:-$ROOT/data/numina_math_cot_50k_decontam_v3}"
VERL_DATA_DIR="${VERL_DATA_DIR:-$ROOT/data/numina_math_cot_50k_decontam_v3_verl}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$ROOT/local_manifests/verl_commands}"

export PYTHONPATH="$ROOT/src:$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHON_BIN
cd "$ROOT"

cleanup_ray() {
  if [ -x "$RAY_BIN" ]; then
    "$RAY_BIN" stop --force >/dev/null 2>&1 || true
  fi
}

mkdir -p "$VERL_DATA_DIR" "$RUN_ROOT" "$MANIFEST_ROOT"

"$PYTHON_BIN" -m qwen_single_layer_rl.data.verl_format \
  --train "$CANON_DATA_DIR/train.parquet" \
  --val "$CANON_DATA_DIR/val.parquet" \
  --out-dir "$VERL_DATA_DIR"

PLAN_DIR="$RUN_ROOT/plan_$(basename "${CONFIG%.yaml}")_$(date +%Y%m%d_%H%M%S)"
"$PYTHON_BIN" -m qwen_single_layer_rl.training.dry_run --config "$CONFIG" --out "$PLAN_DIR"

MANIFEST="$MANIFEST_ROOT/$(basename "${CONFIG%.yaml}")_command.json"
"$PYTHON_BIN" -m qwen_single_layer_rl.training.verl_command \
  --config "$CONFIG" \
  --project-root "$ROOT" \
  --verl-root "$VERL_ROOT" \
  --model-path "$MODEL_PATH" \
  --data-dir "$VERL_DATA_DIR" \
  --run-root "$RUN_ROOT" \
  --manifest-out "$MANIFEST"

echo "Launching veRL from command manifest: $MANIFEST"
SHELL_CMD="$(
  "$PYTHON_BIN" -m qwen_single_layer_rl.training.verl_command \
    --config "$CONFIG" \
    --project-root "$ROOT" \
    --verl-root "$VERL_ROOT" \
    --model-path "$MODEL_PATH" \
    --data-dir "$VERL_DATA_DIR" \
    --run-root "$RUN_ROOT" \
    --print-shell
)"
cleanup_ray
set +e
eval "$SHELL_CMD"
STATUS=$?
set -e
cleanup_ray
exit "$STATUS"
