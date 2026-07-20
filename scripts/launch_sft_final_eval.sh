#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:?Usage: launch_sft_final_eval.sh CONFIG CHECKPOINT_DIR WORK_DIR [--limit N]}"
CHECKPOINT_DIR="${2:?Missing checkpoint directory}"
WORK_DIR="${3:?Missing evaluation work directory}"
shift 3
PYTHON_BIN="${EVAL_PYTHON_BIN:-python}"
TRAINING_PYTHON_BIN="${TRAINING_PYTHON_BIN:-$ROOT/envs/vllm0102_verl061/bin/python}"
MODEL_PATH="${MODEL_PATH:-$ROOT/models/Qwen3-1.7B-Base}"

TRAINING_SITE_PACKAGES="$($TRAINING_PYTHON_BIN -c 'import site; print(site.getsitepackages()[0])')"
export PYTHONPATH="$ROOT/src:$TRAINING_SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

echo "SFT_FINAL_EVAL_START $(date -Is)"
echo "CONFIG=$CONFIG"
echo "CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "WORK_DIR=$WORK_DIR"
COMMON_ARGS=(
  --config "$CONFIG"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --model-path "$MODEL_PATH"
)
if [[ "$(basename "$CONFIG")" == "layer10_whole_layer_baseline_sft.yaml" ]]; then
  COMMON_ARGS+=(--amc-first)
  echo "SFT_FINAL_EVAL_ORDER amc_first reason=whole_layer_baseline"
else
  echo "SFT_FINAL_EVAL_ORDER main_first reason=default"
fi

limited=0
for argument in "$@"; do
  if [[ "$argument" == "--limit" || "$argument" == --limit=* ]]; then
    limited=1
    break
  fi
done

if [[ "$limited" == "0" && "${SFT_FINAL_EVAL_PREFLIGHT:-1}" == "1" ]]; then
  PREFLIGHT_DIR="${WORK_DIR}_batched_preflight"
  rm -rf "$PREFLIGHT_DIR"
  echo "SFT_FINAL_EVAL_PREFLIGHT_START $(date -Is) batch_size=${SFT_EVAL_BATCH_SIZE:-8}"
  "$PYTHON_BIN" -m qwen_single_layer_rl.eval.run_evalscope \
    "${COMMON_ARGS[@]}" \
    --work-dir "$PREFLIGHT_DIR" \
    --limit 1 \
    --amc-repeats 2 \
    --max-tokens 64 \
    --amc-temperature 1.0 \
    --amc-top-p 1.0 \
    --eval-batch-size "${SFT_EVAL_BATCH_SIZE:-8}"
  echo "SFT_FINAL_EVAL_PREFLIGHT_END $(date -Is)"
fi

"$PYTHON_BIN" -m qwen_single_layer_rl.eval.run_evalscope \
  "${COMMON_ARGS[@]}" \
  --work-dir "$WORK_DIR" \
  "$@"
echo "SFT_FINAL_EVAL_END $(date -Is)"
