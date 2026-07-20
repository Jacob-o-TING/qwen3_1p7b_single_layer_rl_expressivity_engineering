#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SESSION="${SFT_SESSION:-qwen_sft_ordered_$STAMP}"
RUN_ROOT="${SFT_OUTPUT_ROOT:-$ROOT/runs/sft_ordered_$STAMP}"
LOG="$ROOT/logs/sft_ordered_$STAMP.log"
DEFAULT_TRAINING_PYTHON="$ROOT/envs/vllm0102_verl061/bin/python"
DEFAULT_EVAL_PYTHON="$ROOT/envs/evalscope181/bin/python"

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  NPROC_PER_NODE="$(nvidia-smi -L | grep -c '^GPU ')"
fi
if [[ "$NPROC_PER_NODE" -le 0 ]]; then
  echo "No visible GPUs were detected" >&2
  exit 1
fi
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_TRAINING_PYTHON}"
TRAINING_PYTHON_BIN="${TRAINING_PYTHON_BIN:-$PYTHON_BIN}"
EVAL_PYTHON_BIN="${EVAL_PYTHON_BIN:-$DEFAULT_EVAL_PYTHON}"
TARGET_EFFECTIVE_PACKED_BATCH_SIZE="${TARGET_EFFECTIVE_PACKED_BATCH_SIZE:-8}"
MICRO_BATCH_SIZE="${SFT_MICRO_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="$(
  PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$TRAINING_PYTHON_BIN" \
    -m qwen_single_layer_rl.sft.handoff accumulation \
    --target "$TARGET_EFFECTIVE_PACKED_BATCH_SIZE" \
    --world-size "$NPROC_PER_NODE" \
    --micro-batch-size "$MICRO_BATCH_SIZE"
)"
EVAL_ROOT="${SFT_EVAL_ROOT:-$RUN_ROOT/evaluations}"

mkdir -p "$ROOT/logs" "$RUN_ROOT" "$EVAL_ROOT"
printf '%s\n' "$SESSION" > "$ROOT/logs/current_sft.screen"
printf '%s\n' "$LOG" > "$ROOT/logs/current_sft.logpath"
printf '%s\n' "$RUN_ROOT" > "$ROOT/logs/current_sft.runroot"
printf '%s\n' "$EVAL_ROOT" > "$ROOT/logs/current_sft.evalroot"

export SFT_OUTPUT_ROOT="$RUN_ROOT"
export SFT_EVAL_ROOT="$EVAL_ROOT"
export PYTHON_BIN TRAINING_PYTHON_BIN EVAL_PYTHON_BIN NPROC_PER_NODE

screen -dmS "$SESSION" bash -lc "
  set -o pipefail
  cd '$ROOT'
  echo 'SFT_ORDERED_TOPOLOGY world_size=$NPROC_PER_NODE micro_batch=$MICRO_BATCH_SIZE accumulation=$GRADIENT_ACCUMULATION_STEPS effective_batch=$TARGET_EFFECTIVE_PACKED_BATCH_SIZE' | tee '$LOG'
  bash scripts/launch_sft_ordered_variants.sh \
    --micro-batch-size '$MICRO_BATCH_SIZE' \
    --gradient-accumulation-steps '$GRADIENT_ACCUMULATION_STEPS' \
    2>&1 | tee -a '$LOG'
"

echo "SFT_SESSION=$SESSION"
echo "SFT_LOG=$LOG"
echo "SFT_RUN_ROOT=$RUN_ROOT"
echo "SFT_EVAL_ROOT=$EVAL_ROOT"
echo "SFT_TOPOLOGY=world_size=$NPROC_PER_NODE micro_batch=$MICRO_BATCH_SIZE accumulation=$GRADIENT_ACCUMULATION_STEPS effective_batch=$TARGET_EFFECTIVE_PACKED_BATCH_SIZE"
echo "MONITOR=bash $ROOT/scripts/monitor_sft.sh"
