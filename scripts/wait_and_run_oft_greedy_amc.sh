#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING_PYTHON_BIN="${TRAINING_PYTHON_BIN:-$ROOT/envs/vllm0102_verl061/bin/python}"
RUN_ROOT="${SFT_OUTPUT_ROOT:?SFT_OUTPUT_ROOT is required}"
EVAL_ROOT="${SFT_EVAL_ROOT:-$RUN_ROOT/evaluations}"
POLL_SECONDS="${SFT_OFT_GREEDY_WAIT_SECONDS:-60}"
RUN_DIR="$RUN_ROOT/layer10_whole_layer_oft"
EVAL_DIR="$EVAL_ROOT/layer10_whole_layer_oft"
CONFIG="$ROOT/configs/sft/layer10_whole_layer_oft_sft.yaml"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
echo "SFT_OFT_GREEDY_WAITER_START $(date -Is) poll_seconds=$POLL_SECONDS"

while true; do
  if [[ -f "$RUN_DIR/train_result.json" ]]; then
    if checkpoint_dir="$(
      "$TRAINING_PYTHON_BIN" -m qwen_single_layer_rl.sft.handoff \
        resolve-checkpoint --run-dir "$RUN_DIR" 2>/dev/null
    )"; then
      if "$TRAINING_PYTHON_BIN" -m qwen_single_layer_rl.sft.handoff \
        eval-status --eval-dir "$EVAL_DIR" --checkpoint-dir "$checkpoint_dir" \
        >/dev/null 2>&1; then
        break
      fi
    fi
  fi
  sleep "$POLL_SECONDS"
done

echo "SFT_OFT_GREEDY_PRIMARY_RECEIPT_READY $(date -Is) checkpoint=$checkpoint_dir"
bash scripts/launch_greedy_amc_controls_before_triglu.sh \
  post-variant \
  layer10_whole_layer_oft \
  amc_greedy_modal_path_oft_sft50k_v1 \
  "$RUN_DIR" \
  "$CONFIG" \
  checkpoint \
  greedy
echo "SFT_OFT_GREEDY_WAITER_END $(date -Is)"
