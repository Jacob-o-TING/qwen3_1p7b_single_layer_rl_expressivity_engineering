#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${SFT_RUN_ID:-sft_ordered_20260711_sft50k_v1}"
SESSION="${SFT_SESSION:-qwen_sft_ordered_20260711_sft50k_v1}"
RUN_ROOT="${SFT_OUTPUT_ROOT:-$ROOT/runs/$RUN_ID}"
EVAL_ROOT="${SFT_EVAL_ROOT:-$RUN_ROOT/evaluations}"
LOG="${SFT_LOG:-$ROOT/logs/$SESSION.log}"
TRAINING_PYTHON_BIN="${TRAINING_PYTHON_BIN:-$ROOT/envs/vllm0102_verl061/bin/python}"
EVAL_PYTHON_BIN="${EVAL_PYTHON_BIN:-$ROOT/envs/evalscope181/bin/python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

if screen -ls 2>/dev/null | grep -q "[.]$SESSION"; then
  echo "Session already active: $SESSION" >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG")"
echo "SFT_DURABLE_RESUME_LAUNCH $(date -Is) session=$SESSION run_root=$RUN_ROOT nproc_per_node=$NPROC_PER_NODE" | tee -a "$LOG"

screen -dmS "$SESSION" bash -lc "
  set -o pipefail
  cd '$ROOT'
  export PYTHON_BIN='$EVAL_PYTHON_BIN'
  export EVAL_PYTHON_BIN='$EVAL_PYTHON_BIN'
  export TRAINING_PYTHON_BIN='$TRAINING_PYTHON_BIN'
  export NPROC_PER_NODE='$NPROC_PER_NODE'
  export SFT_SESSION='$SESSION'
  export SFT_OUTPUT_ROOT='$RUN_ROOT'
  export SFT_EVAL_ROOT='$EVAL_ROOT'
  bash scripts/launch_sft_ordered_variants.sh \
    --micro-batch-size 1 \
    --gradient-accumulation-steps 8 \
    2>&1 | tee -a '$LOG'
  exit \${PIPESTATUS[0]}
"

echo "Resume session launched: $SESSION"
echo "Monitor: cd $ROOT && bash scripts/monitor_sft.sh"
