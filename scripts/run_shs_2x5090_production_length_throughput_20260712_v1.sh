#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="$ROOT/envs/vllm0102_verl061/bin/python"
CONFIG="$ROOT/configs/runtime/shs_2x5090_production_length_throughput_20260712_v1.yaml"
RUN_ID="shs_2x5090_production_length_throughput_20260712_v1"
OUT="$ROOT/runs/runtime_smokes/$RUN_ID"
LOG="$ROOT/logs/$RUN_ID.log"
LOCK="$ROOT/runs/runtime_smokes/$RUN_ID.lock"

mkdir -p "$OUT" "$ROOT/logs" "$(dirname "$LOCK")"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "Refusing concurrent launch: $LOCK" >&2
  exit 31
fi

export PYTHONPATH="$ROOT/src"
export VLLM_USE_V1=1
export SHS_INFERENCE_MUL_BACKEND=reference

START_EPOCH="$(date +%s)"
set +e
{
  echo "RUN_START $(date -Is)"
  echo "RUN_ID $RUN_ID"
  echo "CONFIG $CONFIG"
  echo "BACKEND reference PyTorch/cuBLAS inside vLLM"
  echo "TOPOLOGY 2 independent TP=1 replicas on CUDA physical GPUs 0 and 1"
  nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu --format=csv,noheader
  "$PYTHON_BIN" "$ROOT/scripts/run_shs_2x5090_production_length_throughput.py" --config "$CONFIG"
} 2>&1 | tee -a "$LOG"
STATUS=${PIPESTATUS[0]}
set -e
echo "RUN_END $(date -Is) status=$STATUS wall_seconds=$(($(date +%s) - START_EPOCH))" | tee -a "$LOG"
exit "$STATUS"
