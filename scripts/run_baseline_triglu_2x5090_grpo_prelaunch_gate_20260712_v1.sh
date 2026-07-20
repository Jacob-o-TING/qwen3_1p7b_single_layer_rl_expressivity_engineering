#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="$ROOT/envs/vllm0102_verl061/bin/python"
CONFIG="$ROOT/configs/runtime/baseline_triglu_2x5090_grpo_prelaunch_gate_20260712_v1.yaml"
RUN_ID="baseline_triglu_2x5090_grpo_prelaunch_gate_20260712_v1"
OUT="$ROOT/runs/runtime_smokes/$RUN_ID"
LOG="$OUT/run.log"

mkdir -p "$OUT"
if [[ -e "$OUT/manifest.json" ]]; then
  echo "Refusing to overwrite completed manifest: $OUT/manifest.json" >&2
  exit 17
fi

export PYTHONPATH="$ROOT/src"
export PYTHONHASHSEED=20260707
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

started_epoch="$(date +%s)"
set +e
{
  echo "RUN_START $(date -Is)"
  echo "RUN_ID $RUN_ID"
  echo "SCOPE baseline+TriGLU 2x5090 prelaunch only; no 20/50/98 batch authorization"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
  "$PYTHON_BIN" "$ROOT/scripts/run_baseline_triglu_2x5090_grpo_prelaunch_gate.py" \
    --config "$CONFIG" \
    --source-root "$ROOT" \
    --output-dir "$OUT"
} 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
set -e

echo "RUN_END $(date -Is) status=$status wall_seconds=$(($(date +%s) - started_epoch))" | tee -a "$LOG"
exit "$status"
