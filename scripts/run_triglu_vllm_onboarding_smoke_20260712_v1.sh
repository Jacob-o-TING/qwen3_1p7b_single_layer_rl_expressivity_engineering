#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="$ROOT/envs/vllm0102_verl061/bin/python"
RUN_ID=triglu_vllm_onboarding_smoke_20260712_v1
OUT="$ROOT/runs/runtime_smokes/$RUN_ID"
LOG="$ROOT/logs/$RUN_ID.log"

cd "$ROOT"
mkdir -p "$OUT" "$(dirname "$LOG")"
export PYTHONPATH="$ROOT/src"
export VLLM_USE_V1=1
RESUME_ARGS=()
if [[ "${TRIGLU_ONBOARDING_RESUME:-0}" == "1" ]]; then
  RESUME_ARGS+=(--resume)
fi

set -o pipefail
{
  echo "RUN_START $(date -Is)"
  echo "RUN_ID $RUN_ID"
  echo "SCREEN qwen_triglu_vllm_smoke_20260712_v1"
  "$PYTHON" -m pip install -e . --no-deps
  "$PYTHON" scripts/run_triglu_vllm_onboarding_smoke.py \
    --output-dir "$OUT" \
    --base-model /root/autodl-tmp/qwen3_single_layer_rl/models/Qwen3-1.7B-Base \
    --config configs/sft/layer10_whole_layer_triglu_side_ffn_sft.yaml \
    --checkpoint runs/sft_ordered_20260711_sft50k_v1/layer10_whole_layer_triglu_side_ffn/checkpoints/step_00003916/trainable_state.pt \
    --validation-jsonl data/numina_math_cot_50k_decontam_v3/val.jsonl \
    --source-root "$ROOT" \
    "${RESUME_ARGS[@]}"
  status=$?
  echo "RUN_EXIT $status"
  echo "RUN_END $(date -Is)"
  exit "$status"
} 2>&1 | tee -a "$LOG"
exit "${PIPESTATUS[0]}"
