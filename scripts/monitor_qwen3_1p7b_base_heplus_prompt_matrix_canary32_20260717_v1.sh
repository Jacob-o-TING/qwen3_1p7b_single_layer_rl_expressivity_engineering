#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/runs/eval_protocol/qwen3_1p7b_base_heplus_prompt_matrix_canary32_20260717_v1"
echo "=== HumanEval+ prompt protocol matrix: untuned base, 32 tasks x 3 cells ==="
if [[ -f "$OUT/state.env" ]]; then
  # shellcheck disable=SC1090
  source "$OUT/state.env"
  echo "phase: ${PHASE:-unknown}"
else
  echo "phase: not started"
fi

if [[ -s "$OUT/summary.txt" ]]; then
  cat "$OUT/summary.txt"
else
  for cell in evalscope_chat_instruction_control evalscope_raw_instruction_nochat canonical_completion_nochat; do
    reviewed=0
    complete=0
    for shard in 0 1; do
      worker="$OUT/workers/${cell}.shard$(printf '%02d' "$shard")-of-02"
      if [[ -s "$worker/progress.json" ]]; then
        value=$(grep -o '"reviewed": [0-9]*' "$worker/progress.json" | awk '{print $2}' || true)
        reviewed=$((reviewed + ${value:-0}))
      fi
      [[ -f "$worker/WORKER_COMPLETE" ]] && complete=$((complete + 1))
    done
    printf '%-40s reviewed=%2d/32 shards=%d/2\n' "$cell" "$reviewed" "$complete"
  done
fi

[[ ! -f "$OUT/MATRIX_FAILED" ]] || { echo "--- failure ---"; cat "$OUT/MATRIX_FAILED"; }
echo "--- GPUs ---"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
echo "disk free: $(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')G"
