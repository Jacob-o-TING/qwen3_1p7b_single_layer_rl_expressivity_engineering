#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/runs/eval_protocol/qwen3_1p7b_heplus_prompt_matrix_full164_20260717_v1"
CELL=evalscope_raw_instruction_nochat
echo "=== HumanEval+ full-164: untuned base, raw EvalScope instruction, no chat template ==="
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
  reviewed=0
  complete=0
  for shard in 0 1 2 3 4 5; do
    worker="$OUT/workers/${CELL}.shard$(printf '%02d' "$shard")-of-06"
    if [[ -s "$worker/progress.json" ]]; then
      value=$(grep -o '"reviewed": [0-9]*' "$worker/progress.json" | awk '{print $2}' || true)
      reviewed=$((reviewed + ${value:-0}))
    fi
    [[ -f "$worker/WORKER_COMPLETE" ]] && complete=$((complete + 1))
  done
  printf 'reviewed=%3d/164 shards=%d/6\n' "$reviewed" "$complete"
fi

[[ ! -f "$OUT/FULL164_FAILED" ]] || { echo "--- failure ---"; cat "$OUT/FULL164_FAILED"; }
echo "--- GPUs ---"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
echo "disk free: $(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')G"
