#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/envs/vllm0102_verl061/bin/python"
PARENT="$ROOT/runs/eval_protocol/qwen3_1p7b_heplus_prompt_matrix_full164_20260717_v1"
CELL=evalscope_raw_instruction_nochat
mkdir -p "$PARENT/models"
exec > >(tee -a "$PARENT/allmodels_controller.log") 2>&1
exec 9>"$PARENT/allmodels_controller.lock"
flock -n 9 || { echo "HEPLUS_ALLMODELS_CONTROLLER_ALREADY_RUNNING"; exit 9; }

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_V1=1
export TOKENIZERS_PARALLELISM=false

old_root_mode=$(stat -c %a /root)
restore_root_mode() { chmod "$old_root_mode" /root 2>/dev/null || true; }
trap 'rc=$?; restore_root_mode; if (( rc != 0 )); then printf "HEPLUS_ALLMODELS_FAILED rc=%s time=%s\n" "$rc" "$(date -Is)" | tee "$PARENT/ALLMODELS_FAILED"; fi; exit "$rc"' EXIT INT TERM
chmod 701 /root

if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; then
  echo "HEPLUS_ALLMODELS_GPU_BUSY_REFUSING_TO_CONTEND"
  exit 70
fi

write_state() {
  printf 'PHASE=%s\nMODEL=%s\nUPDATED_UNIX=%s\n' "$1" "$2" "$(date +%s)" >"$PARENT/allmodels_state.env.tmp"
  mv "$PARENT/allmodels_state.env.tmp" "$PARENT/allmodels_state.env"
}

run_model() {
  local label=$1 config=$2 out=$3
  write_state PREPARE "$label"
  "$PY" "$ROOT/scripts/run_humanevalplus_prompt_protocol_matrix.py" \
    --root "$ROOT" --config "$config" prepare
  write_state GENERATE_AND_REVIEW "$label"
  local pids=() status=0
  for gpu in 0 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/scripts/run_humanevalplus_prompt_protocol_matrix.py" \
      --root "$ROOT" --config "$config" worker \
      --cell "$CELL" --shard-index "$gpu" --shard-count 6 \
      >"$out/${CELL}.shard${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
  if (( status != 0 )); then
    echo "HEPLUS_MODEL_WORKER_FAILURE model=$label status=$status"
    return "$status"
  fi
  write_state MERGE "$label"
  "$PY" "$ROOT/scripts/run_humanevalplus_prompt_protocol_matrix.py" \
    --root "$ROOT" --config "$config" merge --shard-count 6
}

run_model \
  triglu_step294 \
  "$ROOT/configs/eval/qwen3_1p7b_heplus_nochat_triglu294_full164_20260717_v1.yaml" \
  "$PARENT/models/triglu_step294"
run_model \
  baseline_step196 \
  "$ROOT/configs/eval/qwen3_1p7b_heplus_nochat_baseline196_full164_20260717_v1.yaml" \
  "$PARENT/models/baseline_step196"

write_state COMPLETE all_models
rm -f "$PARENT/ALLMODELS_FAILED"
touch "$PARENT/ALLMODELS_COMPLETE"
trap - EXIT INT TERM
restore_root_mode
echo "HEPLUS_ALLMODELS_COMPLETE time=$(date -Is)"
