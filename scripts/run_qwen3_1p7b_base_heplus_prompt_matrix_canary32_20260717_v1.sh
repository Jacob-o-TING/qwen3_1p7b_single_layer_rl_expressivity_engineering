#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/envs/vllm0102_verl061/bin/python"
CONFIG="$ROOT/configs/eval/qwen3_1p7b_base_heplus_prompt_matrix_canary32_20260717_v1.yaml"
OUT="$ROOT/runs/eval_protocol/qwen3_1p7b_base_heplus_prompt_matrix_canary32_20260717_v1"
LOG="$OUT/controller.log"
mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
exec 9>"$OUT/controller.lock"
flock -n 9 || { echo "MATRIX_CONTROLLER_ALREADY_RUNNING"; exit 9; }

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_V1=1
export TOKENIZERS_PARALLELISM=false

old_root_mode=$(stat -c %a /root)
restore_root_mode() { chmod "$old_root_mode" /root 2>/dev/null || true; }
trap 'rc=$?; restore_root_mode; if (( rc != 0 )); then printf "MATRIX_FAILED rc=%s time=%s\n" "$rc" "$(date -Is)" | tee "$OUT/MATRIX_FAILED"; fi; exit "$rc"' EXIT INT TERM
chmod 701 /root

if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; then
  echo "MATRIX_GPU_BUSY_REFUSING_TO_CONTEND"
  exit 70
fi

write_state() {
  printf 'PHASE=%s\nUPDATED_UNIX=%s\n' "$1" "$(date +%s)" >"$OUT/state.env.tmp"
  mv "$OUT/state.env.tmp" "$OUT/state.env"
}

write_state PREPARE
"$PY" "$ROOT/scripts/run_humanevalplus_prompt_protocol_matrix.py" \
  --root "$ROOT" --config "$CONFIG" prepare

write_state GENERATE_AND_REVIEW
cells=(
  evalscope_chat_instruction_control
  evalscope_raw_instruction_nochat
  canonical_completion_nochat
)
pids=()
gpu=0
for cell in "${cells[@]}"; do
  for shard in 0 1; do
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/scripts/run_humanevalplus_prompt_protocol_matrix.py" \
      --root "$ROOT" --config "$CONFIG" worker \
      --cell "$cell" --shard-index "$shard" --shard-count 2 \
      >"$OUT/${cell}.shard${shard}.log" 2>&1 &
    pids+=("$!")
    gpu=$((gpu + 1))
  done
done

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
if (( status != 0 )); then
  echo "MATRIX_WORKER_FAILURE status=$status"
  exit "$status"
fi

write_state MERGE
"$PY" "$ROOT/scripts/run_humanevalplus_prompt_protocol_matrix.py" \
  --root "$ROOT" --config "$CONFIG" merge --shard-count 2
write_state COMPLETE
rm -f "$OUT/MATRIX_FAILED"
trap - EXIT INT TERM
restore_root_mode
echo "MATRIX_COMPLETE run_id=qwen3_1p7b_base_heplus_prompt_matrix_canary32_20260717_v1 time=$(date -Is)"
