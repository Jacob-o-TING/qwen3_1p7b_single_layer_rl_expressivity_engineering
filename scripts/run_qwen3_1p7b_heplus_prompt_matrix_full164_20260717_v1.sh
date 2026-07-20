#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/envs/vllm0102_verl061/bin/python"
CONFIG="$ROOT/configs/eval/qwen3_1p7b_heplus_prompt_matrix_full164_20260717_v1.yaml"
OUT="$ROOT/runs/eval_protocol/qwen3_1p7b_heplus_prompt_matrix_full164_20260717_v1"
CELL=evalscope_raw_instruction_nochat
mkdir -p "$OUT"
exec > >(tee -a "$OUT/controller.log") 2>&1
exec 9>"$OUT/controller.lock"
flock -n 9 || { echo "FULL164_CONTROLLER_ALREADY_RUNNING"; exit 9; }

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_V1=1
export TOKENIZERS_PARALLELISM=false

old_root_mode=$(stat -c %a /root)
restore_root_mode() { chmod "$old_root_mode" /root 2>/dev/null || true; }
trap 'rc=$?; restore_root_mode; if (( rc != 0 )); then printf "FULL164_FAILED rc=%s time=%s\n" "$rc" "$(date -Is)" | tee "$OUT/FULL164_FAILED"; fi; exit "$rc"' EXIT INT TERM
chmod 701 /root

if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; then
  echo "FULL164_GPU_BUSY_REFUSING_TO_CONTEND"
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
pids=()
for gpu in 0 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/scripts/run_humanevalplus_prompt_protocol_matrix.py" \
    --root "$ROOT" --config "$CONFIG" worker \
    --cell "$CELL" --shard-index "$gpu" --shard-count 6 \
    >"$OUT/${CELL}.shard${gpu}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
if (( status != 0 )); then
  echo "FULL164_WORKER_FAILURE status=$status"
  exit "$status"
fi

write_state MERGE
"$PY" "$ROOT/scripts/run_humanevalplus_prompt_protocol_matrix.py" \
  --root "$ROOT" --config "$CONFIG" merge --shard-count 6
write_state COMPLETE
rm -f "$OUT/FULL164_FAILED"
trap - EXIT INT TERM
restore_root_mode
echo "FULL164_COMPLETE run_id=qwen3_1p7b_heplus_prompt_matrix_full164_20260717_v1 time=$(date -Is)"
