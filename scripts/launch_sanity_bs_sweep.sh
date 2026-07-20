#!/usr/bin/env bash
set -euo pipefail

BASE_CONFIG="${1:-configs/layer10_whole_layer_shs.yaml}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
RAY_BIN="${RAY_BIN:-$(dirname "$PYTHON_BIN")/ray}"
VERL_ROOT="${VERL_ROOT:-/root/autodl-tmp/verl-v0.6.1-qwenpatch}"
MODEL_PATH="${MODEL_PATH:-$ROOT/models/Qwen3-1.7B-Base}"
CANON_DATA_DIR="${CANON_DATA_DIR:-$ROOT/data/numina_math_cot_50k_decontam_v3}"
VERL_DATA_DIR="${VERL_DATA_DIR:-$ROOT/data/numina_math_cot_50k_decontam_v3_verl}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$ROOT/local_manifests/verl_commands}"
BATCH_SIZES="${BATCH_SIZES:-8 16 32 64 128}"
CASE_TIMEOUT_SECONDS="${CASE_TIMEOUT_SECONDS:-3600}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
INFERENCE_BACKEND="${INFERENCE_BACKEND:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-}"
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE="${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-1}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-64}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.85}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-131072}"

export PYTHONPATH="$ROOT/src:$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHON_BIN
cd "$ROOT"

BASE_CONFIG_ABS="$("$PYTHON_BIN" - <<PY
from pathlib import Path
root = Path("$ROOT").resolve()
base = Path("$BASE_CONFIG")
if not base.is_absolute():
    base = root / base
print(base.resolve())
PY
)"
BASE_CONFIG_INHERITS="$("$PYTHON_BIN" - <<PY
import os
from pathlib import Path
generated = Path("$ROOT") / "configs" / "sanity_generated"
base = Path("$BASE_CONFIG_ABS")
print(os.path.relpath(base, generated))
PY
)"
VARIANT_LABEL="${VARIANT_LABEL:-$(basename "${BASE_CONFIG%.yaml}")}"
VARIANT_LABEL="$(printf '%s' "$VARIANT_LABEL" | tr -c '[:alnum:]_.-' '_')"

cleanup_ray() {
  if [ -x "$RAY_BIN" ]; then
    "$RAY_BIN" stop --force >/dev/null 2>&1 || true
  fi
}

mkdir -p "$VERL_DATA_DIR" "$RUN_ROOT" "$MANIFEST_ROOT" "$ROOT/logs" "$ROOT/configs/sanity_generated"

"$PYTHON_BIN" -m qwen_single_layer_rl.data.verl_format \
  --train "$CANON_DATA_DIR/train.parquet" \
  --val "$CANON_DATA_DIR/val.parquet" \
  --out-dir "$VERL_DATA_DIR" >/dev/null

GPU_SAMPLE_LOG="$ROOT/logs/sanity_bs_sweep_resp3072_${RUN_STAMP}_gpu_samples.csv"
echo "timestamp,gpu_index,name,utilization_gpu,memory_used_mib,memory_total_mib" > "$GPU_SAMPLE_LOG"
(
  while true; do
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits |
      awk -v ts="$(date -Is)" '{print ts "," $0}' >> "$GPU_SAMPLE_LOG" || true
    sleep 30
  done
) &
SAMPLER_PID=$!
trap 'kill "$SAMPLER_PID" >/dev/null 2>&1 || true; cleanup_ray' EXIT

echo "SANITY_BS_SWEEP_START $(date -Is)"
echo "BASE_CONFIG=$BASE_CONFIG"
echo "BASE_CONFIG_INHERITS=$BASE_CONFIG_INHERITS"
echo "VARIANT_LABEL=$VARIANT_LABEL"
echo "INFERENCE_BACKEND=${INFERENCE_BACKEND:-from_config}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE:-from_config}"
echo "BATCH_SIZES=$BATCH_SIZES"
echo "CASE_TIMEOUT_SECONDS=$CASE_TIMEOUT_SECONDS"
echo "RESPONSE_LENGTH=3072"
echo "GPU_SAMPLE_LOG=$GPU_SAMPLE_LOG"

for BS in $BATCH_SIZES; do
  if [ "$BS" -le 0 ]; then
    echo "Invalid batch size: $BS" >&2
    exit 2
  fi

  MINI_BATCH="$BS"
  if [ "$MINI_BATCH" -gt 128 ]; then
    MINI_BATCH=128
  fi

  CFG="$ROOT/configs/sanity_generated/${VARIANT_LABEL}_sanity_bs${BS}_resp3072_${RUN_STAMP}.yaml"
  RUN_ID="qwen3_1p7b_${VARIANT_LABEL}_sanity_bs${BS}_resp3072_seed20260707_${RUN_STAMP}"
  BACKEND_LINE=""
  if [ -n "$INFERENCE_BACKEND" ]; then
    BACKEND_LINE="  inference_backend: ${INFERENCE_BACKEND}"
  fi
  NPROC_LINE=""
  if [ -n "$NPROC_PER_NODE" ]; then
    NPROC_LINE="  nproc_per_node: ${NPROC_PER_NODE}"
  fi
  cat > "$CFG" <<EOF
inherits: ${BASE_CONFIG_INHERITS}
experiment:
  name: qwen3_1p7b_${VARIANT_LABEL}_sanity_bs${BS}_resp3072
grpo:
  train_batch_size: ${BS}
  ppo_mini_batch_size: ${MINI_BATCH}
  ppo_micro_batch_size: 8
  max_response_length: 3072
  epochs: 1
runtime:
${BACKEND_LINE}
${NPROC_LINE}
  rollout:
    tensor_model_parallel_size: ${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE}
    max_num_seqs: ${ROLLOUT_MAX_NUM_SEQS}
    gpu_memory_utilization: ${ROLLOUT_GPU_MEMORY_UTILIZATION}
    max_num_batched_tokens: ${ROLLOUT_MAX_NUM_BATCHED_TOKENS}
logging:
  run_id_template: "${RUN_ID}"
EOF

  PLAN_DIR="$RUN_ROOT/plan_${VARIANT_LABEL}_sanity_bs${BS}_resp3072_${RUN_STAMP}"
  MANIFEST="$MANIFEST_ROOT/${VARIANT_LABEL}_sanity_bs${BS}_resp3072_${RUN_STAMP}_command.json"
  "$PYTHON_BIN" -m qwen_single_layer_rl.training.dry_run --config "$CFG" --out "$PLAN_DIR" >/dev/null
  "$PYTHON_BIN" -m qwen_single_layer_rl.training.verl_command \
    --config "$CFG" \
    --project-root "$ROOT" \
    --verl-root "$VERL_ROOT" \
    --model-path "$MODEL_PATH" \
    --data-dir "$VERL_DATA_DIR" \
    --run-root "$RUN_ROOT" \
    --manifest-out "$MANIFEST" >/dev/null

  SHELL_CMD="$(
    "$PYTHON_BIN" -m qwen_single_layer_rl.training.verl_command \
      --config "$CFG" \
      --project-root "$ROOT" \
      --verl-root "$VERL_ROOT" \
      --model-path "$MODEL_PATH" \
      --data-dir "$VERL_DATA_DIR" \
      --run-root "$RUN_ROOT" \
      --print-shell
  )"

  echo "RUN_CASE_START $(date -Is) batch_size=${BS} mini_batch=${MINI_BATCH} manifest=${MANIFEST}"
  cleanup_ray
  START_SECONDS=$SECONDS
  set +e
  if command -v timeout >/dev/null 2>&1; then
    timeout --foreground "$CASE_TIMEOUT_SECONDS" bash -lc "$SHELL_CMD trainer.total_training_steps=1 trainer.test_freq=0 trainer.save_freq=0"
    STATUS=$?
  else
    bash -lc "$SHELL_CMD trainer.total_training_steps=1 trainer.test_freq=0 trainer.save_freq=0"
    STATUS=$?
  fi
  set -e
  CASE_SECONDS=$((SECONDS - START_SECONDS))
  cleanup_ray
  echo "RUN_CASE_END $(date -Is) batch_size=${BS} status=${STATUS} wall_seconds=${CASE_SECONDS}"

  if [ "$STATUS" -ne 0 ]; then
    echo "SANITY_BS_SWEEP_STOP_AFTER_FAILURE batch_size=${BS} status=${STATUS}"
    exit "$STATUS"
  fi
done

echo "SANITY_BS_SWEEP_END $(date -Is)"
