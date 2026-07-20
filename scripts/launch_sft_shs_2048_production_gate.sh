#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
GATE_ROOT="${GATE_ROOT:-$ROOT/runs/sft_shs_2048_production_gate_$RUN_STAMP}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
TIMED_STEPS="${TIMED_STEPS:-5}"
CONFIG="configs/sft/layer10_whole_layer_shs_sft.yaml"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$GATE_ROOT" "$ROOT/logs"
cd "$ROOT"

run_case() {
  local case_name="$1"
  local compile_mode="$2"
  local out_dir="$GATE_ROOT/$case_name"
  local inductor_cache="$GATE_ROOT/inductor_cache/$case_name"
  echo "PRODUCTION_GATE_CASE_START $(date -Is) case=$case_name compile=$compile_mode"
  TORCHINDUCTOR_CACHE_DIR="$inductor_cache" "$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nproc_per_node=1 \
    -m qwen_single_layer_rl.sft.trainer \
    --config "$CONFIG" \
    --output-dir "$out_dir" \
    --run-id "$case_name" \
    --compile-mode "$compile_mode" \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --micro-batch-size "$MICRO_BATCH_SIZE" \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
    --epochs 1 \
    --benchmark \
    --no-resume \
    --warmup-steps "$WARMUP_STEPS" \
    --timed-steps "$TIMED_STEPS"
  echo "PRODUCTION_GATE_CASE_END $(date -Is) case=$case_name status=0"
}

echo "$GATE_ROOT" > "$ROOT/logs/current_sft_benchmark.runroot"
echo "SFT_SHS_2048_GATE_START $(date -Is) root=$GATE_ROOT"
run_case shs_eager eager
run_case shs_compile default
"$PYTHON_BIN" -m qwen_single_layer_rl.sft.summarize_benchmarks \
  --root "$GATE_ROOT" \
  --pair shs
echo "SFT_SHS_2048_GATE_END $(date -Is) root=$GATE_ROOT"
