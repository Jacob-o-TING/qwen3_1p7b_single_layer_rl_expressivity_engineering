#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$ROOT/runs/sft_compile_short_benchmark_$RUN_STAMP}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
TIMED_STEPS="${TIMED_STEPS:-20}"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$BENCHMARK_ROOT" "$ROOT/logs"
cd "$ROOT"

run_case() {
  local case_name="$1"
  local config="$2"
  local compile_mode="$3"
  local out_dir="$BENCHMARK_ROOT/$case_name"
  local case_log="$BENCHMARK_ROOT/$case_name.log"
  local inductor_cache="$BENCHMARK_ROOT/inductor_cache/$case_name"
  mkdir -p "$out_dir"
  echo "BENCHMARK_CASE_START $(date -Is) case=$case_name config=$config compile=$compile_mode"
  set +e
  TORCHINDUCTOR_CACHE_DIR="$inductor_cache" "$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nproc_per_node=1 \
    -m qwen_single_layer_rl.sft.trainer \
    --config "$config" \
    --output-dir "$out_dir" \
    --run-id "$case_name" \
    --compile-mode "$compile_mode" \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --micro-batch-size "$MICRO_BATCH_SIZE" \
    --gradient-accumulation-steps 1 \
    --epochs 1 \
    --benchmark \
    --no-resume \
    --warmup-steps "$WARMUP_STEPS" \
    --timed-steps "$TIMED_STEPS" \
    2>&1 | tee "$case_log"
  status=${PIPESTATUS[0]}
  set -e
  echo "BENCHMARK_CASE_END $(date -Is) case=$case_name status=$status"
  if [[ "$status" -ne 0 ]]; then
    return "$status"
  fi
}

echo "$BENCHMARK_ROOT" > "$ROOT/logs/current_sft_benchmark.runroot"
echo "SFT_BENCHMARK_START $(date -Is) root=$BENCHMARK_ROOT max_seq=$MAX_SEQ_LENGTH"
run_case baseline_eager configs/sft/layer10_whole_layer_baseline_sft.yaml eager
run_case baseline_compile configs/sft/layer10_whole_layer_baseline_sft.yaml default
run_case shs_eager configs/sft/layer10_whole_layer_shs_sft.yaml eager
run_case shs_compile configs/sft/layer10_whole_layer_shs_sft.yaml default
"$PYTHON_BIN" -m qwen_single_layer_rl.sft.summarize_benchmarks --root "$BENCHMARK_ROOT"
echo "SFT_BENCHMARK_END $(date -Is) root=$BENCHMARK_ROOT"
