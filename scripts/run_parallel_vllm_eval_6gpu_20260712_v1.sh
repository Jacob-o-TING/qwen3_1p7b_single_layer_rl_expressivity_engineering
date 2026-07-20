#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:?config}"
MODEL="${2:?merged model}"
OUT="${3:?output dir}"
PY="$ROOT/envs/vllm0102_verl061/bin/python"
mkdir -p "$OUT"
export PYTHONPATH="$ROOT/src:/root/autodl-tmp/verl-v0.6.1-qwenpatch${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_V1=1
export TOKENIZERS_PARALLELISM=false

run_rank() {
  local gpu="$1" name="$2" seed="$3"; shift 3
  mkdir -p "$OUT/$name"
  if [[ -s "$OUT/$name/model_summary.json" || -f "$OUT/$name/RANK_COMPLETE" ]]; then
    echo "EVAL_RANK_ALREADY_COMPLETE name=$name gpu=$gpu"
    touch "$OUT/$name/RANK_COMPLETE"
    return 0
  fi
  local cache_args=() latest
  latest=$(find "$OUT/$name/main" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort | tail -1 || true)
  [[ -z "$latest" ]] || cache_args+=(--main-use-cache "$OUT/$name/main/$latest")
  latest=$(find "$OUT/$name/amc_average_at_32" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort | tail -1 || true)
  [[ -z "$latest" ]] || cache_args+=(--amc-use-cache "$OUT/$name/amc_average_at_32/$latest")
  latest=$(find "$OUT/$name/amc_greedy" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort | tail -1 || true)
  [[ -z "$latest" ]] || cache_args+=(--amc-greedy-use-cache "$OUT/$name/amc_greedy/$latest")
  if (( ${#cache_args[@]} )); then
    echo "EVAL_RANK_RESUME name=$name gpu=$gpu cache_args=${cache_args[*]}"
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m qwen_single_layer_rl.eval.run_evalscope \
    --config "$CONFIG" --base-model-only --model-path "$MODEL" --work-dir "$OUT/$name" \
    --backend vllm --vllm-model-impl auto --vllm-enforce-eager \
    --vllm-gpu-memory-utilization 0.85 --vllm-max-num-seqs 128 \
    --vllm-max-num-batched-tokens 32768 --microbatch-wait-seconds 0.25 \
    --seed "$seed" --max-tokens 3072 --eval-batch-size 128 "${cache_args[@]}" "$@" \
    >"$OUT/$name.log" 2>&1
  touch "$OUT/$name/RANK_COMPLETE"
}

run_rank 0 math500 20260707 --datasets paper_math500 --amc-repeats 0 --skip-amc-greedy & p0=$!
run_rank 1 gsm8k 20260707 --datasets paper_gsm8k --amc-repeats 0 --skip-amc-greedy & p1=$!
run_rank 2 olympiad 20260707 --datasets paper_olympiadbench --amc-repeats 0 --skip-amc-greedy & p2=$!
run_rank 3 amc_sample_00_10 20260707 --amc-only --amc-repeats 11 --skip-amc-greedy & p3=$!
run_rank 4 amc_sample_11_21 20260718 --amc-only --amc-repeats 11 --skip-amc-greedy & p4=$!
run_rank 5 amc_sample_22_31_and_greedy 20260729 --amc-only --amc-repeats 10 & p5=$!

status=0
for pid in "$p0" "$p1" "$p2" "$p3" "$p4" "$p5"; do wait "$pid" || status=$?; done
if (( status != 0 )); then
  echo "PARALLEL_EVAL_FAILED $status" >&2
  exit "$status"
fi
touch "$OUT/PARALLEL_EVAL_COMPLETE"
"$PY" "$ROOT/scripts/summarize_parallel_eval.py" "$OUT" --json-out "$OUT/summary.json" | tee "$OUT/summary.txt"
