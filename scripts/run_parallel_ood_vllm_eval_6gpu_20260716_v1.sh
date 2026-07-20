#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:?config}"
MODEL="${2:?model}"
OUT="${3:?output dir}"
PY="$ROOT/envs/vllm0102_verl061/bin/python"
mkdir -p "$OUT"
export PYTHONPATH="$ROOT/src:/root/autodl-tmp/verl-v0.6.1-qwenpatch${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_V1=1
export TOKENIZERS_PARALLELISM=false

"$PY" - <<'PY'
import langdetect
PY

run_rank() {
  local gpu="$1" name="$2" sandbox="$3"; shift 3
  mkdir -p "$OUT/$name"
  if [[ -s "$OUT/$name/model_summary.json" || -f "$OUT/$name/RANK_COMPLETE" ]]; then
    echo "OOD_RANK_ALREADY_COMPLETE name=$name gpu=$gpu"
    touch "$OUT/$name/RANK_COMPLETE"
    return 0
  fi
  local cache_args=() latest sandbox_args=()
  latest=$(find "$OUT/$name/main" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort | tail -1 || true)
  [[ -z "$latest" ]] || cache_args+=(--main-use-cache "$OUT/$name/main/$latest")
  [[ "$sandbox" == yes ]] && sandbox_args+=(--local-code-sandbox)
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m qwen_single_layer_rl.eval.run_evalscope \
    --config "$CONFIG" --base-model-only --model-path "$MODEL" --work-dir "$OUT/$name" \
    --backend vllm --vllm-model-impl auto --vllm-enforce-eager \
    --vllm-gpu-memory-utilization 0.85 --vllm-max-num-seqs 128 \
    --vllm-max-num-batched-tokens 32768 --microbatch-wait-seconds 0.25 \
    --seed 20260707 --max-tokens 3072 --eval-batch-size 128 \
    --amc-repeats 0 --skip-amc-greedy "${cache_args[@]}" "${sandbox_args[@]}" "$@" \
    >"$OUT/$name.log" 2>&1
  touch "$OUT/$name/RANK_COMPLETE"
}

run_rank 0 code_heplus_mbpp yes --datasets humaneval_plus mbpp & p0=$!
run_rank 1 reasoning_gpqa no --datasets gpqa_diamond & p1=$!
run_rank 2 reasoning_mmlupro no --datasets mmlu_pro & p2=$!
run_rank 3 language_ceval no --datasets ceval & p3=$!
run_rank 4 language_ifeval_mgsm no --datasets ifeval mgsm & p4=$!

status=0
for pid in "$p0" "$p1" "$p2" "$p3" "$p4"; do wait "$pid" || status=$?; done
if (( status != 0 )); then
  echo "PARALLEL_OOD_EVAL_FAILED $status" >&2
  exit "$status"
fi

"$ROOT/scripts/run_livecodebench_parallel6_vllm_20260717_v1.sh" \
  "$CONFIG" "$MODEL" "$OUT"
touch "$OUT/PARALLEL_OOD_EVAL_COMPLETE"
"$PY" "$ROOT/scripts/summarize_ood_eval.py" "$OUT" --json-out "$OUT/summary.json" | tee "$OUT/summary.txt"
