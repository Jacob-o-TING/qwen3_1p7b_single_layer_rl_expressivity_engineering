#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:?config}"
MODEL="${2:?model}"
MODEL_OUT="${3:?model output root}"
PY="$ROOT/envs/vllm0102_verl061/bin/python"
RUN_ID=livecodebench_release_latest_6way_hashshard_20260717_v1
OUT="$MODEL_OUT/code_lcb_parallel6_release_latest_20260717_v1"
CANONICAL="$MODEL_OUT/code_lcb"
mkdir -p "$OUT/shards" "$CANONICAL"
export PYTHONPATH="$ROOT/src:/root/autodl-tmp/verl-v0.6.1-qwenpatch${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_V1=1
export TOKENIZERS_PARALLELISM=false

run_shard() {
  local gpu="$1" shard_index="$2" shard out latest
  shard=$(printf 'shard_%02d' "$shard_index")
  out="$OUT/shards/$shard"
  mkdir -p "$out"
  if [[ -f "$out/RANK_COMPLETE" ]]; then
    echo "LCB_SHARD_ALREADY_COMPLETE shard=$shard gpu=$gpu"
    return 0
  fi
  local cache_args=()
  latest=$(find "$out/main" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort | tail -1 || true)
  [[ -z "$latest" ]] || cache_args+=(--main-use-cache "$out/main/$latest")
  echo "LCB_SHARD_START run_id=$RUN_ID shard=$shard gpu=$gpu time=$(date -Is)"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m qwen_single_layer_rl.eval.run_evalscope \
    --config "$CONFIG" --base-model-only --model-path "$MODEL" --work-dir "$out" \
    --backend vllm --vllm-model-impl auto --vllm-enforce-eager \
    --vllm-gpu-memory-utilization 0.85 --vllm-max-num-seqs 128 \
    --vllm-max-num-batched-tokens 32768 --microbatch-wait-seconds 0.25 \
    --seed 20260707 --max-tokens 3072 --eval-batch-size 128 \
    --amc-repeats 0 --skip-amc-greedy --local-code-sandbox \
    --datasets live_code_bench --dataset-subsets release_latest \
    --dataset-shard-count 6 --dataset-shard-index "$shard_index" \
    "${cache_args[@]}" >"$OUT/$shard.log" 2>&1
  touch "$out/RANK_COMPLETE"
  echo "LCB_SHARD_COMPLETE run_id=$RUN_ID shard=$shard gpu=$gpu time=$(date -Is)"
}

pids=()
for gpu in 0 1 2 3 4 5; do
  run_shard "$gpu" "$gpu" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
if (( status != 0 )); then
  echo "LCB_PARALLEL_FAILED status=$status" >&2
  exit "$status"
fi

"$PY" "$ROOT/scripts/merge_livecodebench_shards.py" "$OUT" \
  --shard-count 6 --expected-samples 1055 | tee "$OUT/merge.log"
cp "$OUT/merge_summary.json" "$CANONICAL/model_summary.json"
touch "$OUT/LCB_PARALLEL_COMPLETE" "$CANONICAL/RANK_COMPLETE"
echo "LCB_PARALLEL_COMPLETE run_id=$RUN_ID time=$(date -Is)"
