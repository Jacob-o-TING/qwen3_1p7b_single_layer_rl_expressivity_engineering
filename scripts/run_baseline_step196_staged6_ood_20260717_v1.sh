#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GRPO_ROOT="$ROOT/runs/grpo_priority/triglu_priority_6x5090_grpo_step128_to294_then_baseline128_to196_20260715_v1"
RUN_ID=qwen3_1p7b_ood_6x5090_baseline_step196_20260717_v1
OUT="$ROOT/runs/ood_eval/$RUN_ID"
MODEL="$GRPO_ROOT/exports/baseline_step_196"
CONFIG="$ROOT/configs/runtime/baseline_6x5090_grpo_resume196_to294_cosine5e7_to5e8_seed20260707_v1.yaml"
PY="$ROOT/envs/vllm0102_verl061/bin/python"
mkdir -p "$OUT/logs"
exec > >(tee -a "$OUT/logs/controller.log") 2>&1
exec 9>"$OUT/controller.lock"
flock 9

export PYTHONPATH="$ROOT/src:/root/autodl-tmp/verl-v0.6.1-qwenpatch${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_V1=1
export TOKENIZERS_PARALLELISM=false

old_root_mode=$(stat -c %a /root)
restore_root_mode() { chmod "$old_root_mode" /root 2>/dev/null || true; }
trap restore_root_mode EXIT INT TERM
chmod 701 /root

[[ -f "$MODEL/EXPORT_COMPLETE" && -s "$MODEL/config.json" ]]
compgen -G "$MODEL/*.safetensors" >/dev/null
[[ $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) -eq 6 ]]
free=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
(( free >= 80 )) || { echo "OOD_DISK_GUARD_FAIL free=${free}G required=80G"; exit 80; }

write_state() {
  local phase="$1" benchmark="$2" stage_index="$3" stage_count="$4"
  printf 'PHASE=%s\nMODEL=baseline_step196\nBENCHMARK=%s\nSTAGE_INDEX=%s\nSTAGE_COUNT=%s\nUPDATED_UNIX=%s\n' \
    "$phase" "$benchmark" "$stage_index" "$stage_count" "$(date +%s)" >"$OUT/state.env.tmp"
  mv "$OUT/state.env.tmp" "$OUT/state.env"
}

run_stage() {
  local stage_index="$1" stage_name="$2" dataset="$3" expected="$4" sandbox="$5" parser_v2="$6"
  local stage="$OUT/$stage_name"
  if [[ -f "$stage/RANK_COMPLETE" ]]; then
    echo "STAGED6_ALREADY_COMPLETE stage=$stage_name dataset=$dataset"
    return 0
  fi
  mkdir -p "$stage/shards"
  write_state EVAL "$dataset" "$stage_index" 8
  echo "STAGED6_START stage=$stage_index/8 dataset=$dataset expected=$expected time=$(date -Is)"

  run_shard() {
    local gpu="$1" shard_index="$2" shard latest
    shard=$(printf 'shard_%02d' "$shard_index")
    local shard_out="$stage/shards/$shard"
    mkdir -p "$shard_out"
    if [[ -f "$shard_out/RANK_COMPLETE" ]]; then
      echo "STAGED6_SHARD_ALREADY_COMPLETE dataset=$dataset shard=$shard"
      return 0
    fi
    local cache_args=() sandbox_args=() parser_args=()
    latest=$(find "$shard_out/main" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort | tail -1 || true)
    [[ -z "$latest" ]] || cache_args+=(--main-use-cache "$shard_out/main/$latest")
    [[ "$sandbox" == yes ]] && sandbox_args+=(--local-code-sandbox)
    [[ "$parser_v2" == yes ]] && parser_args+=(--humanevalplus-parser-v2)
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m qwen_single_layer_rl.eval.run_evalscope \
      --config "$CONFIG" --base-model-only --model-path "$MODEL" --work-dir "$shard_out" \
      --backend vllm --vllm-model-impl auto --vllm-enforce-eager \
      --vllm-gpu-memory-utilization 0.85 --vllm-max-num-seqs 128 \
      --vllm-max-num-batched-tokens 32768 --microbatch-wait-seconds 0.25 \
      --seed 20260707 --max-tokens 3072 --eval-batch-size 128 \
      --amc-repeats 0 --skip-amc-greedy --datasets "$dataset" \
      --dataset-shard-count 6 --dataset-shard-index "$shard_index" \
      "${cache_args[@]}" "${sandbox_args[@]}" "${parser_args[@]}" \
      >"$stage/$shard.log" 2>&1
    touch "$shard_out/RANK_COMPLETE"
  }

  pids=()
  for gpu in 0 1 2 3 4 5; do
    run_shard "$gpu" "$gpu" &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
  if (( status != 0 )); then
    echo "STAGED6_FAILED dataset=$dataset status=$status" >&2
    return "$status"
  fi
  "$PY" "$ROOT/scripts/merge_generic_evalscope_shards.py" "$stage" \
    --dataset "$dataset" --shard-count 6 --expected-identities "$expected" \
    | tee "$stage/merge.log"
  touch "$stage/RANK_COMPLETE"
  ray stop --force >/dev/null 2>&1 || true
  echo "STAGED6_COMPLETE stage=$stage_index/8 dataset=$dataset time=$(date -Is)"
}

trap 'rc=$?; [[ $rc -eq 0 ]] || printf "BASELINE_STEP196_OOD_FAILED rc=%s time=%s\n" "$rc" "$(date -Is)" | tee "$OUT/OOD_FAILED"; restore_root_mode; exit $rc' EXIT
run_stage 1 reasoning_gpqa_staged6 gpqa_diamond 198 no no
run_stage 2 reasoning_mmlupro_staged6 mmlu_pro 12032 no no
run_stage 3 code_humanevalplus_staged6 humaneval_plus 164 yes yes
run_stage 4 code_mbpp_staged6 mbpp 500 yes no
run_stage 5 language_ceval_staged6 ceval 1346 no no
run_stage 6 language_ifeval_staged6 ifeval 541 no no
run_stage 7 language_mgsm_staged6 mgsm 2750 no no

write_state EVAL live_code_bench 8 8
"$ROOT/scripts/run_livecodebench_parallel6_vllm_20260717_v1.sh" \
  "$CONFIG" "$MODEL" "$OUT"
touch "$OUT/PARALLEL_OOD_EVAL_COMPLETE"
"$PY" "$ROOT/scripts/summarize_ood_eval.py" "$OUT" --json-out "$OUT/summary.json" \
  | tee "$OUT/summary.txt"
write_state COMPLETE all 8 8
touch "$OUT/OOD_COMPLETE"
rm -f "$OUT/OOD_FAILED"
trap - EXIT
restore_root_mode
echo "BASELINE_STEP196_OOD_COMPLETE run_id=$RUN_ID time=$(date -Is)"
