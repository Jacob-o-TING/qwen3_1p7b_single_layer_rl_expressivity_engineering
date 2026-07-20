#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GRPO_ROOT="$ROOT/runs/grpo_priority/triglu_priority_6x5090_grpo_step128_to294_then_baseline128_to196_20260715_v1"
RUN_ID=qwen3_1p7b_ood_6x5090_step294_20260716_v1
OUT="$ROOT/runs/ood_eval/$RUN_ID"
BASE_MODEL=/root/autodl-tmp/qwen3_single_layer_rl/models/Qwen3-1.7B-Base
PY="$ROOT/envs/vllm0102_verl061/bin/python"
mkdir -p "$OUT/logs"
exec > >(tee -a "$OUT/logs/controller.log") 2>&1
exec 9>"$OUT/controller.lock"
flock 9

old_root_mode=$(stat -c %a /root)
restore_root_mode() { chmod "$old_root_mode" /root 2>/dev/null || true; }
trap restore_root_mode EXIT INT TERM
chmod 701 /root
rm -f "$OUT/OOD_FAILED"

free=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
(( free >= 100 )) || { echo "OOD_DISK_GUARD_FAIL free=${free}G required=100G"; exit 80; }
[[ $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) -eq 6 ]]
[[ -x /usr/bin/setpriv && -r /usr/lib/x86_64-linux-gnu/libseccomp.so.2 ]]

run_model() {
  local name="$1" config model output="$OUT/$1"
  case "$name" in
    triglu)
      config="$ROOT/configs/runtime/triglu_6x5090_grpo_resume196_to294_cosine5e7_to5e8_seed20260707_v1.yaml"
      model="$GRPO_ROOT/exports/triglu_step_294"
      ;;
    baseline)
      config="$ROOT/configs/runtime/baseline_6x5090_grpo_resume196_to294_cosine5e7_to5e8_seed20260707_v1.yaml"
      model="$GRPO_ROOT/exports/baseline_step_294"
      ;;
    untuned_base)
      config="$ROOT/configs/runtime/baseline_6x5090_grpo_resume196_to294_cosine5e7_to5e8_seed20260707_v1.yaml"
      model="$BASE_MODEL"
      ;;
    *)
      echo "OOD_UNKNOWN_MODEL model=$name" >&2
      return 64
      ;;
  esac
  if [[ -f "$output/PARALLEL_OOD_EVAL_COMPLETE" ]]; then
    echo "OOD_MODEL_ALREADY_COMPLETE model=$name"
    return 0
  fi
  printf 'PHASE=EVAL\nMODEL=%s\nUPDATED_UNIX=%s\n' "$name" "$(date +%s)" >"$OUT/state.env.tmp"
  mv "$OUT/state.env.tmp" "$OUT/state.env"
  [[ -s "$model/config.json" ]]
  compgen -G "$model/*.safetensors" >/dev/null
  "$ROOT/scripts/run_parallel_ood_vllm_eval_6gpu_20260716_v1.sh" "$config" "$model" "$output" \
    2>&1 | tee "$OUT/logs/${name}.log"
  ray stop --force >/dev/null 2>&1 || true
}

trap 'rc=$?; [[ $rc -eq 0 ]] || printf "OOD_FAILED rc=%s time=%s\n" "$rc" "$(date -Is)" | tee "$OUT/OOD_FAILED"; restore_root_mode; exit $rc' EXIT
requested_models=("$@")
if (( ${#requested_models[@]} == 0 )); then
  requested_models=(triglu untuned_base baseline)
fi
echo "OOD_PHASE_START $(date -Is) models=${requested_models[*]}"
for model_name in "${requested_models[@]}"; do
  run_model "$model_name"
done

if [[ -f "$OUT/triglu/PARALLEL_OOD_EVAL_COMPLETE" && \
      -f "$OUT/untuned_base/PARALLEL_OOD_EVAL_COMPLETE" ]]; then
  touch "$OUT/PRE_BASELINE_OOD_COMPLETE"
fi
if [[ -f "$OUT/triglu/PARALLEL_OOD_EVAL_COMPLETE" && \
      -f "$OUT/untuned_base/PARALLEL_OOD_EVAL_COMPLETE" && \
      -f "$OUT/baseline/PARALLEL_OOD_EVAL_COMPLETE" ]]; then
  "$PY" "$ROOT/scripts/summarize_ood_eval.py" "$OUT" --compare-models | tee "$OUT/comparison.txt"
  printf 'PHASE=COMPLETE\nMODEL=all\nUPDATED_UNIX=%s\n' "$(date +%s)" >"$OUT/state.env"
  touch "$OUT/OOD_COMPLETE"
  echo "OOD_COMPLETE $(date -Is)"
else
  printf 'PHASE=WAITING_FOR_BASELINE\nMODEL=baseline\nUPDATED_UNIX=%s\n' "$(date +%s)" >"$OUT/state.env"
  echo "OOD_PHASE_COMPLETE $(date -Is) models=${requested_models[*]} waiting_for=baseline"
fi
rm -f "$OUT/OOD_FAILED"
trap - EXIT
restore_root_mode
