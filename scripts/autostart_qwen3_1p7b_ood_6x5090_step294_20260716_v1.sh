#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GRPO_ROOT="$ROOT/runs/grpo_priority/triglu_priority_6x5090_grpo_step128_to294_then_baseline128_to196_20260715_v1"
OUT="$ROOT/runs/ood_eval/qwen3_1p7b_ood_6x5090_step294_20260716_v1"
OOD_RUNNER="$ROOT/scripts/run_qwen3_1p7b_ood_6x5090_step294_20260716_v1.sh"
mkdir -p "$OUT/logs"
exec > >(tee -a "$OUT/logs/autostart.log") 2>&1

if [[ -f "$OUT/OOD_COMPLETE" ]]; then
  echo "OOD_ALREADY_COMPLETE"
  exit 0
fi
echo "OOD_AUTOSTART_ARMED $(date -Is) order=triglu_math,triglu_ood,untuned_base_ood,baseline_third98,baseline_math,baseline_ood"

fail_safe_exit() {
  local rc=$?
  trap - EXIT HUP INT TERM
  if (( rc != 0 )); then
    echo "OOD_AUTOSTART_FAILED rc=$rc time=$(date -Is)"
  fi
  exit "$rc"
}
trap fail_safe_exit EXIT HUP INT TERM

wait_for_grpo_artifact() {
  local path="$1" label="$2"
  while [[ ! -f "$path" ]]; do
    if [[ -f "$GRPO_ROOT/WAVE_FAILED" ]]; then
      echo "OOD_AUTOSTART_BLOCKED_BY_GRPO_FAILURE $(cat "$GRPO_ROOT/WAVE_FAILED")"
      return 1
    fi
    sleep 2
  done
  echo "OOD_PREREQUISITE_READY label=$label time=$(date -Is)"
}

wait_for_math_eval_exit_and_idle_gpus() {
  local deadline=$(( $(date +%s) + 1800 ))
  while pgrep -f '[r]un_parallel_vllm_eval_6gpu_20260712_v1[.]sh' >/dev/null; do
    (( $(date +%s) < deadline )) || { echo "MATH_EVAL_EXIT_TIMEOUT"; return 1; }
    sleep 2
  done
  ray stop --force >/dev/null 2>&1 || true
  while ! nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | \
      awk '$1 > 1024 {busy=1} END {exit busy}'; do
    (( $(date +%s) < deadline )) || { echo "GPU_IDLE_TIMEOUT"; return 1; }
    sleep 5
  done
  echo "OOD_GPU_HANDOFF_READY $(date -Is)"
}

if [[ ! -f "$OUT/PRE_BASELINE_OOD_COMPLETE" ]]; then
  wait_for_grpo_artifact "$GRPO_ROOT/TRIGLU_294_PRE_BASELINE_OOD_READY" triglu_pre_baseline_barrier
  wait_for_grpo_artifact "$GRPO_ROOT/evaluations/triglu_step_294/PARALLEL_EVAL_COMPLETE" triglu_math_step294
  wait_for_grpo_artifact "$GRPO_ROOT/exports/triglu_step_294/EXPORT_COMPLETE" triglu_export_step294
  wait_for_math_eval_exit_and_idle_gpus
  bash "$OOD_RUNNER" triglu untuned_base
  [[ -f "$OUT/PRE_BASELINE_OOD_COMPLETE" ]]
else
  echo "PRE_BASELINE_OOD_ALREADY_COMPLETE"
fi

wait_for_grpo_artifact "$GRPO_ROOT/evaluations/baseline_step_294/PARALLEL_EVAL_COMPLETE" baseline_math_step294
wait_for_grpo_artifact "$GRPO_ROOT/exports/baseline_step_294/EXPORT_COMPLETE" baseline_export_step294
while [[ ! -f "$GRPO_ROOT/WAVE_COMPLETE" ]]; do
  if [[ -f "$GRPO_ROOT/WAVE_FAILED" ]]; then
    echo "OOD_AUTOSTART_BLOCKED_BY_GRPO_FAILURE $(cat "$GRPO_ROOT/WAVE_FAILED")"
    exit 1
  fi
  sleep 2
done
wait_for_math_eval_exit_and_idle_gpus
bash "$OOD_RUNNER" baseline
[[ -f "$OUT/OOD_COMPLETE" ]]
trap - EXIT HUP INT TERM
echo "OOD_AUTOSTART_COMPLETE $(date -Is)"
