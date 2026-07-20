#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_WAVE=triglu_baseline_6x5090_grpo_retro30_60_then_98to196_interleaved_20260714_v1
SOURCE_ROOT="$ROOT/runs/grpo_interleaved/$SOURCE_WAVE"
SOURCE_SCREEN=qwen_grpo_98to196_interleaved_20260714_v1
SOURCE_TRIGLU_RUN=triglu_6x5090_grpo_untunedbase_resume98_to196_seed20260707_v1
SOURCE_BASELINE_RUN=baseline_6x5090_grpo_untunedbase_resume98_to196_seed20260707_v1
TRIGLU_THIRD_RUN=triglu_6x5090_grpo_resume196_to294_cosine5e7_to5e8_seed20260707_v1
WAVE=triglu_priority_6x5090_grpo_step128_to294_then_baseline128_to196_20260715_v1
RUN_ROOT="$ROOT/runs/grpo_priority/$WAVE"
SUCCESSOR_SCREEN=qwen_triglu_priority_to294_then_baseline196_20260715_v1
CONTROLLER="$ROOT/scripts/run_triglu_priority_to294_then_baseline196_20260715_v1.sh"
CONTROLLER_NEXT="$CONTROLLER.next"
STATE="$RUN_ROOT/handoff_state.env"
RESET_RECEIPT="$RUN_ROOT/receipts/third_stage_reference_fail_closed.txt"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/receipts"
exec > >(tee -a "$RUN_ROOT/logs/handoff_watcher.log") 2>&1

set_state() {
  printf 'HANDOFF_PHASE=%q\nUPDATED_UNIX=%q\n' "$1" "$(date +%s)" >"$STATE.tmp"
  mv "$STATE.tmp" "$STATE"
}

fail() {
  local message="$1"
  set_state FAILED
  printf 'HANDOFF_FAILED time=%s reason=%s\n' "$(date -Is)" "$message" | tee "$RUN_ROOT/HANDOFF_FAILED"
  exit 1
}

validate_checkpoint() {
  local checkpoint="$1" actor="$1/actor"
  [[ -s "$checkpoint/data.pt" ]]
  [[ -s "$actor/fsdp_config.json" ]]
  [[ $(find "$actor" -maxdepth 1 -type f -name 'model_world_size_6_rank_*.pt' | wc -l) -eq 6 ]]
  [[ $(find "$actor" -maxdepth 1 -type f -name 'optim_world_size_6_rank_*.pt' | wc -l) -eq 6 ]]
  [[ $(find "$actor" -maxdepth 1 -type f -name 'extra_state_world_size_6_rank_*.pt' | wc -l) -eq 6 ]]
}

source_screen_exists() {
  screen -ls 2>/dev/null | grep -q "[.]$SOURCE_SCREEN[[:space:]]"
}

successor_screen_exists() {
  screen -ls 2>/dev/null | grep -q "[.]$SUCCESSOR_SCREEN[[:space:]]"
}

triglu_158_trainer_running() {
  pgrep -af 'verl[.]trainer[.]main_ppo.*trainer[.]experiment_name=triglu_6x5090_grpo_untunedbase_resume98_to196_seed20260707_v1.*trainer[.]total_training_steps=158' >/dev/null
}

baseline_158_trainer_running() {
  pgrep -af 'verl[.]trainer[.]main_ppo.*trainer[.]experiment_name=baseline_6x5090_grpo_untunedbase_resume98_to196_seed20260707_v1.*trainer[.]total_training_steps=158' >/dev/null
}

if successor_screen_exists || [[ -f "$RESET_RECEIPT" ]]; then
  [[ -f "$CONTROLLER_NEXT" ]] || fail missing_extended_controller_next
  bash -n "$CONTROLLER_NEXT" || fail invalid_extended_controller_next
  rm -f "$RUN_ROOT/HANDOFF_FAILED"
  set_state WAITING_FOR_BOTH_TO_294_EXTENSION
  echo "EXTENSION_WATCHER_READY $(date -Is) active_screen=$SUCCESSOR_SCREEN"
  while successor_screen_exists; do
    if [[ ! -f "$RESET_RECEIPT" ]]; then
      [[ ! -f "$RUN_ROOT/WAVE_FAILED" ]] || fail active_controller_failed_before_extension
    fi
    sleep 5
  done

  blocked_marker="$SOURCE_ROOT/exports/triglu_step_98/EXPORT_COMPLETE.blocked_before_third_stage_step196_reference"
  if [[ -f "$RESET_RECEIPT" ]]; then
    set_state VALIDATING_STEP196_REFERENCE_RESET_BOUNDARY
    validate_checkpoint "$SOURCE_ROOT/$SOURCE_BASELINE_RUN/checkpoints/global_step_196" \
      || fail baseline_step196_checkpoint_missing
    validate_checkpoint "$SOURCE_ROOT/$SOURCE_TRIGLU_RUN/checkpoints/global_step_196" \
      || fail triglu_step196_checkpoint_missing
    [[ -f "$RUN_ROOT/evaluations/baseline_step_196/PARALLEL_EVAL_COMPLETE" ]] \
      || fail baseline_step196_eval_missing
    [[ -f "$SOURCE_ROOT/evaluations/triglu_step_196/PARALLEL_EVAL_COMPLETE" || \
       -f "$RUN_ROOT/evaluations/triglu_step_196/PARALLEL_EVAL_COMPLETE" ]] \
      || fail triglu_step196_eval_missing
    for checkpoint in "$RUN_ROOT/$TRIGLU_THIRD_RUN/checkpoints"/global_step_*; do
      [[ -e "$checkpoint" ]] || continue
      step=${checkpoint##*_}
      (( step <= 196 )) || fail third_stage_update_exists_before_step196_reference_reset
    done
    [[ -f "$blocked_marker" ]] || fail fail_closed_step98_marker_missing
    mv "$blocked_marker" "${blocked_marker%.blocked_before_third_stage_step196_reference}"
    echo "STEP196_REFERENCE_RESET_BOUNDARY_VALIDATED $(date -Is)"
  else
    [[ -f "$RUN_ROOT/WAVE_COMPLETE" ]] || fail active_controller_exited_without_completion
    validate_checkpoint "$SOURCE_ROOT/$SOURCE_BASELINE_RUN/checkpoints/global_step_196" \
      || fail baseline_step196_checkpoint_missing
    validate_checkpoint "$RUN_ROOT/$TRIGLU_THIRD_RUN/checkpoints/global_step_294" \
      || fail triglu_step294_checkpoint_missing
    [[ -f "$RUN_ROOT/evaluations/baseline_step_196/PARALLEL_EVAL_COMPLETE" ]] \
      || fail baseline_step196_eval_missing
    [[ -f "$RUN_ROOT/evaluations/triglu_step_294/PARALLEL_EVAL_COMPLETE" ]] \
      || fail triglu_step294_eval_missing
  fi

  ray stop --force >/dev/null 2>&1 || true
  cp "$CONTROLLER_NEXT" "$CONTROLLER.promote"
  chmod +x "$CONTROLLER.promote"
  mv "$CONTROLLER.promote" "$CONTROLLER"
  rm -f "$RUN_ROOT/WAVE_COMPLETE"
  screen -dmS "$SUCCESSOR_SCREEN" bash "$CONTROLLER"
  sleep 3
  successor_screen_exists || fail extended_controller_failed_to_start
  set_state BOTH_TO_294_EXTENSION_LAUNCHED
  echo "EXTENSION_HANDOFF_COMPLETE $(date -Is) successor_screen=$SUCCESSOR_SCREEN"
  exit 0
fi

[[ -x "$CONTROLLER" || -f "$CONTROLLER" ]] || fail missing_successor_controller
[[ $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) -eq 6 ]] || fail gpu_count_not_six
[[ -f "$SOURCE_ROOT/evaluations/triglu_step_128/PARALLEL_EVAL_COMPLETE" ]] || fail triglu_step128_eval_missing
source_screen_exists || fail source_controller_screen_missing
successor_screen_exists && fail successor_screen_already_exists
rm -f "$RUN_ROOT/HANDOFF_FAILED"
set_state WAITING_FOR_TRIGLU_158
echo "HANDOFF_WATCHER_READY $(date -Is) source_screen=$SOURCE_SCREEN successor_screen=$SUCCESSOR_SCREEN"

seen_triglu_158=false
while true; do
  if [[ -f "$SOURCE_ROOT/state.env" ]]; then
    # shellcheck disable=SC1090
    source "$SOURCE_ROOT/state.env"
    if [[ "${PHASE:-}" == TRAIN && "${VARIANT:-}" == triglu && "${TARGET:-0}" -eq 158 ]]; then
      seen_triglu_158=true
      set_state TRIGLU_158_ACTIVE
    fi
  fi

  if baseline_158_trainer_running; then
    echo "LATE_HANDOFF_BASELINE_158_DETECTED $(date -Is)"
    seen_triglu_158=true
  fi

  checkpoint="$SOURCE_ROOT/$SOURCE_TRIGLU_RUN/checkpoints/global_step_158"
  if [[ "$seen_triglu_158" == true ]] && validate_checkpoint "$checkpoint" && ! triglu_158_trainer_running; then
    break
  fi
  source_screen_exists || fail source_controller_disappeared_before_triglu158_checkpoint
  sleep 0.25
done

set_state TAKING_OVER
echo "TAKEOVER_BOUNDARY_REACHED $(date -Is) checkpoint=$checkpoint"
screen -S "$SOURCE_SCREEN" -X quit || true

for _ in $(seq 1 120); do
  source_screen_exists || break
  sleep 1
done
source_screen_exists && fail source_screen_did_not_exit

ray stop --force >/dev/null 2>&1 || true
for _ in $(seq 1 120); do
  if ! pgrep -af 'verl[.]trainer[.]main_ppo|run_evalscope|parallel_vllm_eval' >/dev/null; then
    break
  fi
  sleep 1
done
pgrep -af 'verl[.]trainer[.]main_ppo|run_evalscope|parallel_vllm_eval' >/dev/null && fail gpu_processes_remain_after_takeover

validate_checkpoint "$SOURCE_ROOT/$SOURCE_BASELINE_RUN/checkpoints/global_step_128" || fail baseline_step128_checkpoint_missing
[[ -f "$SOURCE_ROOT/evaluations/baseline_step_128/PARALLEL_EVAL_COMPLETE" ]] || fail baseline_step128_eval_missing
[[ -f "$SOURCE_ROOT/data_order/new_step_128.json" ]] || fail paired_step128_data_receipt_missing

screen -dmS "$SUCCESSOR_SCREEN" bash "$CONTROLLER"
sleep 3
successor_screen_exists || fail successor_screen_failed_to_start
set_state LAUNCHED
{
  echo "HANDOFF_COMPLETE $(date -Is)"
  echo "source_checkpoint=$checkpoint"
  echo "baseline_deferred_checkpoint=$SOURCE_ROOT/$SOURCE_BASELINE_RUN/checkpoints/global_step_128"
  echo "successor_screen=$SUCCESSOR_SCREEN"
  echo "source_screen_terminated=true"
} | tee "$RUN_ROOT/receipts/handoff_complete.txt"
