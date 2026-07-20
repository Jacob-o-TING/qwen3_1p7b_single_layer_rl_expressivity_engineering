#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_WAVE=triglu_priority_6x5090_grpo_step128_to294_then_baseline128_to196_20260715_v1
SOURCE_ROOT="$ROOT/runs/grpo_priority/$SOURCE_WAVE"
SOURCE_SCREEN=qwen_triglu_priority_to294_then_baseline196_20260715_v1
SOURCE_RUN_ID=triglu_6x5090_grpo_untunedbase_resume98_to196_seed20260707_v1
CHECKPOINT_WAVE=triglu_baseline_6x5090_grpo_retro30_60_then_98to196_interleaved_20260714_v1
CHECKPOINT_ROOT="$ROOT/runs/grpo_interleaved/$CHECKPOINT_WAVE"
THIRD_RUN_ID=triglu_6x5090_grpo_resume196_to294_cosine5e7_to5e8_seed20260707_v1
WAVE=baseline_then_oft_fp32_6x5090_grpo_after_triglu196_20260715_v1
RUN_ROOT="$ROOT/runs/grpo_reordered/$WAVE"
SUCCESSOR_SCREEN=qwen_baseline_then_oft_fp32_to196_20260715_v1
CONTROLLER="$ROOT/scripts/run_baseline_then_oft_fp32_after_triglu196_20260715_v1.sh"
STATE="$RUN_ROOT/handoff_state.env"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/receipts"
exec > >(tee -a "$RUN_ROOT/logs/reorder_watcher.log") 2>&1

set_state() {
  printf 'HANDOFF_PHASE=%q\nUPDATED_UNIX=%q\n' "$1" "$(date +%s)" >"$STATE.tmp"
  mv "$STATE.tmp" "$STATE"
}

fail() {
  local reason="$1"
  set_state FAILED
  printf 'HANDOFF_FAILED time=%s reason=%s\n' "$(date -Is)" "$reason" | tee "$RUN_ROOT/HANDOFF_FAILED"
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

rm -f "$RUN_ROOT/HANDOFF_FAILED"
set_state WAITING_FOR_TRIGLU_196_EVAL
echo "REORDER_WATCHER_READY $(date -Is) source_screen=$SOURCE_SCREEN successor_screen=$SUCCESSOR_SCREEN"

while [[ ! -f "$SOURCE_ROOT/evaluations/triglu_step_196/PARALLEL_EVAL_COMPLETE" ]]; do
  [[ ! -f "$SOURCE_ROOT/WAVE_FAILED" ]] || fail source_wave_failed
  if ! screen -ls 2>/dev/null | grep -Fq ".$SOURCE_SCREEN"; then
    fail source_screen_missing_before_triglu196_eval
  fi
  sleep 0.5
done

checkpoint="$CHECKPOINT_ROOT/$SOURCE_RUN_ID/checkpoints/global_step_196"
validate_checkpoint "$checkpoint" || fail triglu_step196_checkpoint_invalid
set_state STOPPING_BEFORE_THIRD_STAGE
echo "TRIGLU_196_EVAL_DURABLE $(date -Is) checkpoint=$checkpoint"

screen -S "$SOURCE_SCREEN" -X quit >/dev/null 2>&1 || true
for _ in $(seq 1 60); do
  if ! screen -ls 2>/dev/null | grep -Fq ".$SOURCE_SCREEN"; then
    break
  fi
  sleep 0.5
done
if screen -ls 2>/dev/null | grep -Fq ".$SOURCE_SCREEN"; then
  fail source_screen_did_not_stop
fi

# The old controller can cross the receipt boundary by a fraction of a second.
# Kill only the explicitly deferred third-stage command if it was spawned; no
# completed TriGLU-196 or baseline checkpoint is touched.
pkill -f "trainer.experiment_name=$THIRD_RUN_ID" >/dev/null 2>&1 || true
pkill -f "$THIRD_RUN_ID.*verl.trainer.main_ppo" >/dev/null 2>&1 || true
ray stop --force >/dev/null 2>&1 || true

third_tracker="$SOURCE_ROOT/$THIRD_RUN_ID/checkpoints/latest_checkpointed_iteration.txt"
third_completed=196
[[ -f "$third_tracker" ]] && third_completed=$(tr -dc '0-9' <"$third_tracker")
(( third_completed <= 196 )) || fail third_stage_update_observed_before_stop
if find "$SOURCE_ROOT/$THIRD_RUN_ID/checkpoints" -maxdepth 1 -type d -name 'global_step_19[7-9]' -print -quit 2>/dev/null | grep -q .; then
  fail third_stage_checkpoint_observed_before_stop
fi

{
  echo "status=STOPPED_BEFORE_TRIGLU_THIRD_STAGE"
  echo "time=$(date -Is)"
  echo "triglu_checkpoint=$checkpoint"
  echo "triglu_eval=$SOURCE_ROOT/evaluations/triglu_step_196/PARALLEL_EVAL_COMPLETE"
  echo "third_stage_completed=$third_completed"
} | tee "$RUN_ROOT/receipts/triglu196_boundary_stop.txt"

set_state WAITING_FOR_SUCCESSOR_CONTROLLER
while [[ ! -x "$CONTROLLER" ]]; do
  sleep 5
done
bash -n "$CONTROLLER" || fail successor_controller_syntax_failed
if screen -ls 2>/dev/null | grep -Fq ".$SUCCESSOR_SCREEN"; then
  fail successor_screen_already_exists
fi

set_state LAUNCHING_SUCCESSOR
screen -dmS "$SUCCESSOR_SCREEN" bash -lc "cd '$ROOT' && exec bash '$CONTROLLER'"
sleep 2
screen -ls 2>/dev/null | grep -Fq ".$SUCCESSOR_SCREEN" || fail successor_screen_launch_failed
set_state COMPLETE
echo "REORDER_HANDOFF_COMPLETE $(date -Is) successor_screen=$SUCCESSOR_SCREEN"
