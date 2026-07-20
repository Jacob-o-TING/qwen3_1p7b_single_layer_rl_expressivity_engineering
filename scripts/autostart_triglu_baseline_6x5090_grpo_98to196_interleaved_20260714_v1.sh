#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OLD_WAVE=triglu_baseline_6x5090_grpo_20to98_serial_20260712_v1
OLD_ROOT="$ROOT/runs/grpo_serial/$OLD_WAVE"
WAVE=triglu_baseline_6x5090_grpo_retro30_60_then_98to196_interleaved_20260714_v1
RUN_ROOT="$ROOT/runs/grpo_interleaved/$WAVE"
OLD_SCREEN=qwen_triglu_baseline_6x5090_grpo_20260712_v1
CONTROLLER_SCREEN=qwen_grpo_98to196_interleaved_20260714_v1
CONTROLLER="$ROOT/scripts/run_triglu_baseline_6x5090_grpo_retro30_60_then_98to196_interleaved_20260714_v1.sh"
STATE="$RUN_ROOT/autostart_state.env"
LOG="$RUN_ROOT/logs/autostart.log"

mkdir -p "$RUN_ROOT/logs"
exec > >(tee -a "$LOG") 2>&1

set_state() {
  printf 'AUTOSTART_PHASE=%q\nOLD_STEP=%q\nUPDATED_UNIX=%q\n' \
    "$1" "$2" "$(date +%s)" >"$STATE.tmp"
  mv "$STATE.tmp" "$STATE"
}

old_step() {
  local tracker="$OLD_ROOT/baseline_6x5090_grpo_untunedbase_b98_seed20260707_v1/checkpoints/latest_checkpointed_iteration.txt"
  if [[ -f "$tracker" ]]; then
    tr -dc '0-9' <"$tracker"
  else
    echo 0
  fi
}

fail() {
  local reason="$1" step
  step=$(old_step)
  set_state FAILED "$step"
  printf 'AUTOSTART_FAILED reason=%s old_step=%s time=%s\n' "$reason" "$step" "$(date -Is)" \
    | tee "$RUN_ROOT/AUTOSTART_FAILED"
  exit 1
}

rm -f "$RUN_ROOT/AUTOSTART_FAILED"
missing_screen_checks=0
echo "AUTOSTART_WATCH_BEGIN $(date -Is)"
while [[ ! -f "$OLD_ROOT/WAVE_COMPLETE" ]]; do
  step=$(old_step)
  set_state WAITING_OLD_WAVE "$step"
  if screen -ls 2>/dev/null | grep -q "\.${OLD_SCREEN}[[:space:]]"; then
    missing_screen_checks=0
  else
    missing_screen_checks=$((missing_screen_checks + 1))
    if (( missing_screen_checks >= 5 )); then
      fail "old_controller_absent_without_WAVE_COMPLETE"
    fi
  fi
  echo "AUTOSTART_WAIT old_step=$step time=$(date -Is)"
  sleep 60
done

set_state VERIFYING_OLD_WAVE 98
grep -q '^PHASE=COMPLETE$' "$OLD_ROOT/state.env" || fail "old_phase_not_complete"
grep -q '^VARIANT=all$' "$OLD_ROOT/state.env" || fail "old_variant_not_all"
grep -q '^TARGET=98$' "$OLD_ROOT/state.env" || fail "old_target_not_98"
for variant in triglu baseline; do
  checkpoint="$OLD_ROOT/${variant}_6x5090_grpo_untunedbase_b98_seed20260707_v1/checkpoints/global_step_98"
  [[ -s "$checkpoint/data.pt" ]] || fail "missing_${variant}_step98_data_state"
  [[ -s "$checkpoint/actor/fsdp_config.json" ]] || fail "missing_${variant}_step98_actor"
  [[ -f "$OLD_ROOT/evaluations/${variant}_step_98/PARALLEL_EVAL_COMPLETE" ]] \
    || fail "missing_${variant}_step98_eval"
done

for _ in $(seq 1 20); do
  old_live=0
  gpu_live=0
  screen -ls 2>/dev/null | grep -q "\.${OLD_SCREEN}[[:space:]]" && old_live=1
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | grep -Eq '^[[:space:]]*[0-9]+' && gpu_live=1
  if (( old_live == 0 && gpu_live == 0 )); then
    break
  fi
  echo "AUTOSTART_DRAIN_WAIT old_screen=$old_live gpu_processes=$gpu_live time=$(date -Is)"
  sleep 15
done
screen -ls 2>/dev/null | grep -q "\.${OLD_SCREEN}[[:space:]]" && fail "old_controller_did_not_exit"
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -Eq '^[[:space:]]*[0-9]+'; then
  fail "gpu_compute_processes_remain_after_old_wave"
fi

[[ -x "$CONTROLLER" ]] || fail "continuation_controller_missing_or_not_executable"
free=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
(( free >= 100 )) || fail "disk_free_below_100G"
if screen -ls 2>/dev/null | grep -q "\.${CONTROLLER_SCREEN}[[:space:]]"; then
  fail "continuation_controller_screen_already_exists"
fi

set_state LAUNCHING 98
echo "AUTOSTART_LAUNCH controller=$CONTROLLER_SCREEN time=$(date -Is)"
screen -dmS "$CONTROLLER_SCREEN" bash "$CONTROLLER"
sleep 8
if [[ -f "$RUN_ROOT/WAVE_FAILED" ]]; then
  fail "continuation_controller_failed_during_startup"
fi
if ! screen -ls 2>/dev/null | grep -q "\.${CONTROLLER_SCREEN}[[:space:]]"; then
  [[ -f "$RUN_ROOT/WAVE_COMPLETE" ]] || fail "continuation_controller_screen_not_running"
fi
set_state LAUNCHED 98
touch "$RUN_ROOT/AUTOSTART_LAUNCHED"
echo "AUTOSTART_LAUNCHED controller=$CONTROLLER_SCREEN time=$(date -Is)"
