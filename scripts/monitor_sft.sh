#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$ROOT/logs/current_sft.logpath" ]]; then
  LOG_PATH_FILE="$ROOT/logs/current_sft.logpath"
  RUN_ROOT_FILE="$ROOT/logs/current_sft.runroot"
  SCREEN_FILE="$ROOT/logs/current_sft.screen"
else
  LOG_PATH_FILE="$ROOT/logs/current_sft_benchmark.logpath"
  RUN_ROOT_FILE="$ROOT/logs/current_sft_benchmark.runroot"
  SCREEN_FILE="$ROOT/logs/current_sft_benchmark.screen"
fi

echo "=== Qwen SFT Dashboard $(date -Is) ==="
if [[ -f "$SCREEN_FILE" ]]; then
  session="$(cat "$SCREEN_FILE")"
  if screen -ls | grep -Fq "$session"; then
    echo "Main session: ACTIVE ($session)"
  else
    echo "Main session: NOT ACTIVE ($session)"
  fi
else
  echo "Main session: unknown (no current session file)"
fi

log_path=""
dashboard_complete=0
if [[ -f "$LOG_PATH_FILE" ]]; then
  log_path="$(cat "$LOG_PATH_FILE")"
fi

if [[ -f "$RUN_ROOT_FILE" ]]; then
  run_root="$(cat "$RUN_ROOT_FILE")"
  echo
  dashboard_python="$ROOT/envs/vllm0102_verl061/bin/python"
  if [[ ! -x "$dashboard_python" ]]; then
    dashboard_python="${PYTHON_BIN:-python}"
  fi
  dashboard_output="$(
    PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$dashboard_python" \
      -m qwen_single_layer_rl.sft.status_dashboard --run-root "$run_root" 2>&1 || true
  )"
  printf '%s\n' "$dashboard_output"
  if grep -Fq 'CURRENT PHASE: All variants complete' <<<"$dashboard_output"; then
    dashboard_complete=1
  fi
fi

if [[ -n "$log_path" ]]; then
  echo "Log: $log_path"
fi

echo
echo "HARDWARE"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
  --format='csv,noheader,nounits' | awk -F, '{printf "GPU: %s | util %s%% | memory %s/%s MiB | power %s W | temp %s C\n", $1,$2,$3,$4,$5,$6}' || true
df -h /root/autodl-tmp 2>/dev/null | awk 'NR==2 {printf "Disk: %s used / %s total | %s free (%s used)\n", $3,$2,$4,$5}' || true

echo
echo "HEALTH"
if [[ -n "${log_path:-}" && -f "$log_path" ]]; then
  errors="$(grep -Ei 'Traceback|(^|[^_])Error:|Exception:|OutOfMemory|CUDA error|status=[1-9]' "$log_path" | tail -n 5 || true)"
  if [[ "$dashboard_complete" == "1" ]]; then
    echo "Completion gate passed; no active trainer or evaluator is expected."
    if [[ -n "$errors" ]]; then
      echo "Historical log errors retained (including the intentional durable-pause SIGTERM); not current run health."
    fi
  elif [[ -n "$errors" ]]; then
    echo "$errors"
  else
    echo "No traceback, OOM, CUDA error, or nonzero exit detected."
  fi
fi

if [[ "${SFT_MONITOR_VERBOSE:-0}" == "1" && -n "${log_path:-}" && -f "$log_path" ]]; then
  echo
  echo "VERBOSE RECENT EVENTS"
  grep -E 'SFT_(ORDERED_[A-Z_]+|DIAGNOSTIC_[A-Z_]+|STEP|VALIDATION|CHECKPOINT_SAVED|RUN_END|FINAL_EVAL_(START|END))' \
    "$log_path" | tail -n 40 || true
fi
