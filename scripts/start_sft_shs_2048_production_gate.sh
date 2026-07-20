#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SESSION="${SFT_BENCHMARK_SESSION:-qwen_sft_shs_2048_gate}"
LOG="$ROOT/logs/sft_shs_2048_production_gate_$STAMP.log"

mkdir -p "$ROOT/logs"
screen -S "$SESSION" -X quit >/dev/null 2>&1 || true
printf '%s\n' "$SESSION" > "$ROOT/logs/current_sft_benchmark.screen"
printf '%s\n' "$LOG" > "$ROOT/logs/current_sft_benchmark.logpath"

screen -L -Logfile "$LOG" -DmS "$SESSION" env \
  RUN_STAMP="$STAMP" \
  PYTHON_BIN="${PYTHON_BIN:-python}" \
  MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}" \
  MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}" \
  GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}" \
  WARMUP_STEPS="${WARMUP_STEPS:-2}" \
  TIMED_STEPS="${TIMED_STEPS:-5}" \
  bash "$ROOT/scripts/launch_sft_shs_2048_production_gate.sh"

echo "SFT_SHS_2048_GATE_SCREEN_STARTED session=$SESSION"
echo "SFT_SHS_2048_GATE_LOG=$LOG"
echo "Monitor with: bash $ROOT/scripts/monitor_sft.sh"
