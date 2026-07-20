#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${SFT_BENCHMARK_SESSION:-qwen_sft_compile_benchmark}"
STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG="$ROOT/logs/sft_compile_short_benchmark_$STAMP.log"

mkdir -p "$ROOT/logs"
screen -S "$SESSION" -X quit >/dev/null 2>&1 || true
printf '%s\n' "$SESSION" > "$ROOT/logs/current_sft_benchmark.screen"
printf '%s\n' "$LOG" > "$ROOT/logs/current_sft_benchmark.logpath"

screen -L -Logfile "$LOG" -DmS "$SESSION" env \
  RUN_STAMP="$STAMP" \
  PYTHON_BIN="${PYTHON_BIN:-python}" \
  MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}" \
  MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}" \
  WARMUP_STEPS="${WARMUP_STEPS:-5}" \
  TIMED_STEPS="${TIMED_STEPS:-20}" \
  bash "$ROOT/scripts/launch_sft_compile_short_benchmark.sh"

echo "SFT_BENCHMARK_SCREEN_STARTED session=$SESSION"
echo "SFT_BENCHMARK_LOG=$LOG"
echo "Monitor with: bash $ROOT/scripts/monitor_sft.sh"
