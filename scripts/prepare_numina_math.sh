#!/usr/bin/env bash
set -euo pipefail

SOURCE=""
SOURCE_HUB="huggingface"
OUT_DIR="data/numina_math_cot_50k"
INPUT_JSONL=""
BENCHMARK_HASHES=""
BENCHMARK_PROBLEMS=""
TARGET_SIZE=50000
SEED=20260707
STREAMING_FLAG="--streaming"
WRITE_PARQUET=true
VAL_SIZE=100
MAX_SOURCE_RECORDS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2 ;;
    --source-hub) SOURCE_HUB="$2"; shift 2 ;;
    --input-jsonl) INPUT_JSONL="$2"; shift 2 ;;
    --benchmark-hashes) BENCHMARK_HASHES="$2"; shift 2 ;;
    --benchmark-problems) BENCHMARK_PROBLEMS="$2"; shift 2 ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    --target-size) TARGET_SIZE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --no-streaming) STREAMING_FLAG="--no-streaming"; shift ;;
    --no-parquet) WRITE_PARQUET=false; shift ;;
    --val-size) VAL_SIZE="$2"; shift 2 ;;
    --max-source-records) MAX_SOURCE_RECORDS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

cmd=(python -m qwen_single_layer_rl.data.prep_numina
  --out-dir "$OUT_DIR"
  --target-size "$TARGET_SIZE"
  --seed "$SEED"
  --val-size "$VAL_SIZE")

if [[ -n "$INPUT_JSONL" ]]; then
  cmd+=(--input-jsonl "$INPUT_JSONL")
else
  if [[ "$SOURCE_HUB" == "modelscope" ]]; then
    cmd+=(--modelscope-dataset "${SOURCE:-AI-MO/NuminaMath-CoT}" --split train "$STREAMING_FLAG")
  else
    cmd+=(--hf-dataset "${SOURCE:-AI-MO/NuminaMath-CoT}" --split train "$STREAMING_FLAG")
  fi
fi

if [[ -n "$BENCHMARK_HASHES" ]]; then
  cmd+=(--benchmark-hashes "$BENCHMARK_HASHES")
fi
if [[ -n "$BENCHMARK_PROBLEMS" ]]; then
  cmd+=(--benchmark-problems "$BENCHMARK_PROBLEMS")
fi
if [[ "$WRITE_PARQUET" == true ]]; then
  cmd+=(--write-parquet)
fi
if [[ -n "$MAX_SOURCE_RECORDS" ]]; then
  cmd+=(--max-source-records "$MAX_SOURCE_RECORDS")
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
