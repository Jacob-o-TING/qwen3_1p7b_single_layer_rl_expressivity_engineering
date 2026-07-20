#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REVISION="${QWEN_EVAL_REVISION:-a45202bd16f1ec06f433442dc1152d0074773465}"
MATH500_REVISION="${MATH500_REVISION:-6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SNAPSHOT_DIR="${BENCHMARK_SNAPSHOT_DIR:-$ROOT/data/eval/qwen2p5_math_$REVISION}"
CONTRACT_DIR="${BENCHMARK_CONTRACT_DIR:-$ROOT/data/decontam/qwen_math_eval_$REVISION}"
BASE_URL="https://raw.githubusercontent.com/QwenLM/Qwen2.5-Math/$REVISION/evaluation/data"

download() {
  local relative_path="$1"
  local destination="$SNAPSHOT_DIR/$relative_path"
  mkdir -p "$(dirname "$destination")"
  if [[ -s "$destination" ]]; then
    return
  fi
  curl -fL --retry 8 --retry-delay 2 --continue-at - \
    -o "$destination.part" "$BASE_URL/$relative_path"
  mv "$destination.part" "$destination"
}

mkdir -p "$SNAPSHOT_DIR/math500"
if [[ ! -s "$SNAPSHOT_DIR/math500/test.jsonl" ]]; then
  curl -fL --retry 8 --retry-delay 2 --continue-at - \
    -o "$SNAPSHOT_DIR/math500/test.jsonl.part" \
    "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/$MATH500_REVISION/test.jsonl"
  mv "$SNAPSHOT_DIR/math500/test.jsonl.part" "$SNAPSHOT_DIR/math500/test.jsonl"
fi
download gsm8k/test.jsonl
download olympiadbench/test.jsonl
download amc23/test.jsonl

(
  cd "$SNAPSHOT_DIR"
  sha256sum math500/test.jsonl gsm8k/test.jsonl olympiadbench/test.jsonl amc23/test.jsonl > SHA256SUMS
)

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON_BIN" -m qwen_single_layer_rl.data.build_benchmark_contract \
  --qwen-eval-root "$SNAPSHOT_DIR" \
  --source-revision "$REVISION" \
  --math500-revision "$MATH500_REVISION" \
  --out-dir "$CONTRACT_DIR"

echo "PAPER_BENCHMARK_PREP_END revision=$REVISION"
echo "SNAPSHOT_DIR=$SNAPSHOT_DIR"
echo "CONTRACT_DIR=$CONTRACT_DIR"
