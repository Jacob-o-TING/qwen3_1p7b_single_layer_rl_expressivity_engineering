#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING_PYTHON_BIN="${TRAINING_PYTHON_BIN:-$ROOT/envs/vllm0102_verl061/bin/python}"
EVAL_PYTHON_BIN="${EVAL_PYTHON_BIN:-$ROOT/envs/evalscope181/bin/python}"
MODEL_PATH="${MODEL_PATH:-$ROOT/models/Qwen3-1.7B-Base}"
if [[ ! -d "$MODEL_PATH" && -d /root/autodl-tmp/qwen3_single_layer_rl/models/Qwen3-1.7B-Base ]]; then
  MODEL_PATH=/root/autodl-tmp/qwen3_single_layer_rl/models/Qwen3-1.7B-Base
fi
RUN_ROOT="${SFT_OUTPUT_ROOT:?SFT_OUTPUT_ROOT is required for the ordered diagnostic hook}"
BATCH_SIZE="${SFT_GREEDY_AMC_BATCH_SIZE:-16}"

TRAINING_SITE_PACKAGES="$($TRAINING_PYTHON_BIN -c 'import site; print(site.getsitepackages()[0])')"
export PYTHONPATH="$ROOT/src:$TRAINING_SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

run_control() {
  local variant="$1"
  local diagnostic_id="$2"
  local run_dir="$3"
  local config="$4"
  local source_mode="${5:-checkpoint}"
  local decode_mode="${6:-greedy}"
  local output_dir="$RUN_ROOT/diagnostics/$diagnostic_id"
  local preflight_dir="${output_dir}_preflight_bs16"
  local receipt="$output_dir/diagnostic_complete.json"
  local checkpoint_dir=""
  local -a model_source_args
  local -a preflight_args
  local -a production_args
  local phase
  local expected_rows
  local do_sample
  local repeats
  local temperature

  if [[ -s "$receipt" ]]; then
    echo "SFT_DIAGNOSTIC_SKIP $(date -Is) id=$diagnostic_id variant=$variant reason=receipt_exists receipt=$receipt"
    return 0
  fi

  if [[ "$source_mode" == "base" ]]; then
    model_source_args=(--base-model-only)
  else
    checkpoint_dir="$(
      "$TRAINING_PYTHON_BIN" -m qwen_single_layer_rl.sft.handoff \
        resolve-checkpoint --run-dir "$run_dir"
    )"
    model_source_args=(--checkpoint-dir "$checkpoint_dir")
  fi

  if [[ "$decode_mode" == "sampled_avg32" ]]; then
    phase="amc_average_at_32"
    expected_rows=1280
    do_sample=true
    repeats=32
    temperature=1.0
    preflight_args=(
      --amc-only --amc-repeats 2 --limit 8 --max-tokens 64
      --amc-temperature 1.0 --amc-top-p 1.0
    )
    production_args=(
      --amc-only --amc-repeats 32 --max-tokens 3072
      --amc-temperature 1.0 --amc-top-p 1.0
    )
  elif [[ "$decode_mode" == "greedy" ]]; then
    phase="main"
    expected_rows=40
    do_sample=false
    repeats=1
    temperature=0.0
    preflight_args=(
      --datasets paper_amc23 --amc-repeats 0 --limit 16 --max-tokens 64
    )
    production_args=(
      --datasets paper_amc23 --amc-repeats 0 --max-tokens 3072
    )
  else
    echo "Unsupported decode mode: $decode_mode" >&2
    return 2
  fi

  echo "SFT_DIAGNOSTIC_START $(date -Is) id=$diagnostic_id variant=$variant source_mode=$source_mode decode_mode=$decode_mode checkpoint=${checkpoint_dir:-none} batch_size=$BATCH_SIZE"
  rm -rf "$preflight_dir"
  "$EVAL_PYTHON_BIN" -m qwen_single_layer_rl.eval.run_evalscope \
    --config "$config" \
    "${model_source_args[@]}" \
    --model-path "$MODEL_PATH" \
    --work-dir "$preflight_dir" \
    "${preflight_args[@]}" \
    --eval-batch-size "$BATCH_SIZE"
  echo "SFT_DIAGNOSTIC_PREFLIGHT_END $(date -Is) id=$diagnostic_id variant=$variant"

  rm -rf "$output_dir"
  "$EVAL_PYTHON_BIN" -m qwen_single_layer_rl.eval.run_evalscope \
    --config "$config" \
    "${model_source_args[@]}" \
    --model-path "$MODEL_PATH" \
    --work-dir "$output_dir" \
    "${production_args[@]}" \
    --eval-batch-size "$BATCH_SIZE"

  export VARIANT="$variant"
  export DIAGNOSTIC_ID="$diagnostic_id"
  export OUTPUT_DIR="$output_dir"
  export CHECKPOINT_DIR="$checkpoint_dir"
  export BASE_MODEL_ONLY="$([[ "$source_mode" == "base" ]] && echo 1 || echo 0)"
  export PHASE="$phase"
  export EXPECTED_ROWS="$expected_rows"
  export DO_SAMPLE="$do_sample"
  export REPEATS="$repeats"
  export TEMPERATURE="$temperature"
  export CONFIG_PATH="$config"
  export MODEL_PATH BATCH_SIZE
  export RECEIPT="$receipt"
  "$TRAINING_PYTHON_BIN" - <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


output_dir = Path(os.environ["OUTPUT_DIR"]).resolve()
phase = os.environ["PHASE"]
reports = list(output_dir.glob(f"{phase}/*/reports/*/paper_amc23.json"))
predictions = list(output_dir.glob(f"{phase}/*/predictions/*/paper_amc23_main.jsonl"))
reviews = list(output_dir.glob(f"{phase}/*/reviews/*/paper_amc23_main.jsonl"))
if len(reports) != 1 or len(predictions) != 1 or len(reviews) != 1:
    raise SystemExit(
        f"Expected one report/prediction/review, found {len(reports)}/{len(predictions)}/{len(reviews)}"
    )
review_rows = sum(1 for line in reviews[0].open(encoding="utf-8") if line.strip())
expected_rows = int(os.environ["EXPECTED_ROWS"])
if review_rows != expected_rows:
    raise SystemExit(f"Expected {expected_rows} AMC reviews, found {review_rows}")
receipt = {
    "schema_version": 1,
    "diagnostic_id": os.environ["DIAGNOSTIC_ID"],
    "variant": os.environ["VARIANT"],
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "checkpoint_dir": (
        None
        if os.environ["BASE_MODEL_ONLY"] == "1"
        else str(Path(os.environ["CHECKPOINT_DIR"]).resolve())
    ),
    "base_model_only": os.environ["BASE_MODEL_ONLY"] == "1",
    "config_path": str(Path(os.environ["CONFIG_PATH"]).resolve()),
    "model_path": str(Path(os.environ["MODEL_PATH"]).resolve()),
    "decode": {
        "do_sample": os.environ["DO_SAMPLE"] == "true",
        "repeats": int(os.environ["REPEATS"]),
        "temperature": float(os.environ["TEMPERATURE"]),
        "max_tokens": 3072,
    },
    "eval_batch_size": int(os.environ["BATCH_SIZE"]),
    "review_rows": review_rows,
    "report": {"path": str(reports[0].resolve()), "sha256": sha256(reports[0])},
    "predictions": {
        "path": str(predictions[0].resolve()),
        "sha256": sha256(predictions[0]),
    },
    "reviews": {"path": str(reviews[0].resolve()), "sha256": sha256(reviews[0])},
}
receipt_path = Path(os.environ["RECEIPT"])
receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
print(receipt_path)
PY

  echo "SFT_DIAGNOSTIC_END $(date -Is) id=$diagnostic_id variant=$variant receipt=$receipt"
}

mode="${1:-pre-triglu}"
if [[ "$mode" == "post-variant" ]]; then
  shift
  run_control "$@"
  exit 0
fi
if [[ "$mode" != "pre-triglu" ]]; then
  echo "Unsupported control set: $mode" >&2
  exit 2
fi

run_control \
  "shs" \
  "amc_greedy_modal_path_shs_sft50k_v1" \
  "$RUN_ROOT/layer10_whole_layer_shs" \
  "$ROOT/configs/sft/layer10_whole_layer_shs_sft.yaml" \
  "checkpoint" \
  "greedy"

run_control \
  "whole_layer_baseline" \
  "amc_greedy_modal_path_baseline_sft50k_v1" \
  "$RUN_ROOT/layer10_whole_layer_baseline" \
  "$ROOT/configs/sft/layer10_whole_layer_baseline_sft.yaml" \
  "checkpoint" \
  "greedy"

run_control \
  "untuned_qwen3_1p7b_base" \
  "amc_greedy_modal_path_untuned_qwen3_1p7b_base_v1" \
  "" \
  "$ROOT/configs/sft/layer10_whole_layer_baseline_sft.yaml" \
  "base" \
  "greedy"

run_control \
  "untuned_qwen3_1p7b_base" \
  "amc_average_at_32_untuned_qwen3_1p7b_base_v1" \
  "" \
  "$ROOT/configs/sft/layer10_whole_layer_baseline_sft.yaml" \
  "base" \
  "sampled_avg32"
