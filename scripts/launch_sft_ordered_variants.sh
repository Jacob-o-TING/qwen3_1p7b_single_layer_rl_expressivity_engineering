#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TRAINING_PYTHON_BIN="${TRAINING_PYTHON_BIN:-${PYTHON_BIN:-python}}"
EVAL_PYTHON_BIN="${EVAL_PYTHON_BIN:-python}"
EVAL_ROOT="${SFT_EVAL_ROOT:-${SFT_OUTPUT_ROOT:-$ROOT/runs}/evaluations}"
AMC_REPEATS="${SFT_AMC_REPEATS:-32}"
EVAL_MAX_TOKENS="${SFT_EVAL_MAX_TOKENS:-3072}"
EVAL_SEED="${SFT_EVAL_SEED:-20260707}"
EVAL_BATCH_SIZE="${SFT_EVAL_BATCH_SIZE:-8}"
mkdir -p "$EVAL_ROOT"

CONFIGS=(
  "configs/sft/layer10_whole_layer_shs_sft.yaml"
  "configs/sft/layer10_whole_layer_baseline_sft.yaml"
  "configs/sft/layer10_whole_layer_triglu_side_ffn_sft.yaml"
  "configs/sft/layer10_whole_layer_oft_sft.yaml"
)

for config in "${CONFIGS[@]}"; do
  variant="$(basename "$config" _sft.yaml)"
  output_dir="${SFT_OUTPUT_ROOT:-$ROOT/runs}/$variant"
  eval_dir="$EVAL_ROOT/$variant"
  echo "SFT_ORDERED_VARIANT_START $(date -Is) variant=$variant config=$config output_dir=$output_dir"
  checkpoint_dir=""
  if checkpoint_dir="$(
    PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$TRAINING_PYTHON_BIN" \
      -m qwen_single_layer_rl.sft.handoff resolve-checkpoint --run-dir "$output_dir" 2>/dev/null
  )"; then
    echo "SFT_ORDERED_TRAINING_SKIP $(date -Is) variant=$variant reason=verified_final_checkpoint checkpoint=$checkpoint_dir"
  else
    bash scripts/launch_sft_single_node.sh "$config" --output-dir "$output_dir" "$@"
    checkpoint_dir="$(
      PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$TRAINING_PYTHON_BIN" \
        -m qwen_single_layer_rl.sft.handoff resolve-checkpoint --run-dir "$output_dir"
    )"
    echo "SFT_ORDERED_TRAINING_COMPLETE $(date -Is) variant=$variant checkpoint=$checkpoint_dir"
  fi

  if PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$TRAINING_PYTHON_BIN" \
    -m qwen_single_layer_rl.sft.handoff eval-status \
    --eval-dir "$eval_dir" --checkpoint-dir "$checkpoint_dir" >/dev/null 2>&1; then
    echo "SFT_ORDERED_EVAL_SKIP $(date -Is) variant=$variant reason=verified_receipt"
  else
    eval_args=(
      --amc-repeats "$AMC_REPEATS"
      --max-tokens "$EVAL_MAX_TOKENS"
      --amc-temperature 1.0
      --amc-top-p 1.0
      --seed "$EVAL_SEED"
      --eval-batch-size "$EVAL_BATCH_SIZE"
    )
    if [[ -n "${SFT_EVAL_LIMIT:-}" ]]; then
      eval_args+=(--limit "$SFT_EVAL_LIMIT")
    fi
    if main_cache="$(
      PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$TRAINING_PYTHON_BIN" \
        -m qwen_single_layer_rl.sft.handoff resolve-eval-cache \
        --eval-dir "$eval_dir" --phase main 2>/dev/null
    )"; then
      eval_args+=(--main-use-cache "$main_cache")
      echo "SFT_ORDERED_EVAL_CACHE $(date -Is) variant=$variant phase=main path=$main_cache"
    fi
    if amc_cache="$(
      PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$TRAINING_PYTHON_BIN" \
        -m qwen_single_layer_rl.sft.handoff resolve-eval-cache \
        --eval-dir "$eval_dir" --phase amc 2>/dev/null
    )"; then
      eval_args+=(--amc-use-cache "$amc_cache")
      echo "SFT_ORDERED_EVAL_CACHE $(date -Is) variant=$variant phase=amc path=$amc_cache"
    fi
    echo "SFT_ORDERED_EVAL_START $(date -Is) variant=$variant work_dir=$eval_dir"
    EVAL_PYTHON_BIN="$EVAL_PYTHON_BIN" TRAINING_PYTHON_BIN="$TRAINING_PYTHON_BIN" \
      SFT_EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
      bash scripts/launch_sft_final_eval.sh \
      "$config" "$checkpoint_dir" "$eval_dir" "${eval_args[@]}"
    receipt_args=(
      --eval-dir "$eval_dir"
      --checkpoint-dir "$checkpoint_dir"
      --config "$config"
    )
    if [[ -n "${SFT_EVAL_LIMIT:-}" && "${SFT_ALLOW_LIMITED_EVAL_RECEIPT:-0}" == "1" ]]; then
      receipt_args+=(--allow-limited)
    fi
    PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$TRAINING_PYTHON_BIN" \
      -m qwen_single_layer_rl.sft.handoff record-eval "${receipt_args[@]}"
    echo "SFT_ORDERED_EVAL_END $(date -Is) variant=$variant receipt=$eval_dir/evaluation_complete.json"
  fi

  post_eval_greedy_id=""
  case "$variant" in
    layer10_whole_layer_triglu_side_ffn)
      post_eval_greedy_id="amc_greedy_modal_path_triglu_sft50k_v1"
      ;;
    layer10_whole_layer_oft)
      post_eval_greedy_id="amc_greedy_modal_path_oft_sft50k_v1"
      ;;
  esac
  if [[ -n "$post_eval_greedy_id" ]]; then
    bash scripts/launch_greedy_amc_controls_before_triglu.sh \
      post-variant \
      "$variant" \
      "$post_eval_greedy_id" \
      "$output_dir" \
      "$config" \
      checkpoint \
      greedy
  fi
  echo "SFT_ORDERED_VARIANT_END $(date -Is) variant=$variant config=$config"
done

echo "SFT_ORDERED_RUN_END $(date -Is) variants=${#CONFIGS[@]}"
