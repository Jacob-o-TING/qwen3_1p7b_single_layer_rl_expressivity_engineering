#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:?Usage: launch_sft_single_node.sh CONFIG [extra trainer args...]}"
shift
PYTHON_BIN="${PYTHON_BIN:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"

if [[ "$(basename "$CONFIG")" == "layer10_whole_layer_oft_sft.yaml" \
  && "${SFT_RUN_TRIGLU_GREEDY_BEFORE_OFT:-1}" == "1" ]]; then
  triglu_run_root="${SFT_OUTPUT_ROOT:?SFT_OUTPUT_ROOT is required}/layer10_whole_layer_triglu_side_ffn"
  triglu_eval_root="${SFT_EVAL_ROOT:-$SFT_OUTPUT_ROOT/evaluations}/layer10_whole_layer_triglu_side_ffn"
  triglu_checkpoint="$(
    "$PYTHON_BIN" -m qwen_single_layer_rl.sft.handoff \
      resolve-checkpoint --run-dir "$triglu_run_root"
  )"
  "$PYTHON_BIN" -m qwen_single_layer_rl.sft.handoff \
    eval-status --eval-dir "$triglu_eval_root" --checkpoint-dir "$triglu_checkpoint" >/dev/null
  bash scripts/launch_greedy_amc_controls_before_triglu.sh \
    post-variant \
    layer10_whole_layer_triglu_side_ffn \
    amc_greedy_modal_path_triglu_sft50k_v1 \
    "$triglu_run_root" \
    "$ROOT/configs/sft/layer10_whole_layer_triglu_side_ffn_sft.yaml" \
    checkpoint \
    greedy
fi

if [[ "$(basename "$CONFIG")" == "layer10_whole_layer_triglu_side_ffn_sft.yaml" \
  && "${SFT_RUN_GREEDY_AMC_CONTROLS_BEFORE_TRIGLU:-${SFT_RUN_SHS_GREEDY_AMC_BEFORE_TRIGLU:-1}}" == "1" ]]; then
  bash scripts/launch_greedy_amc_controls_before_triglu.sh
fi

echo "SFT_LAUNCH_START $(date -Is)"
echo "CONFIG=$CONFIG"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "PYTHON_BIN=$PYTHON_BIN"
"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$NPROC_PER_NODE" \
  -m qwen_single_layer_rl.sft.trainer \
  --config "$CONFIG" \
  "$@"
status=$?
echo "SFT_LAUNCH_END $(date -Is) status=$status"
exit "$status"
