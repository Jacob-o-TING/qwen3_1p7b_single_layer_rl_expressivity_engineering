#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="$ROOT/envs/vllm0102_verl061/bin/python"
VERL_ROOT="/root/autodl-tmp/verl-v0.6.1-qwenpatch"
CONFIG="$ROOT/configs/runtime/shs_2x5090_actor_rollout_weight_sync_20260712_v1.yaml"
MODEL="$ROOT/runs/rtx5090_pair_bringup_20260712_v1/gate_b/fullmodel_parity/deployment_export"
CHECKPOINT="$ROOT/runs/sft_ordered_20260711_sft50k_v1/layer10_whole_layer_shs/checkpoints/step_00003916"
DATA="$ROOT/data/numina_math_cot_50k_decontam_v3_verl"
RUN_ID="shs_2x5090_actor_rollout_weight_sync_20260712_v1"
OUT="$ROOT/runs/$RUN_ID"
RUN_MODE="${RUN_MODE:-train}"

if [[ "$RUN_MODE" != "train" && "$RUN_MODE" != "resume_check" ]]; then
  echo "Unsupported RUN_MODE: $RUN_MODE" >&2
  exit 16
fi

mkdir -p "$OUT"
LOG="$OUT/run.log"
RECEIPT="$OUT/completion_receipt.json"
if [[ "$RUN_MODE" == "resume_check" ]]; then
  LOG="$OUT/resume_check.log"
  RECEIPT="$OUT/resume_receipt.json"
fi
if [[ -e "$RECEIPT" ]]; then
  echo "Refusing to overwrite completed receipt: $RECEIPT" >&2
  exit 17
fi

export PYTHONPATH="$ROOT/src:$VERL_ROOT"
export PYTHON_BIN
export VLLM_USE_V1=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false

if [[ "$RUN_MODE" == "resume_check" ]]; then
  TARGET_STEPS="$(
    "$PYTHON_BIN" - "$CONFIG" <<'PY'
import sys
from pathlib import Path
from qwen_single_layer_rl.config import load_config

print(int(load_config(Path(sys.argv[1]))["grpo"]["total_training_steps"]))
PY
  )"
  set +e
  "$PYTHON_BIN" -m qwen_single_layer_rl.training.resume_gate \
    --checkpoint-root "$OUT/checkpoints" \
    --target-steps "$TARGET_STEPS" \
    --receipt "$RECEIPT" >"$OUT/resume_gate.json"
  status=$?
  set -e
  if [[ "$status" -eq 0 ]]; then
    echo "RESUME_ALREADY_COMPLETE target_steps=$TARGET_STEPS"
    exit 0
  fi
  if [[ "$status" -ne 3 ]]; then
    exit "$status"
  fi
fi

for path in "$PYTHON_BIN" "$CONFIG" "$MODEL/config.json" "$CHECKPOINT/trainable_state.pt" \
  "$DATA/train.parquet" "$DATA/val.parquet"; do
  [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 18; }
done

manifest="$OUT/command_manifest_${RUN_MODE}.json"
"$PYTHON_BIN" -m qwen_single_layer_rl.training.verl_command \
  --config "$CONFIG" --project-root "$ROOT" --verl-root "$VERL_ROOT" \
  --model-path "$MODEL" --data-dir "$DATA" --checkpoint-dir "$CHECKPOINT" \
  --run-root "$ROOT/runs" --manifest-out "$manifest" >/dev/null

shell_cmd="$(
  "$PYTHON_BIN" -m qwen_single_layer_rl.training.verl_command \
    --config "$CONFIG" --project-root "$ROOT" --verl-root "$VERL_ROOT" \
    --model-path "$MODEL" --data-dir "$DATA" --checkpoint-dir "$CHECKPOINT" \
    --run-root "$ROOT/runs" --print-shell
)"

started_epoch="$(date +%s)"
started_iso="$(date -Is)"
set +e
{
  echo "RUN_START $started_iso"
  echo "RUN_ID $RUN_ID"
  echo "RUN_MODE $RUN_MODE"
  echo "SHAPE prompts=2 group_size=4 steps=11 nproc=2 tp=1 replicas=2"
  nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.total --format=csv,noheader
  bash -lc "$shell_cmd"
} 2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
set -e
ended_epoch="$(date +%s)"
ended_iso="$(date -Is)"

receipt_out="$OUT/${RUN_MODE}_attempt_exit_receipt_${ended_epoch}.json"
if [[ "$status" -eq 0 ]]; then
  receipt_out="$RECEIPT"
fi
"$PYTHON_BIN" - "$receipt_out" "$status" "$started_iso" "$ended_iso" \
  "$((ended_epoch - started_epoch))" "$RUN_MODE" <<'PY'
import json
import sys
from pathlib import Path

path, status, started, ended, wall, run_mode = sys.argv[1:]
payload = {
    "run_id": "shs_2x5090_actor_rollout_weight_sync_20260712_v1",
    "exit_status": int(status),
    "started_at": started,
    "ended_at": ended,
    "shell_wall_seconds": int(wall),
    "run_mode": run_mode,
    "claims": {"production_grpo": False, "quality_result": False},
}
Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "RUN_END $ended_iso status=$status wall_seconds=$((ended_epoch - started_epoch))"
exit "$status"
