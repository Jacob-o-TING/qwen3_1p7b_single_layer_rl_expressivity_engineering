#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="$ROOT/envs/vllm0102_verl061/bin/python"
VERL_ROOT="/root/autodl-tmp/verl-v0.6.1-qwenpatch"
CONFIG="$ROOT/configs/runtime/shs_grpo_replica_shard_20260712_v2_realverl.yaml"
MODEL="$ROOT/runs/runtime_smokes/shs_fullmodel_kernel_parity_20260712_v1/deployment_export"
CHECKPOINT="$ROOT/runs/sft_ordered_20260711_sft50k_v1/layer10_whole_layer_shs/checkpoints/step_00003916"
DATA="$ROOT/data/numina_math_cot_50k_decontam_v3_verl"
RUN_ROOT="$ROOT/runs/runtime_smokes"
OUT="$RUN_ROOT/shs_grpo_replica_shard_20260712_v2_realverl"
RUN_MODE="${RUN_MODE:-train}"
if [[ "$RUN_MODE" != "train" && "$RUN_MODE" != "resume_check" ]]; then
  echo "Unsupported RUN_MODE: $RUN_MODE" >&2
  exit 16
fi
LOG="$OUT/run.log"
RECEIPT="$OUT/completion_receipt.json"
if [[ "$RUN_MODE" == "resume_check" ]]; then
  LOG="$OUT/resume_check.log"
  RECEIPT="$OUT/resume_receipt.json"
fi

mkdir -p "$OUT"
if [[ -e "$RECEIPT" ]]; then
  echo "Refusing to overwrite completed mode receipt: $RECEIPT" >&2
  exit 17
fi

export PYTHONPATH="$ROOT/src:$VERL_ROOT"
export PYTHON_BIN
export VLLM_USE_V1=1

if [[ "$RUN_MODE" == "resume_check" ]]; then
  TARGET_STEPS="$(
    "$PYTHON_BIN" - "$CONFIG" <<'PY'
import sys
from pathlib import Path
from qwen_single_layer_rl.config import load_config

config = load_config(Path(sys.argv[1]))
print(int(config["grpo"]["total_training_steps"]))
PY
  )"
  set +e
  "$PYTHON_BIN" -m qwen_single_layer_rl.training.resume_gate \
    --checkpoint-root "$OUT/checkpoints" \
    --target-steps "$TARGET_STEPS" \
    --receipt "$RECEIPT" >"$OUT/resume_gate.json"
  RESUME_GATE_STATUS=$?
  set -e
  if [[ "$RESUME_GATE_STATUS" -eq 0 ]]; then
    echo "RESUME_ALREADY_COMPLETE target_steps=$TARGET_STEPS receipt=$RECEIPT"
    exit 0
  fi
  if [[ "$RESUME_GATE_STATUS" -ne 3 ]]; then
    exit "$RESUME_GATE_STATUS"
  fi
fi

START_EPOCH="$(date +%s)"
START_ISO="$(date -Is)"
MANIFEST="$OUT/command_manifest.json"
if [[ -e "$MANIFEST" ]]; then
  MANIFEST="$OUT/command_manifest_resume_${START_EPOCH}.json"
fi
"$PYTHON_BIN" -m qwen_single_layer_rl.training.verl_command \
  --config "$CONFIG" \
  --project-root "$ROOT" \
  --verl-root "$VERL_ROOT" \
  --model-path "$MODEL" \
  --data-dir "$DATA" \
  --checkpoint-dir "$CHECKPOINT" \
  --run-root "$RUN_ROOT" \
  --manifest-out "$MANIFEST" >/dev/null

SHELL_CMD="$(
  "$PYTHON_BIN" -m qwen_single_layer_rl.training.verl_command \
    --config "$CONFIG" \
    --project-root "$ROOT" \
    --verl-root "$VERL_ROOT" \
    --model-path "$MODEL" \
    --data-dir "$DATA" \
    --checkpoint-dir "$CHECKPOINT" \
    --run-root "$RUN_ROOT" \
    --print-shell
)"

set +e
{
  echo "RUN_START $START_ISO"
  echo "RUN_ID shs_grpo_replica_shard_20260712_v2_realverl"
  echo "RUN_MODE $RUN_MODE"
  echo "SHAPE prompts=128 group_size=4 expected_rollouts=512 tp=1 replicas=1"
  nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
  bash -lc "$SHELL_CMD"
} 2>&1 | tee -a "$LOG"
STATUS=${PIPESTATUS[0]}
set -e
END_EPOCH="$(date +%s)"
END_ISO="$(date -Is)"

RECEIPT_OUT="$OUT/${RUN_MODE}_attempt_exit_receipt_${END_EPOCH}.json"
if [[ "$STATUS" -eq 0 ]]; then
  RECEIPT_OUT="$RECEIPT"
fi
"$PYTHON_BIN" - "$RECEIPT_OUT" "$STATUS" "$START_ISO" "$END_ISO" "$((END_EPOCH - START_EPOCH))" "$RUN_MODE" <<'PY'
import json
import sys
from pathlib import Path

path, status, started, ended, wall, run_mode = sys.argv[1:]
payload = {
    "run_id": "shs_grpo_replica_shard_20260712_v2_realverl",
    "exit_status": int(status),
    "started_at": started,
    "ended_at": ended,
    "shell_wall_seconds": int(wall),
    "run_mode": run_mode,
    "claims": {"production_candidate": False, "production_ready": False},
}
Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "RUN_END $END_ISO status=$STATUS wall_seconds=$((END_EPOCH - START_EPOCH))"
exit "$STATUS"
