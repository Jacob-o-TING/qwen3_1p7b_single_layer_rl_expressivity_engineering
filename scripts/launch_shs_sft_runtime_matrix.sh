#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="shs_sft_runtime_matrix_20260712_v1"
OUT="${OUTPUT_ROOT:-$ROOT/runs/runtime_smokes/$RUN_ID}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
EXPECTED_GPU_COUNT="${EXPECTED_GPU_COUNT:-$NPROC_PER_NODE}"
CHECKPOINT="${CHECKPOINT:-$ROOT/runs/sft_ordered_20260711_sft50k_v1/layer10_whole_layer_shs/checkpoints/step_00003916}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
TIMED_STEPS="${TIMED_STEPS:-50}"

VISIBLE_GPU_COUNT="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)"
if [[ "$VISIBLE_GPU_COUNT" -ne "$EXPECTED_GPU_COUNT" ]]; then
  echo "TOPOLOGY_PREFLIGHT_FAILED visible=$VISIBLE_GPU_COUNT expected=$EXPECTED_GPU_COUNT" >&2
  exit 2
fi
if [[ "$NPROC_PER_NODE" -ne 1 ]]; then
  echo "TRACK_B_CONTRACT_FAILED nproc_per_node=$NPROC_PER_NODE expected_single_gpu=1" >&2
  exit 2
fi

mkdir -p "$OUT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MATRIX_OUT="$OUT"
export MATRIX_RUN_ID="$RUN_ID"
export MATRIX_CHECKPOINT="$CHECKPOINT"
export MATRIX_VISIBLE_GPU_COUNT="$VISIBLE_GPU_COUNT"
export MATRIX_NPROC_PER_NODE="$NPROC_PER_NODE"
export MATRIX_WARMUP_STEPS="$WARMUP_STEPS"
export MATRIX_TIMED_STEPS="$TIMED_STEPS"
"$PYTHON_BIN" - <<'PY'
import hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

out = Path(os.environ["MATRIX_OUT"])
checkpoint = Path(os.environ["MATRIX_CHECKPOINT"])
payload = {
    "run_id": os.environ["MATRIX_RUN_ID"],
    "status": "preregistered",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "cells": ["reference_eager", "reference_compile", "triton_recompute_eager", "triton_recompute_compile"],
    "warmup_steps": int(os.environ["MATRIX_WARMUP_STEPS"]),
    "timed_steps": int(os.environ["MATRIX_TIMED_STEPS"]),
    "max_seq_length": 2048,
    "micro_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "checkpoint_trainable_sha256": sha256(checkpoint / "trainable_state.pt"),
    "resolved_topology": {
        "visible_gpu_count": int(os.environ["MATRIX_VISIBLE_GPU_COUNT"]),
        "nproc_per_node": int(os.environ["MATRIX_NPROC_PER_NODE"]),
        "tensor_parallel_size": None,
        "effective_packed_batch_size": 8,
    },
    "claims": {"production_candidate": False, "production_ready": False},
}
(out / "preregistered_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

run_cell() {
  local cell="$1"
  local backend="$2"
  local compile_mode="$3"
  echo "MATRIX_CELL_START $(date -Is) cell=$cell backend=$backend compile=$compile_mode"
  TORCHINDUCTOR_CACHE_DIR="$OUT/inductor_cache/$cell" "$PYTHON_BIN" -m torch.distributed.run \
    --standalone --nproc_per_node="$NPROC_PER_NODE" \
    -m qwen_single_layer_rl.sft.trainer \
    --config configs/sft/layer10_whole_layer_shs_sft.yaml \
    --output-dir "$OUT/$cell" \
    --run-id "$cell" \
    --compile-mode "$compile_mode" \
    --max-seq-length 2048 \
    --micro-batch-size 1 \
    --gradient-accumulation-steps 8 \
    --epochs 1 --benchmark --no-resume \
    --warmup-steps "$WARMUP_STEPS" --timed-steps "$TIMED_STEPS" \
    --initial-checkpoint-dir "$CHECKPOINT" \
    --shs-mul-backend "$backend"
  echo "MATRIX_CELL_END $(date -Is) cell=$cell status=0"
}

cd "$ROOT"
run_cell reference_eager reference eager
run_cell reference_compile reference default
run_cell triton_recompute_eager triton_reference_recompute eager
run_cell triton_recompute_compile triton_reference_recompute default
"$PYTHON_BIN" scripts/summarize_shs_sft_runtime_matrix.py --root "$OUT"
echo "MATRIX_END $(date -Is) run_id=$RUN_ID"
