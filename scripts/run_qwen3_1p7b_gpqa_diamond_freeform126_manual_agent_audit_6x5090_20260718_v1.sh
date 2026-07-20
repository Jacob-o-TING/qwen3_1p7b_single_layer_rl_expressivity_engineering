#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL=/root/autodl-tmp/verl-v0.6.1-qwenpatch
PY="$ROOT/envs/vllm0102_verl061/bin/python"
CONFIG="$ROOT/configs/eval/qwen3_1p7b_gpqa_diamond_freeform126_manual_agent_audit_majorsteps_6x5090_20260718_v1.yaml"
RUN_ID=qwen3_1p7b_gpqa_diamond_freeform126_greedy3072_manual_agent_audit_6x5090_steps158_196_226_256_294_20260718_v1
SCREEN_NAME=qwen_gpqa_free126_manualaudit_gen6_20260718_v1
OUT="$ROOT/runs/freeform_eval/$RUN_ID"
SOURCE_ROOT="$ROOT/runs/grpo_interleaved/triglu_baseline_6x5090_grpo_retro30_60_then_98to196_interleaved_20260714_v1"
PRIORITY_ROOT="$ROOT/runs/grpo_priority/triglu_priority_6x5090_grpo_step128_to294_then_baseline128_to196_20260715_v1"
OTHER_EVAL_OUT="$ROOT/runs/ood_eval/qwen3_1p7b_other_eval_6x5090_triglu_baseline_steps158_196_226_256_294_20260718_v1"
DATA_ROOT="$ROOT/data/eval/gpqa_diamond_freeform126"
LEDGER="$DATA_ROOT/gpqa_diamond_freeform126.ledger.jsonl"
DATA_MANIFEST="$DATA_ROOT/dataset_manifest.json"
PACKAGE_NAME=gpqa_diamond_freeform126_manual_agent_audit_20260718_v1
PACKAGE="$OUT/audit_package/$PACKAGE_NAME"
PACKAGE_TAR="$OUT/${PACKAGE_NAME}.tar.gz"
STATE="$OUT/state.env"
RUN_START_UNIX=""
CURRENT_CELL=none
CURRENT_PHASE=PREFLIGHT

if [[ -z "${STY:-}" && "${GPQA_MANUAL_AUDIT_IN_SCREEN:-0}" != 1 ]]; then
  if screen -ls 2>/dev/null | grep -Eq "[.]${SCREEN_NAME}[[:space:]]"; then
    echo "GPQA_MANUAL_AUDIT_SCREEN_ALREADY_RUNNING screen=$SCREEN_NAME"
    exit 0
  fi
  screen -dmS "$SCREEN_NAME" env \
    GPQA_MANUAL_AUDIT_IN_SCREEN=1 \
    HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}" \
    GPQA_PREPARE_DATASET="${GPQA_PREPARE_DATASET:-0}" \
    bash "$0"
  echo "GPQA_MANUAL_AUDIT_SCREEN_LAUNCHED screen=$SCREEN_NAME"
  exit 0
fi

mkdir -p "$OUT"/{generation,exports,export_staging,receipts,logs,timings,audit_package}
exec > >(tee -a "$OUT/logs/controller.log") 2>&1
exec 9>"$OUT/controller.lock"
flock -n 9 || { echo "GPQA_MANUAL_AUDIT_CONTROLLER_ALREADY_RUNNING"; exit 9; }

export PYTHONPATH="$ROOT/src:$VERL${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_V1=1
export TOKENIZERS_PARALLELISM=false

if [[ -s "$OUT/run_start_unix.txt" ]]; then
  RUN_START_UNIX=$(tr -dc '0-9' <"$OUT/run_start_unix.txt")
else
  RUN_START_UNIX=$(date +%s)
  printf '%s\n' "$RUN_START_UNIX" >"$OUT/run_start_unix.txt"
fi

count_complete_cells() { find "$OUT/generation" -name CELL_COMPLETE 2>/dev/null | wc -l; }
count_complete_shards() { find "$OUT/generation" -path '*/shards/rank_*/SHARD_COMPLETE' 2>/dev/null | wc -l; }

write_state() {
  local phase="$1" cell="$2" completed="$3" generated="$4" packaged="$5"
  printf 'PHASE=%s\nCELL=%s\nCOMPLETED_CELLS=%s\nGENERATED_ROWS=%s\nPACKAGED_ROWS=%s\nRUN_START_UNIX=%s\nUPDATED_UNIX=%s\n' \
    "$phase" "$cell" "$completed" "$generated" "$packaged" "$RUN_START_UNIX" "$(date +%s)" >"$STATE.tmp"
  mv "$STATE.tmp" "$STATE"
}

on_exit() {
  local rc=$?
  if (( rc != 0 )); then
    write_state FAILED "$CURRENT_CELL" "$(count_complete_cells)" "$(( $(count_complete_shards) * 21 ))" 0
    printf 'WAVE_FAILED rc=%s phase=%s cell=%s time=%s\n' "$rc" "$CURRENT_PHASE" "$CURRENT_CELL" "$(date -Is)" \
      | tee "$OUT/WAVE_FAILED"
  fi
  exit "$rc"
}
trap on_exit EXIT INT TERM

disk_guard() {
  local free
  free=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
  (( free >= 60 )) || { echo "GPQA_MANUAL_AUDIT_DISK_GUARD_FAIL free=${free}G required=60G"; return 80; }
}

config_for() {
  local variant="$1" step="$2"
  if (( step > 196 )); then
    echo "$ROOT/configs/runtime/${variant}_6x5090_grpo_resume196_to294_cosine5e7_to5e8_seed20260707_v1.yaml"
  else
    echo "$ROOT/configs/runtime/${variant}_6x5090_grpo_untunedbase_resume98_to196_seed20260707_v1.yaml"
  fi
}

checkpoint_for() {
  local variant="$1" step="$2"
  if (( step == 158 )); then
    echo "$SOURCE_ROOT/${variant}_6x5090_grpo_untunedbase_resume98_to196_seed20260707_v1/checkpoints/global_step_158"
  else
    echo "$PRIORITY_ROOT/${variant}_6x5090_grpo_resume196_to294_cosine5e7_to5e8_seed20260707_v1/checkpoints/global_step_${step}"
  fi
}

complete_export() {
  local path="$1"
  [[ -f "$path/EXPORT_COMPLETE" && -s "$path/config.json" ]] && compgen -G "$path/*.safetensors" >/dev/null
}

prepare_export() {
  local variant="$1" step="$2" label="${1}_step${2}" checkpoint runtime_config model reuse
  local reuse_args=()
  checkpoint=$(checkpoint_for "$variant" "$step")
  runtime_config=$(config_for "$variant" "$step")
  model="$OUT/exports/$label"
  reuse="$PRIORITY_ROOT/exports/${variant}_step_${step}"
  if complete_export "$reuse"; then
    model="$reuse"
    reuse_args=(--reuse "$reuse")
  fi
  "$PY" "$ROOT/scripts/export_qwen3_major_checkpoint.py" \
    --variant "$variant" --global-step "$step" --checkpoint "$checkpoint" \
    --runtime-config "$runtime_config" --output "$OUT/exports/$label" --staging-root "$OUT/export_staging" \
    --receipt "$OUT/receipts/${label}_export.json" "${reuse_args[@]}" \
    >"$OUT/logs/${label}_export.log" 2>&1
  complete_export "$model"
  printf '%s\n' "$model"
}

prune_disposable_export() {
  local model="$1" label="$2"
  [[ "$model" == "$OUT/exports/"* ]] || return 0
  rm -rf -- "$model"
  printf '{"status":"PRUNED","path":"%s"}\n' "$model" >"$OUT/receipts/${label}_export_pruned.json"
}

run_generation_cell() {
  local variant="$1" step="$2" label="${1}_step${2}" cell model started status plugin_args=()
  cell="$OUT/generation/$label"
  CURRENT_CELL="$label"
  if [[ -f "$cell/CELL_COMPLETE" ]]; then
    "$PY" "$ROOT/scripts/run_gpqa_diamond_freeform126_worker.py" --root "$ROOT" --config "$CONFIG" \
      merge-generation --variant "$variant" --global-step "$step" --cell "$cell" >/dev/null
    echo "GENERATION_CELL_ALREADY_COMPLETE cell=$label"
    return 0
  fi
  disk_guard
  CURRENT_PHASE=EXPORT
  write_state EXPORT "$label" "$(count_complete_cells)" "$(( $(count_complete_shards) * 21 ))" 0
  model=$(prepare_export "$variant" "$step")
  [[ "$variant" != triglu ]] || plugin_args=(--plugin triglu)
  mkdir -p "$cell/shards"
  CURRENT_PHASE=GENERATE
  write_state GENERATE "$label" "$(count_complete_cells)" "$(( $(count_complete_shards) * 21 ))" 0
  started=$(date +%s)
  local pids=()
  status=0
  for gpu in 0 1 2 3 4 5; do
    mkdir -p "$OUT/compile_cache/vllm/rank_${gpu}" "$OUT/compile_cache/torchinductor/rank_${gpu}"
    CUDA_VISIBLE_DEVICES="$gpu" \
    VLLM_CACHE_ROOT="$OUT/compile_cache/vllm/rank_${gpu}" \
    TORCHINDUCTOR_CACHE_DIR="$OUT/compile_cache/torchinductor/rank_${gpu}" \
    "$PY" "$ROOT/scripts/run_gpqa_diamond_freeform126_worker.py" \
      --root "$ROOT" --config "$CONFIG" generate --model "$model" --variant "$variant" \
      --global-step "$step" --rank "$gpu" --output "$cell/shards/rank_${gpu}" \
      "${plugin_args[@]}" >"$cell/rank_${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
  (( status == 0 )) || return "$status"
  "$PY" "$ROOT/scripts/run_gpqa_diamond_freeform126_worker.py" --root "$ROOT" --config "$CONFIG" \
    merge-generation --variant "$variant" --global-step "$step" --cell "$cell"
  "$PY" - "$label" "$started" "$(date +%s)" "$OUT/timings/${label}.json" <<'PY'
import json
import sys
from pathlib import Path
label, started, finished, output = sys.argv[1:]
Path(output).write_text(json.dumps({"cell": label, "wall_seconds": int(finished)-int(started)}, indent=2)+"\n")
PY
  prune_disposable_export "$model" "$label"
  echo "GENERATION_CELL_COMPLETE cell=$label time=$(date -Is)"
}

CURRENT_PHASE=PREFLIGHT
write_state PREFLIGHT none 0 0 0
rm -f "$OUT/WAVE_FAILED" "$OUT/GENERATION_WAVE_COMPLETE" "$OUT/AUDIT_PACKAGE_READY"
[[ -x "$PY" && -s "$CONFIG" ]]
if screen -ls 2>/dev/null | grep -Eq '[.]qwen_other_majorsteps6_20260718_v1[[:space:]]'; then
  if [[ ! -f "$OTHER_EVAL_OUT/WAVE_COMPLETE" ]]; then
    echo "WAITING_ACTIVE_OTHER_OOD_WAVE; refusing to contend or interrupt"
    exit 75
  fi
fi
[[ $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) -eq 6 ]]
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; then
  echo "GPQA_MANUAL_AUDIT_GPU_BUSY_REFUSING_TO_CONTEND"
  exit 77
fi
disk_guard
if [[ "${GPQA_PREPARE_DATASET:-0}" == 1 ]] && [[ ! -s "$LEDGER" || ! -s "$DATA_MANIFEST" ]]; then
  CURRENT_PHASE=ASSET_PREP
  write_state ASSET_PREP none 0 0 0
  "$PY" "$ROOT/scripts/prepare_gpqa_diamond_freeform126_assets.py" \
    --root "$ROOT" --config "$CONFIG" dataset | tee "$OUT/logs/dataset_preparation.log"
fi
[[ -s "$LEDGER" && -s "$DATA_MANIFEST" ]]
"$PY" - "$LEDGER" "$DATA_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path
from qwen_single_layer_rl.eval.gpqa_freeform import file_sha256, load_jsonl
ledger, manifest_path = map(Path, sys.argv[1:])
rows = load_jsonl(ledger)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
assert len(rows) == len({row["question_id"] for row in rows}) == 126
assert [sum(row["generation_rank"] == rank for row in rows) for rank in range(6)] == [21] * 6
assert manifest["ledger_sha256"] == file_sha256(ledger)
assert manifest["resolved_revision"] not in {"", "main", None}
PY
for step in 158 196 226 256 294; do
  for variant in triglu baseline; do
    [[ -s "$(checkpoint_for "$variant" "$step")/data.pt" ]]
  done
done

run_generation_cell triglu 158
run_generation_cell baseline 158
run_generation_cell triglu 196
run_generation_cell baseline 196
run_generation_cell triglu 226
run_generation_cell baseline 226
run_generation_cell triglu 256
run_generation_cell baseline 256
run_generation_cell triglu 294
run_generation_cell baseline 294

CURRENT_PHASE=PACKAGE
CURRENT_CELL=all
write_state PACKAGE all 10 1260 0
"$PY" "$ROOT/scripts/build_gpqa_diamond_freeform126_manual_audit_package.py" \
  --root "$ROOT" --config "$CONFIG" --run-root "$OUT" --output "$PACKAGE" build \
  | tee "$OUT/logs/audit_package.log"
tar -czf "$PACKAGE_TAR" -C "$(dirname "$PACKAGE")" "$(basename "$PACKAGE")"
"$PY" - "$PACKAGE_TAR" "$PACKAGE" "$OUT/audit_package_receipt.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
archive, package, receipt = map(Path, sys.argv[1:])
payload = {
    "status": "AUDIT_PACKAGE_READY",
    "archive": str(archive.resolve()),
    "archive_bytes": archive.stat().st_size,
    "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    "package": str(package.resolve()),
    "rows": 1260,
    "question_chunks": 126,
}
receipt.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n", encoding="utf-8")
PY
touch "$OUT/GENERATION_WAVE_COMPLETE" "$OUT/AUDIT_PACKAGE_READY"
CURRENT_PHASE=AWAITING_MANUAL_AUDIT
write_state AWAITING_MANUAL_AUDIT all 10 1260 1260
rm -f "$OUT/WAVE_FAILED"
trap - EXIT INT TERM
echo "GPQA_MANUAL_AUDIT_PACKAGE_READY run_id=$RUN_ID archive=$PACKAGE_TAR time=$(date -Is)"
