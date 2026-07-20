#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp}
PROJECT="$ROOT/qwen3_1p7b_single_layer_rl_expressivity_engineering"
BUNDLE_ID=qwen3_1p7b_triglu_baseline_grpo_matched10_full_repro_20260719_v1
OUT_DIR="$PROJECT/archive_ready"
OUT="$OUT_DIR/$BUNDLE_ID.tar.gz"
META="$OUT_DIR/${BUNDLE_ID}_metadata"
LIST="$META/archive_file_list.txt"
SOURCE_SUMS="$META/source_sha256sums.txt"
INVENTORY="$META/source_inventory.json"
ENV_JSON="$META/remote_environment_manifest.json"
TRAIN_FREEZE="$META/training_rollout_pip_freeze.txt"
EVAL_FREEZE="$META/eval_scoring_pip_freeze.txt"
TRAIN_ENV_JSON="$META/training_rollout_environment.json"
EVAL_ENV_JSON="$META/eval_scoring_environment.json"
RECEIPT="$OUT_DIR/${BUNDLE_ID}_receipt.json"
ARCHIVE_SUM="$OUT.sha256"
MODE=${1:---dry-run}

TRAIN_PYTHON=${TRAIN_PYTHON:-$PROJECT/envs/vllm0102_verl061/bin/python}
EVAL_PYTHON=${EVAL_PYTHON:-$PROJECT/envs/evalscope181/bin/python}
MODEL="$ROOT/qwen3_single_layer_rl/models/Qwen3-1.7B-Base"
VERL="$ROOT/verl-v0.6.1-qwenpatch"

SERIAL="$PROJECT/runs/grpo_serial/triglu_baseline_6x5090_grpo_20to98_serial_20260712_v1"
INTERLEAVED="$PROJECT/runs/grpo_interleaved/triglu_baseline_6x5090_grpo_retro30_60_then_98to196_interleaved_20260714_v1"
PRIORITY="$PROJECT/runs/grpo_priority/triglu_priority_6x5090_grpo_step128_to294_then_baseline128_to196_20260715_v1"
REORDERED="$PROJECT/runs/grpo_reordered/baseline_then_oft_fp32_6x5090_grpo_after_triglu196_20260715_v1"

mkdir -p "$META"
rm -f "$META/pip_freeze.txt"
test ! -e "$OUT" || { echo "Refusing to overwrite $OUT" >&2; exit 2; }
test -x "$TRAIN_PYTHON"
test -x "$EVAL_PYTHON"
command -v pigz >/dev/null || { echo "pigz is required for bounded packaging wall time" >&2; exit 2; }
test -d "$PROJECT" && test -d "$MODEL" && test -d "$VERL"

CHECKPOINTS=(
  "$INTERLEAVED/triglu_6x5090_grpo_untunedbase_resume98_to196_seed20260707_v1/checkpoints/global_step_158"
  "$INTERLEAVED/baseline_6x5090_grpo_untunedbase_resume98_to196_seed20260707_v1/checkpoints/global_step_158"
  "$INTERLEAVED/triglu_6x5090_grpo_untunedbase_resume98_to196_seed20260707_v1/checkpoints/global_step_196"
  "$INTERLEAVED/baseline_6x5090_grpo_untunedbase_resume98_to196_seed20260707_v1/checkpoints/global_step_196"
  "$PRIORITY/triglu_6x5090_grpo_resume196_to294_cosine5e7_to5e8_seed20260707_v1/checkpoints/global_step_226"
  "$PRIORITY/baseline_6x5090_grpo_resume196_to294_cosine5e7_to5e8_seed20260707_v1/checkpoints/global_step_226"
  "$PRIORITY/triglu_6x5090_grpo_resume196_to294_cosine5e7_to5e8_seed20260707_v1/checkpoints/global_step_256"
  "$PRIORITY/baseline_6x5090_grpo_resume196_to294_cosine5e7_to5e8_seed20260707_v1/checkpoints/global_step_256"
  "$PRIORITY/triglu_6x5090_grpo_resume196_to294_cosine5e7_to5e8_seed20260707_v1/checkpoints/global_step_294"
  "$PRIORITY/baseline_6x5090_grpo_resume196_to294_cosine5e7_to5e8_seed20260707_v1/checkpoints/global_step_294"
)

REFERENCES=(
  "$INTERLEAVED/exports/triglu_step_98"
  "$INTERLEAVED/exports/baseline_step_98"
  "$PRIORITY/exports/triglu_step_196"
  "$PRIORITY/exports/baseline_step_196"
)

for path in "${CHECKPOINTS[@]}" "${REFERENCES[@]}"; do
  test -d "$path" || { echo "Missing required tree: $path" >&2; exit 3; }
done

for checkpoint in "${CHECKPOINTS[@]}"; do
  test -s "$checkpoint/data.pt" || { echo "Missing data state: $checkpoint/data.pt" >&2; exit 3; }
  [[ $(find "$checkpoint" -type f -name 'model_world_size_6_rank_*.pt' | wc -l) -eq 6 ]] || {
    echo "Expected six model shards: $checkpoint" >&2; exit 3;
  }
  [[ $(find "$checkpoint" -type f -name 'optim_world_size_6_rank_*.pt' | wc -l) -eq 6 ]] || {
    echo "Expected six optimizer shards: $checkpoint" >&2; exit 3;
  }
  [[ $(find "$checkpoint" -type f -name 'extra_state_world_size_6_rank_*.pt' | wc -l) -eq 6 ]] || {
    echo "Expected six extra-state shards: $checkpoint" >&2; exit 3;
  }
done

EXPECTED_MODEL_SHA=6df85b39330e5a425ee36253d0f894e4387e4f0a15b9c53cb467d668e6b3a841
ACTUAL_MODEL_SHA=$(sha256sum "$MODEL/model.safetensors" | awk '{print $1}')
[[ "$ACTUAL_MODEL_SHA" == "$EXPECTED_MODEL_SHA" ]] || {
  echo "Base model SHA mismatch: $ACTUAL_MODEL_SHA" >&2; exit 3;
}
test -s "$PROJECT/data/numina_math_cot_50k_decontam_v3_verl/train.parquet"
test -s "$PROJECT/data_manifests/numina_math_cot_50k_decontam_v3/manifest.json"

"$TRAIN_PYTHON" -m pip freeze > "$TRAIN_FREEZE"
"$EVAL_PYTHON" -m pip freeze > "$EVAL_FREEZE"

capture_environment() {
  local python=$1 label=$2 output=$3
  "$python" - "$label" "$output" <<'PY'
import importlib.util, json, platform, subprocess, sys
from importlib import metadata
from pathlib import Path

def version(distribution):
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unavailable:PackageNotFoundError"

gpu_lines = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
    text=True,
).splitlines()
payload = {
    "schema_version": 1,
    "label": sys.argv[1],
    "interpreter": sys.executable,
    "python": sys.version,
    "platform": platform.platform(),
    "gpus": gpu_lines,
    "packages": {name: version(name) for name in [
        "torch", "vllm", "triton", "transformers", "evalscope", "math-verify",
        "verl", "ray", "numpy", "pandas", "pyarrow", "datasets", "accelerate",
        "safetensors",
    ]},
    "flash_attn_installed": importlib.util.find_spec("flash_attn") is not None,
}
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

capture_environment "$TRAIN_PYTHON" training_rollout "$TRAIN_ENV_JSON"
capture_environment "$EVAL_PYTHON" eval_scoring "$EVAL_ENV_JSON"
"$TRAIN_PYTHON" - "$TRAIN_ENV_JSON" "$EVAL_ENV_JSON" "$ENV_JSON" <<'PY'
import json, sys
from pathlib import Path

training = json.loads(Path(sys.argv[1]).read_text())
evaluation = json.loads(Path(sys.argv[2]).read_text())
payload = {
    "schema_version": 1,
    "production_source_of_truth": "training_rollout",
    "base_conda_environment_is_not_authoritative": True,
    "environments": {"training_rollout": training, "eval_scoring": evaluation},
}
Path(sys.argv[3]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

build_list() {
  : > "$LIST"

  add_tree() {
    local tree=$1
    test -d "$tree" || return 0
    find "$tree" -type f ! -path '*/__pycache__/*' ! -name '*.pyc' -printf '%P\0' |
      while IFS= read -r -d '' rel; do
        printf '%s\n' "${tree#$ROOT/}/$rel"
      done >> "$LIST"
  }

  add_metadata_only() {
    local tree=$1
    test -d "$tree" || return 0
    find "$tree" \
      -type d \( -name checkpoints -o -name exports -o -name evaluations -o -name archive_ready \) -prune -o \
      -type f ! -path '*/__pycache__/*' ! -name '*.pyc' -printf '%P\0' |
      while IFS= read -r -d '' rel; do
        printf '%s\n' "${tree#$ROOT/}/$rel"
      done >> "$LIST"
  }

  for tree in \
    "$PROJECT/configs" "$PROJECT/src" "$PROJECT/scripts" "$PROJECT/tests" \
    "$PROJECT/docs" "$PROJECT/data_manifests" "$PROJECT/audit_inputs" \
    "$PROJECT/eval_artifacts" "$PROJECT/logs"; do
    add_tree "$tree"
  done

  find "$PROJECT" -maxdepth 1 -type f -printf '%P\0' |
    while IFS= read -r -d '' rel; do printf '%s\n' "qwen3_1p7b_single_layer_rl_expressivity_engineering/$rel"; done >> "$LIST"

  add_tree "$PROJECT/data/numina_math_cot_50k_decontam_v3"
  add_tree "$PROJECT/data/numina_math_cot_50k_decontam_v3_verl"

  add_metadata_only "$SERIAL"
  add_metadata_only "$INTERLEAVED"
  add_metadata_only "$PRIORITY"
  add_metadata_only "$REORDERED"

  for tree in "${CHECKPOINTS[@]}" "${REFERENCES[@]}"; do add_tree "$tree"; done
  add_tree "$INTERLEAVED/evaluations"
  add_tree "$PRIORITY/evaluations"
  add_tree "$REORDERED/evaluations"
  add_tree "$PROJECT/runs/ood_eval"
  add_tree "$PROJECT/runs/eval_protocol"
  add_tree "$PROJECT/runs/freeform_eval"
  add_tree "$MODEL"
  add_tree "$VERL"
  add_tree "$META"
  LC_ALL=C sort -u -o "$LIST" "$LIST"
}

build_list
grep -vF "${SOURCE_SUMS#$ROOT/}" "$LIST" |
  grep -vF "${INVENTORY#$ROOT/}" |
  grep -vF "${LIST#$ROOT/}" |
  (cd "$ROOT" && xargs -d '\n' sha256sum) > "$SOURCE_SUMS"
build_list

"$TRAIN_PYTHON" - "$ROOT" "$LIST" "$INVENTORY" <<'PY'
import json, os, shutil, sys
from pathlib import Path
root, list_path, output = map(Path, sys.argv[1:])
paths = [line for line in list_path.read_text().splitlines() if line]
seen = set(); source_bytes = 0
for rel in paths:
    stat = (root / rel).stat()
    key = (stat.st_dev, stat.st_ino)
    if key not in seen:
        seen.add(key)
        source_bytes += stat.st_size
free_bytes = shutil.disk_usage(root).free
payload = {
    "schema_version": 1,
    "file_entries": len(paths),
    "unique_inodes": len(seen),
    "unique_source_bytes": source_bytes,
    "free_bytes_before_archive": free_bytes,
    "minimum_post_source_margin_bytes": 6 * 1024**3,
    "space_gate_pass": free_bytes >= source_bytes + 6 * 1024**3,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, sort_keys=True))
if not payload["space_gate_pass"]:
    raise SystemExit("Insufficient worst-case space for single-pass archive")
PY
build_list

if [[ "$MODE" == "--dry-run" ]]; then
  echo "DRY_RUN_COMPLETE bundle=$BUNDLE_ID list=$LIST inventory=$INVENTORY"
  exit 0
fi
[[ "$MODE" == "--execute" ]] || { echo "Usage: $0 [--dry-run|--execute]" >&2; exit 4; }

echo "ARCHIVE_BUILD_START $(date -Is)"
(cd "$ROOT" && tar -I 'pigz -1 -p 32' -cf "$OUT" -T "$LIST")
(cd "$OUT_DIR" && sha256sum "$BUNDLE_ID.tar.gz" > "$BUNDLE_ID.tar.gz.sha256")
pigz -t -p 16 "$OUT"
ENTRY_COUNT=$(tar -I 'pigz -d -p 16' -tf "$OUT" | wc -l)

"$TRAIN_PYTHON" - "$BUNDLE_ID" "$OUT" "$ARCHIVE_SUM" "$INVENTORY" "$RECEIPT" "$ENTRY_COUNT" <<'PY'
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
bundle_id, archive, sum_file, inventory, receipt, entry_count = sys.argv[1:]
archive_path = Path(archive)
payload = {
    "schema_version": 1,
    "bundle_id": bundle_id,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "archive_path": str(archive_path),
    "archive_bytes": archive_path.stat().st_size,
    "archive_sha256": Path(sum_file).read_text().split()[0],
    "tar_entry_count": int(entry_count),
    "gzip_test": "pass",
    "tar_list_test": "pass",
    "source_inventory": json.loads(Path(inventory).read_text()),
    "checkpoint_count": 10,
    "frozen_reference_export_count": 4,
    "remote_shutdown_requested": False,
}
Path(receipt).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

cp "$SOURCE_SUMS" "$OUT_DIR/${BUNDLE_ID}_source_sha256sums.txt"
echo "ARCHIVE_BUILD_COMPLETE $(date -Is)"
cat "$RECEIPT"
