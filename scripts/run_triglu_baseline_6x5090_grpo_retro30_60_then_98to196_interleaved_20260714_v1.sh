#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL=/root/autodl-tmp/verl-v0.6.1-qwenpatch
BASE_MODEL=/root/autodl-tmp/qwen3_single_layer_rl/models/Qwen3-1.7B-Base
TRIGLU_MODEL="$ROOT/runs/runtime_smokes/baseline_triglu_2x5090_grpo_prelaunch_gate_20260712_v1/triglu_exact_noop_export"
DATA="$ROOT/data/numina_math_cot_50k_decontam_v3_verl"
OLD_WAVE=triglu_baseline_6x5090_grpo_20to98_serial_20260712_v1
OLD_ROOT="$ROOT/runs/grpo_serial/$OLD_WAVE"
WAVE=triglu_baseline_6x5090_grpo_retro30_60_then_98to196_interleaved_20260714_v1
RUN_ROOT="$ROOT/runs/grpo_interleaved/$WAVE"
PY="$ROOT/envs/vllm0102_verl061/bin/python"
STATE="$RUN_ROOT/state.env"
CURRENT_VARIANT=none
CURRENT_TARGET=0

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/evaluations" "$RUN_ROOT/data_order" \
  "$RUN_ROOT/retention" "$RUN_ROOT/exports" "$RUN_ROOT/references"
exec > >(tee -a "$RUN_ROOT/logs/controller.log") 2>&1

export PYTHONPATH="$ROOT/src:$VERL${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_V1=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

set_state() {
  printf 'PHASE=%q\nVARIANT=%q\nTARGET=%q\nSTART_UNIX=%q\n' \
    "$1" "$2" "$3" "$(date +%s)" >"$STATE.tmp"
  mv "$STATE.tmp" "$STATE"
}

on_error() {
  local rc=$?
  trap - ERR
  set_state FAILED "$CURRENT_VARIANT" "$CURRENT_TARGET"
  printf 'WAVE_FAILED rc=%s variant=%s target=%s time=%s\n' \
    "$rc" "$CURRENT_VARIANT" "$CURRENT_TARGET" "$(date -Is)" \
    | tee "$RUN_ROOT/WAVE_FAILED"
  exit "$rc"
}
trap on_error ERR

disk_guard() {
  local free
  free=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
  (( free >= 100 )) || {
    echo "DISK_GUARD_FAIL free=${free}G required=100G"
    return 80
  }
}

config_for() {
  echo "$ROOT/configs/runtime/$1_6x5090_grpo_untunedbase_resume98_to196_seed20260707_v1.yaml"
}

old_run_id_for() {
  echo "$1_6x5090_grpo_untunedbase_b98_seed20260707_v1"
}

run_id_for() {
  echo "$1_6x5090_grpo_untunedbase_resume98_to196_seed20260707_v1"
}

model_for() {
  [[ "$1" == triglu ]] && echo "$TRIGLU_MODEL" || echo "$BASE_MODEL"
}

reference_model_for() {
  echo "$RUN_ROOT/exports/${1}_step_98"
}

old_checkpoint() {
  local variant="$1" step="$2"
  echo "$OLD_ROOT/$(old_run_id_for "$variant")/checkpoints/global_step_${step}"
}

new_checkpoint() {
  local variant="$1" step="$2"
  echo "$RUN_ROOT/$(run_id_for "$variant")/checkpoints/global_step_${step}"
}

checkpoint_for_step() {
  local variant="$1" step="$2" candidate
  candidate=$(new_checkpoint "$variant" "$step")
  if [[ -d "$candidate" ]]; then
    echo "$candidate"
  else
    old_checkpoint "$variant" "$step"
  fi
}

validate_checkpoint() {
  local checkpoint="$1" actor="$1/actor"
  [[ -s "$checkpoint/data.pt" ]]
  [[ -s "$actor/fsdp_config.json" ]]
  [[ $(find "$actor" -maxdepth 1 -type f -name 'model_world_size_6_rank_*.pt' | wc -l) -eq 6 ]]
  [[ $(find "$actor" -maxdepth 1 -type f -name 'optim_world_size_6_rank_*.pt' | wc -l) -eq 6 ]]
  [[ $(find "$actor" -maxdepth 1 -type f -name 'extra_state_world_size_6_rank_*.pt' | wc -l) -eq 6 ]]
}

data_order_receipt() {
  local step="$1" scope="$2" triglu_data baseline_data output
  if [[ "$scope" == old ]]; then
    triglu_data="$(old_checkpoint triglu "$step")/data.pt"
    baseline_data="$(old_checkpoint baseline "$step")/data.pt"
  else
    triglu_data="$(new_checkpoint triglu "$step")/data.pt"
    baseline_data="$(new_checkpoint baseline "$step")/data.pt"
  fi
  output="$RUN_ROOT/data_order/${scope}_step_${step}.json"
  "$PY" - "$triglu_data" "$baseline_data" "$DATA/train.parquet" "$step" "$output" <<'PY'
import hashlib
import json
import struct
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch

triglu_path, baseline_path, parquet_path, step_text, output_path = map(Path, sys.argv[1:])
step = int(str(step_text))

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def normalize(value):
    if isinstance(value, dict):
        return {
            str(key): normalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) != "_base_seed"
        }
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)

states = {
    "triglu": torch.load(triglu_path, map_location="cpu", weights_only=False),
    "baseline": torch.load(baseline_path, map_location="cpu", weights_only=False),
}
normalized = {name: normalize(state) for name, state in states.items()}
if normalized["triglu"] != normalized["baseline"]:
    raise RuntimeError("normalized dataloader states differ between variants")

row_count = pq.ParquetFile(parquet_path).metadata.num_rows
batch_size = 504
full_batches = row_count // batch_size
generator = torch.Generator().manual_seed(20260707)
remaining_steps = step
ledger_hash = hashlib.sha256()
while remaining_steps:
    permutation = torch.randperm(row_count, generator=generator)
    take_batches = min(remaining_steps, full_batches)
    consumed = permutation[: take_batches * batch_size].tolist()
    for row_id in consumed:
        ledger_hash.update(struct.pack("<q", row_id))
    remaining_steps -= take_batches

main = normalized["triglu"].get("_snapshot", {}).get("_main_snapshot", {})
sampler = main.get("_sampler_iter_state", {})
payload = {
    "status": "PASS",
    "scope": str(output_path.stem).split("_step_")[0],
    "global_step": step,
    "variants_normalized_equal": True,
    "ignored_state_fields": ["_base_seed"],
    "explicit_random_sampler_seed": 20260707,
    "train_rows": row_count,
    "batch_size": batch_size,
    "samples_yielded_in_current_iterator": sampler.get("samples_yielded"),
    "sampler_iter_yielded": main.get("_sampler_iter_yielded"),
    "expected_prompt_index_ledger_sha256": ledger_hash.hexdigest(),
    "raw_data_pt_sha256": {
        "triglu": sha256(triglu_path),
        "baseline": sha256(baseline_path),
    },
}
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("DATA_ORDER_RECEIPT_PASS", json.dumps(payload, sort_keys=True))
PY
}

write_resume_receipt() {
  "$PY" - "$OLD_ROOT" "$RUN_ROOT/resume_source_receipt.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

old_root, output = map(Path, sys.argv[1:])
payload = {"status": "PASS", "source_wave_root": str(old_root.resolve()), "variants": {}}
for variant in ("triglu", "baseline"):
    run_id = f"{variant}_6x5090_grpo_untunedbase_b98_seed20260707_v1"
    checkpoint = old_root / run_id / "checkpoints" / "global_step_98"
    files = []
    for path in sorted(item for item in checkpoint.rglob("*") if item.is_file()):
        relative = str(path.relative_to(checkpoint))
        entry = {"path": relative, "size": path.stat().st_size}
        if path.stat().st_size <= 1 << 20:
            entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append(entry)
    payload["variants"][variant] = {
        "checkpoint": str(checkpoint.resolve()),
        "file_count": len(files),
        "files": files,
    }
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("RESUME_SOURCE_RECEIPT_PASS")
PY
}

train_to() {
  local variant="$1" target="$2" config run_id model reference manifest cmd log tracker completed resume_source
  CURRENT_VARIANT="$variant"
  CURRENT_TARGET="$target"
  config=$(config_for "$variant")
  run_id=$(run_id_for "$variant")
  model=$(model_for "$variant")
  reference=$(reference_model_for "$variant")
  [[ -f "$reference/EXPORT_COMPLETE" ]]
  [[ -s "$reference/config.json" ]]
  compgen -G "$reference/*.safetensors" >/dev/null
  manifest="$RUN_ROOT/${variant}_command_manifest_to_${target}.json"
  log="$RUN_ROOT/logs/${variant}_to_${target}.log"
  tracker="$RUN_ROOT/$run_id/checkpoints/latest_checkpointed_iteration.txt"
  completed=0
  [[ -f "$tracker" ]] && completed=$(tr -dc '0-9' <"$tracker")
  if (( completed >= target )) && validate_checkpoint "$(new_checkpoint "$variant" "$target")"; then
    echo "TRAIN_SEGMENT_ALREADY_COMPLETE variant=$variant target=$target checkpoint=$completed"
    return 0
  fi

  disk_guard
  set_state TRAIN "$variant" "$target"
  local command_args=(
    --config "$config"
    --project-root "$ROOT"
    --verl-root "$VERL"
    --model-path "$model"
    --reference-model-path "$reference"
    --data-dir "$DATA"
    --run-root "$RUN_ROOT"
    --manifest-out "$manifest"
    --print-shell
  )
  if (( completed == 0 )); then
    resume_source=$(old_checkpoint "$variant" 98)
    validate_checkpoint "$resume_source"
    command_args+=(--resume-from-path "$resume_source")
  fi
  cmd=$($PY -m qwen_single_layer_rl.training.verl_command "${command_args[@]}")
  echo "TRAIN_SEGMENT_START variant=$variant target=$target resume_completed=$completed $(date -Is)" | tee -a "$log"
  set -o pipefail
  bash -lc "$cmd trainer.total_training_steps=$target" 2>&1 | tee -a "$log"
  local rc=${PIPESTATUS[0]}
  (( rc == 0 )) || return "$rc"
  validate_checkpoint "$(new_checkpoint "$variant" "$target")"
  echo "TRAIN_SEGMENT_COMPLETE variant=$variant target=$target $(date -Is)" | tee -a "$log"
  "$PY" - "$RUN_ROOT/$run_id/audits/actor_rank0_model_surgery_audit.json" "$variant" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
names = payload["trainable_parameter_names"]
expected = 23 if sys.argv[2] == "triglu" else 11
assert len(names) == expected, (len(names), expected, names)
PY
  ray stop --force >/dev/null 2>&1 || true
}

export_checkpoint() {
  local variant="$1" step="$2" checkpoint actor merge_actor out config staging
  checkpoint=$(checkpoint_for_step "$variant" "$step")
  validate_checkpoint "$checkpoint"
  actor="$checkpoint/actor"
  merge_actor="$actor"
  config=$(config_for "$variant")
  out="$RUN_ROOT/exports/${variant}_step_${step}"
  if [[ -f "$out/EXPORT_COMPLETE" ]] && { [[ ! -s "$out/config.json" ]] || ! compgen -G "$out/*.safetensors" >/dev/null; }; then
    echo "INVALID_EXPORT_MARKER_REMOVED variant=$variant step=$step" >&2
    rm -rf "$out"
  fi
  if [[ ! -f "$out/EXPORT_COMPLETE" ]]; then
    rm -rf "$out"
    mkdir -p "$RUN_ROOT/exports"
    if [[ "$variant" == triglu ]]; then
      staging="$RUN_ROOT/export_staging/${variant}_step_${step}/actor"
      rm -rf "$staging"
      mkdir -p "$staging"
      for path in "$actor"/*; do
        [[ "$(basename "$path")" == huggingface ]] && continue
        ln -s "$path" "$staging/$(basename "$path")"
      done
      cp -a "$actor/huggingface" "$staging/huggingface"
      "$PY" - "$staging/huggingface/triglu_hf_model.py" >&2 <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
needle = "from . import TRIGLU_ARCHITECTURE"
if text.count(needle) != 1:
    raise RuntimeError(f"expected one package constant import in {path}")
text = text.replace(needle, 'TRIGLU_ARCHITECTURE = "Qwen3TriGLUForCausalLM"')
path.write_text(text, encoding="utf-8")
root = path.parent
(root / "configuration_qwen3_triglu.py").write_text(
    "from qwen_single_layer_rl.vllm.triglu_hf_model import Qwen3TriGLUConfig\n",
    encoding="utf-8",
)
(root / "modeling_qwen3_triglu.py").write_text(
    "from qwen_single_layer_rl.vllm.triglu_hf_model import Qwen3TriGLUForCausalLM, Qwen3TriGLUModel\n",
    encoding="utf-8",
)
(root / "export_staging_repair.json").write_text(
    json.dumps({"status": "staging_only_repair", "source_checkpoint_unchanged": True}, indent=2) + "\n",
    encoding="utf-8",
)
PY
      HF_MODULES_CACHE="$staging/.hf_modules_preflight" "$PY" - "$staging/huggingface" >&2 <<'PY'
import json
import sys
from pathlib import Path
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

root = Path(sys.argv[1])
config = AutoConfig.from_pretrained(root, trust_remote_code=True)
raw = json.loads((root / "config.json").read_text(encoding="utf-8"))
for key in ("AutoModel", "AutoModelForCausalLM"):
    get_class_from_dynamic_module(raw["auto_map"][key], root)
assert config.model_type == "qwen3_triglu"
PY
      merge_actor="$staging"
    fi
    HF_MODULES_CACHE="${staging:-$out/.hf_modules_cache}" "$PY" -m verl.model_merger merge \
      --backend fsdp --trust-remote-code --local_dir "$merge_actor" --target_dir "$out" >&2
    if [[ "$variant" == triglu ]]; then
      "$PY" - "$config" "$out" >&2 <<'PY'
import json
import sys
from pathlib import Path
from qwen_single_layer_rl.config import load_config
from qwen_single_layer_rl.vllm.triglu_hf_model import Qwen3TriGLUConfig

cfg = load_config(Path(sys.argv[1]))
out = Path(sys.argv[2])
raw = json.loads((out / "config.json").read_text(encoding="utf-8"))
for key in ("model_type", "architectures", "triglu_variant", "auto_map"):
    raw.pop(key, None)
custom = Qwen3TriGLUConfig(triglu_variant=cfg["architecture_variant"]["params"], **raw)
custom.save_pretrained(out)
(out / "configuration_qwen3_triglu.py").write_text(
    "from qwen_single_layer_rl.vllm.triglu_hf_model import Qwen3TriGLUConfig\n", encoding="utf-8"
)
(out / "modeling_qwen3_triglu.py").write_text(
    "from qwen_single_layer_rl.vllm.triglu_hf_model import Qwen3TriGLUForCausalLM, Qwen3TriGLUModel\n",
    encoding="utf-8",
)
PY
    fi
    [[ -s "$out/config.json" ]]
    compgen -G "$out/*.safetensors" >/dev/null
    touch "$out/EXPORT_COMPLETE"
    [[ -n "${staging:-}" ]] && rm -rf "$(dirname "$(dirname "$staging")")"
  fi
  echo "$out"
}

prune_export() {
  local variant="$1" step="$2" out="$RUN_ROOT/exports/${variant}_step_${step}"
  [[ "$step" -eq 98 || "$step" -eq 196 ]] && return 0
  [[ -f "$RUN_ROOT/evaluations/${variant}_step_${step}/PARALLEL_EVAL_COMPLETE" ]]
  "$PY" - "$RUN_ROOT" "$out" "$RUN_ROOT/retention/${variant}_step_${step}_export.json" <<'PY'
import json
import shutil
import sys
from pathlib import Path

run_root, target, receipt = map(Path, sys.argv[1:])
resolved_root = run_root.resolve()
resolved_target = target.resolve()
if not resolved_target.is_relative_to(resolved_root / "exports"):
    raise RuntimeError(f"refusing export prune outside new wave: {resolved_target}")
size = sum(path.stat().st_size for path in target.rglob("*") if path.is_file()) if target.exists() else 0
if target.exists():
    shutil.rmtree(target)
receipt.write_text(json.dumps({"status": "PRUNED", "path": str(target), "bytes": size}, indent=2) + "\n")
PY
}

prepare_reference() {
  local variant="$1" model receipt
  model=$(reference_model_for "$variant")
  if [[ ! -f "$model/EXPORT_COMPLETE" ]]; then
    export_checkpoint "$variant" 98 >/dev/null
  fi
  [[ -s "$model/config.json" ]]
  compgen -G "$model/*.safetensors" >/dev/null
  receipt="$RUN_ROOT/references/${variant}_step98_reference.json"
  "$PY" - "$variant" "$model" "$(old_checkpoint "$variant" 98)" "$receipt" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

variant, model_text, checkpoint_text, receipt_text = sys.argv[1:]
model = Path(model_text).resolve()
checkpoint = Path(checkpoint_text).resolve()
receipt = Path(receipt_text)
files = []
for path in sorted(item for item in model.rglob("*") if item.is_file()):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    files.append({"path": str(path.relative_to(model)), "bytes": path.stat().st_size, "sha256": digest})
if not any(item["path"].endswith(".safetensors") for item in files):
    raise RuntimeError(f"reference export has no safetensors: {model}")
payload = {
    "status": "FROZEN_OWN_STEP98_REFERENCE_READY",
    "variant": variant,
    "reference_global_step": 98,
    "reference_model_path": str(model),
    "source_checkpoint": str(checkpoint),
    "files": files,
}
receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"REFERENCE_POLICY_READY variant={variant} step=98 model={model}")
PY
}

prune_redundant_regular_checkpoint() {
  local variant="$1" endpoint="$2" redundant_step="" target receipt
  case "$endpoint" in
    128) redundant_step=100 ;;
    158) redundant_step=130 ;;
    196) redundant_step=160 ;;
    *) return 0 ;;
  esac
  [[ -f "$RUN_ROOT/evaluations/${variant}_step_${endpoint}/PARALLEL_EVAL_COMPLETE" ]]
  validate_checkpoint "$(new_checkpoint "$variant" "$endpoint")"
  target=$(new_checkpoint "$variant" "$redundant_step")
  receipt="$RUN_ROOT/retention/${variant}_step_${redundant_step}_checkpoint.json"
  "$PY" - "$RUN_ROOT" "$target" "$receipt" "$endpoint" <<'PY'
import json
import shutil
import sys
from pathlib import Path

run_root, target, receipt = map(Path, sys.argv[1:4])
endpoint = int(sys.argv[4])
resolved_root = run_root.resolve()
resolved_target = target.resolve()
expected_parent = resolved_root / target.parent.relative_to(run_root.resolve())
if resolved_target.parent != expected_parent or not target.name.startswith("global_step_"):
    raise RuntimeError(f"refusing checkpoint prune outside exact new-wave checkpoint path: {resolved_target}")
size = sum(path.stat().st_size for path in target.rglob("*") if path.is_file()) if target.exists() else 0
if target.exists():
    shutil.rmtree(target)
receipt.write_text(
    json.dumps(
        {
            "status": "PRUNED_REDUNDANT_REGULAR_CHECKPOINT",
            "path": str(target),
            "bytes": size,
            "protected_endpoint": endpoint,
        },
        indent=2,
    )
    + "\n"
)
PY
}

evaluate() {
  local variant="$1" step="$2" config model out
  CURRENT_VARIANT="$variant"
  CURRENT_TARGET="$step"
  config=$(config_for "$variant")
  set_state EVAL "$variant" "$step"
  disk_guard
  out="$RUN_ROOT/evaluations/${variant}_step_${step}"
  if [[ -f "$out/PARALLEL_EVAL_COMPLETE" ]]; then
    echo "PARALLEL_EVAL_ALREADY_COMPLETE variant=$variant step=$step"
  else
    model=$(export_checkpoint "$variant" "$step")
    rm -rf "$out"
    "$ROOT/scripts/run_parallel_vllm_eval_6gpu_20260712_v1.sh" "$config" "$model" "$out" \
      2>&1 | tee "$RUN_ROOT/logs/${variant}_eval_${step}.log"
    [[ -f "$out/PARALLEL_EVAL_COMPLETE" ]]
    ray stop --force >/dev/null 2>&1 || true
  fi
  prune_export "$variant" "$step"
  prune_redundant_regular_checkpoint "$variant" "$step"
}

CURRENT_VARIANT=none
CURRENT_TARGET=98
set_state PREFLIGHT none 98
rm -f "$RUN_ROOT/WAVE_FAILED"
[[ $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) -eq 6 ]]
[[ -s "$BASE_MODEL/model.safetensors.index.json" || -s "$BASE_MODEL/model.safetensors" ]]
[[ -s "$TRIGLU_MODEL/config.json" ]]
grep -q qwen3_triglu "$TRIGLU_MODEL/config.json"
[[ $(sha256sum "$DATA/train.parquet" | awk '{print $1}') == 16c145f165236a292140dd4fb86c4a0c4f7c6241a2390668cbf1d9364ded43d9 ]]
[[ $(sha256sum "$DATA/val.parquet" | awk '{print $1}') == b6a3e2c0538d686258736fc8d0c655b296b52433205bbab2665f89347de6a3b3 ]]
[[ -f "$OLD_ROOT/WAVE_COMPLETE" ]]
grep -q '^PHASE=COMPLETE$' "$OLD_ROOT/state.env"
grep -q '^VARIANT=all$' "$OLD_ROOT/state.env"
grep -q '^TARGET=98$' "$OLD_ROOT/state.env"
for variant in triglu baseline; do
  validate_checkpoint "$(old_checkpoint "$variant" 30)"
  validate_checkpoint "$(old_checkpoint "$variant" 60)"
  validate_checkpoint "$(old_checkpoint "$variant" 98)"
  [[ -f "$OLD_ROOT/evaluations/${variant}_step_98/PARALLEL_EVAL_COMPLETE" ]]
done
disk_guard
data_order_receipt 98 old
write_resume_receipt
prepare_reference triglu
prepare_reference baseline

evaluate triglu 30
evaluate baseline 30
evaluate triglu 60
evaluate baseline 60

for target in 128 158 196; do
  train_to triglu "$target"
  evaluate triglu "$target"
  train_to baseline "$target"
  evaluate baseline "$target"
  data_order_receipt "$target" new
done

CURRENT_VARIANT=all
CURRENT_TARGET=196
set_state COMPLETE all 196
touch "$RUN_ROOT/WAVE_COMPLETE"
rm -f "$RUN_ROOT/WAVE_FAILED"
trap - ERR
echo "WAVE_COMPLETE $(date -Is)"
