#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL=/root/autodl-tmp/verl-v0.6.1-qwenpatch
BASE_MODEL=/root/autodl-tmp/qwen3_single_layer_rl/models/Qwen3-1.7B-Base
DATA="$ROOT/data/numina_math_cot_50k_decontam_v3_verl"
OLD_WAVE=triglu_baseline_6x5090_grpo_20to98_serial_20260712_v1
OLD_ROOT="$ROOT/runs/grpo_serial/$OLD_WAVE"
SOURCE_WAVE=triglu_baseline_6x5090_grpo_retro30_60_then_98to196_interleaved_20260714_v1
SOURCE_ROOT="$ROOT/runs/grpo_interleaved/$SOURCE_WAVE"
PRIORITY_WAVE=triglu_priority_6x5090_grpo_step128_to294_then_baseline128_to196_20260715_v1
PRIORITY_ROOT="$ROOT/runs/grpo_priority/$PRIORITY_WAVE"
WAVE=baseline_then_oft_fp32_6x5090_grpo_after_triglu196_20260715_v1
RUN_ROOT="$ROOT/runs/grpo_reordered/$WAVE"
BASELINE_RUN_ID=baseline_6x5090_grpo_untunedbase_resume98_to196_seed20260707_v1
OFT_RUN_ID=oft_fp32_swigluonly_6x5090_grpo_untunedbase_to196_seed20260707_v1
OFT_MODEL="$ROOT/runs/runtime_models/oft_fp32_swigluonly_exact_identity_20260715_v1"
BASELINE_CONFIG="$ROOT/configs/runtime/baseline_6x5090_grpo_untunedbase_resume98_to196_seed20260707_v1.yaml"
OFT_CONFIG="$ROOT/configs/runtime/oft_fp32_swigluonly_6x5090_grpo_untunedbase_to196_seed20260707_v1.yaml"
PY="$ROOT/envs/vllm0102_verl061/bin/python"
STATE="$RUN_ROOT/state.env"
CURRENT_VARIANT=none
CURRENT_TARGET=0

mkdir -p "$RUN_ROOT"/{logs,evaluations,exports,references,receipts,data_order,retention,preflight}
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
    "$rc" "$CURRENT_VARIANT" "$CURRENT_TARGET" "$(date -Is)" | tee "$RUN_ROOT/WAVE_FAILED"
  exit "$rc"
}
trap on_error ERR

disk_guard() {
  local free
  free=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
  (( free >= 100 )) || { echo "DISK_GUARD_FAIL free=${free}G required=100G"; return 80; }
}

validate_checkpoint() {
  local checkpoint="$1" actor="$1/actor"
  [[ -s "$checkpoint/data.pt" ]]
  [[ -s "$actor/fsdp_config.json" ]]
  [[ $(find "$actor" -maxdepth 1 -type f -name 'model_world_size_6_rank_*.pt' | wc -l) -eq 6 ]]
  [[ $(find "$actor" -maxdepth 1 -type f -name 'optim_world_size_6_rank_*.pt' | wc -l) -eq 6 ]]
  [[ $(find "$actor" -maxdepth 1 -type f -name 'extra_state_world_size_6_rank_*.pt' | wc -l) -eq 6 ]]
}

old_checkpoint() {
  echo "$OLD_ROOT/${1}_6x5090_grpo_untunedbase_b98_seed20260707_v1/checkpoints/global_step_${2}"
}

baseline_checkpoint() {
  local step="$1"
  if (( step <= 98 )); then old_checkpoint baseline "$step"; else echo "$SOURCE_ROOT/$BASELINE_RUN_ID/checkpoints/global_step_${step}"; fi
}

oft_checkpoint() {
  echo "$RUN_ROOT/$OFT_RUN_ID/checkpoints/global_step_${1}"
}

audit_trainables() {
  local variant="$1" root="$2" run_id="$3"
  "$PY" - "$root/$run_id/audits/actor_rank0_model_surgery_audit.json" "$variant" <<'PY'
import json
import sys

path, variant = sys.argv[1:]
payload = json.load(open(path, encoding="utf-8"))
names = payload["trainable_parameter_names"]
if len(names) != 11 or payload["actual_trainable_parameter_names"] != sorted(names):
    raise RuntimeError(f"{variant}: expected exactly 11 trainable tensors, got {names}")
if variant == "oft":
    oft = [name for name in names if ".oft_" in name and name.endswith(".oft_like.oft_like")]
    attention = [name for name in names if ".self_attn." in name]
    norms = [name for name in names if name.endswith("layernorm.weight")]
    if len(oft) != 3 or len(attention) != 6 or len(norms) != 2:
        raise RuntimeError(f"OFT trainable partition is not 3+6+2: {names}")
    if any("self_attn" in name and "oft" in name for name in names):
        raise RuntimeError("attention OFT unexpectedly trainable")
    if any(".base_mlp." in name for name in names):
        raise RuntimeError("base SwiGLU unexpectedly trainable")
print(f"TRAINABLE_AUDIT_PASS variant={variant} count={len(names)}")
PY
}

train_baseline_to() {
  local target="$1" tracker completed manifest cmd log reference
  CURRENT_VARIANT=baseline; CURRENT_TARGET="$target"
  tracker="$SOURCE_ROOT/$BASELINE_RUN_ID/checkpoints/latest_checkpointed_iteration.txt"
  completed=0; [[ -f "$tracker" ]] && completed=$(tr -dc '0-9' <"$tracker")
  if (( completed >= target )) && validate_checkpoint "$(baseline_checkpoint "$target")"; then
    echo "TRAIN_SEGMENT_ALREADY_COMPLETE variant=baseline target=$target checkpoint=$completed"
    return
  fi
  reference="$SOURCE_ROOT/exports/baseline_step_98"
  [[ -f "$reference/EXPORT_COMPLETE" ]]; compgen -G "$reference/*.safetensors" >/dev/null
  disk_guard; set_state TRAIN baseline "$target"
  manifest="$RUN_ROOT/baseline_command_manifest_to_${target}.json"
  log="$RUN_ROOT/logs/baseline_to_${target}.log"
  cmd=$($PY -m qwen_single_layer_rl.training.verl_command \
    --config "$BASELINE_CONFIG" --project-root "$ROOT" --verl-root "$VERL" \
    --model-path "$BASE_MODEL" --reference-model-path "$reference" --data-dir "$DATA" \
    --run-root "$SOURCE_ROOT" --manifest-out "$manifest" --print-shell)
  echo "TRAIN_SEGMENT_START variant=baseline target=$target resume_completed=$completed $(date -Is)" | tee -a "$log"
  set -o pipefail
  bash -lc "$cmd trainer.total_training_steps=$target" 2>&1 | tee -a "$log"
  local rc=${PIPESTATUS[0]}; (( rc == 0 )) || return "$rc"
  validate_checkpoint "$(baseline_checkpoint "$target")"
  audit_trainables baseline "$SOURCE_ROOT" "$BASELINE_RUN_ID"
  echo "TRAIN_SEGMENT_COMPLETE variant=baseline target=$target $(date -Is)" | tee -a "$log"
  ray stop --force >/dev/null 2>&1 || true
}

oft_reference_for_target() {
  local target="$1"
  if (( target <= 98 )); then echo "$OFT_MODEL"; else echo "$RUN_ROOT/exports/oft_step_98"; fi
}

train_oft_to() {
  local target="$1" tracker completed manifest cmd log reference
  CURRENT_VARIANT=oft; CURRENT_TARGET="$target"
  tracker="$RUN_ROOT/$OFT_RUN_ID/checkpoints/latest_checkpointed_iteration.txt"
  completed=0; [[ -f "$tracker" ]] && completed=$(tr -dc '0-9' <"$tracker")
  if (( completed >= target )) && validate_checkpoint "$(oft_checkpoint "$target")"; then
    echo "TRAIN_SEGMENT_ALREADY_COMPLETE variant=oft target=$target checkpoint=$completed"
    return
  fi
  reference=$(oft_reference_for_target "$target")
  [[ -f "$reference/EXPORT_COMPLETE" ]]; compgen -G "$reference/*.safetensors" >/dev/null
  disk_guard; set_state TRAIN oft "$target"
  manifest="$RUN_ROOT/oft_command_manifest_to_${target}.json"
  log="$RUN_ROOT/logs/oft_to_${target}.log"
  cmd=$($PY -m qwen_single_layer_rl.training.verl_command \
    --config "$OFT_CONFIG" --project-root "$ROOT" --verl-root "$VERL" \
    --model-path "$OFT_MODEL" --reference-model-path "$reference" --data-dir "$DATA" \
    --run-root "$RUN_ROOT" --manifest-out "$manifest" --print-shell)
  echo "TRAIN_SEGMENT_START variant=oft target=$target resume_completed=$completed $(date -Is)" | tee -a "$log"
  set -o pipefail
  bash -lc "$cmd trainer.total_training_steps=$target" 2>&1 | tee -a "$log"
  local rc=${PIPESTATUS[0]}; (( rc == 0 )) || return "$rc"
  validate_checkpoint "$(oft_checkpoint "$target")"
  audit_trainables oft "$RUN_ROOT" "$OFT_RUN_ID"
  echo "TRAIN_SEGMENT_COMPLETE variant=oft target=$target $(date -Is)" | tee -a "$log"
  ray stop --force >/dev/null 2>&1 || true
}

export_checkpoint() {
  local variant="$1" step="$2" checkpoint config out staging merge_actor actor
  if [[ "$variant" == baseline ]]; then
    checkpoint=$(baseline_checkpoint "$step"); config="$BASELINE_CONFIG"
  else
    checkpoint=$(oft_checkpoint "$step"); config="$OFT_CONFIG"
  fi
  validate_checkpoint "$checkpoint"
  actor="$checkpoint/actor"
  out="$RUN_ROOT/exports/${variant}_step_${step}"
  if [[ -f "$out/EXPORT_COMPLETE" ]] && [[ -s "$out/config.json" ]] && compgen -G "$out/*.safetensors" >/dev/null; then
    echo "$out"; return
  fi
  rm -rf "$out"
  merge_actor="$actor"
  if [[ "$variant" == oft ]]; then
    staging="$RUN_ROOT/export_staging/oft_step_${step}/actor"
    rm -rf "$staging"; mkdir -p "$staging"
    for path in "$actor"/*; do
      [[ "$(basename "$path")" == huggingface ]] && continue
      ln -s "$path" "$staging/$(basename "$path")"
    done
    cp -a "$actor/huggingface" "$staging/huggingface"
    cp "$ROOT/src/qwen_single_layer_rl/vllm/oft_hf_model.py" "$staging/huggingface/oft_hf_model.py"
    "$PY" - "$staging/huggingface/oft_hf_model.py" <<'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
text = text.replace("from . import OFT_ARCHITECTURE", 'OFT_ARCHITECTURE = "Qwen3OFTForCausalLM"')
path.write_text(text, encoding="utf-8")
PY
    merge_actor="$staging"
  fi
  HF_MODULES_CACHE="$RUN_ROOT/.hf_modules_merge_${variant}_${step}" "$PY" -m verl.model_merger merge \
    --backend fsdp --trust-remote-code --local_dir "$merge_actor" --target_dir "$out" >&2
  if [[ "$variant" == oft ]]; then
    "$PY" - "$OFT_CONFIG" "$out" <<'PY'
import json
import sys
from pathlib import Path
from qwen_single_layer_rl.config import load_config
from qwen_single_layer_rl.vllm.oft_hf_model import Qwen3OFTConfig

cfg = load_config(Path(sys.argv[1])); out = Path(sys.argv[2])
raw = json.loads((out / "config.json").read_text(encoding="utf-8"))
for key in ("model_type", "architectures", "auto_map", "oft_variant"):
    raw.pop(key, None)
Qwen3OFTConfig(oft_variant=cfg["architecture_variant"]["params"], **raw).save_pretrained(out)
(out / "configuration_qwen3_oft.py").write_text(
    "from qwen_single_layer_rl.vllm.oft_hf_model import Qwen3OFTConfig\n", encoding="utf-8"
)
(out / "modeling_qwen3_oft.py").write_text(
    "from qwen_single_layer_rl.vllm.oft_hf_model import Qwen3OFTForCausalLM, Qwen3OFTModel\n",
    encoding="utf-8",
)
PY
    rm -rf "$RUN_ROOT/export_staging/oft_step_${step}"
  fi
  [[ -s "$out/config.json" ]]; compgen -G "$out/*.safetensors" >/dev/null
  touch "$out/EXPORT_COMPLETE"
  echo "$out"
}

prune_export() {
  local variant="$1" step="$2" out="$RUN_ROOT/exports/${variant}_step_${step}"
  [[ "$step" -eq 98 || "$step" -eq 196 ]] && return
  [[ -f "$RUN_ROOT/evaluations/${variant}_step_${step}/PARALLEL_EVAL_COMPLETE" ]]
  "$PY" - "$RUN_ROOT" "$out" "$RUN_ROOT/retention/${variant}_step_${step}_export.json" <<'PY'
import json, shutil, sys
from pathlib import Path
root, target, receipt = map(Path, sys.argv[1:])
if not target.resolve().is_relative_to((root / "exports").resolve()):
    raise RuntimeError(f"unsafe export prune: {target}")
size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file()) if target.exists() else 0
if target.exists(): shutil.rmtree(target)
receipt.write_text(json.dumps({"status":"PRUNED","path":str(target),"bytes":size}, indent=2)+"\n")
PY
}

prune_regular_checkpoint() {
  local variant="$1" endpoint="$2" redundant=""
  case "$endpoint" in 128) redundant=100;; 158) redundant=130;; 196) redundant=160;; *) return;; esac
  [[ -f "$RUN_ROOT/evaluations/${variant}_step_${endpoint}/PARALLEL_EVAL_COMPLETE" ]]
  local target
  if [[ "$variant" == baseline ]]; then target=$(baseline_checkpoint "$redundant"); else target=$(oft_checkpoint "$redundant"); fi
  "$PY" - "$SOURCE_ROOT" "$RUN_ROOT" "$target" "$RUN_ROOT/retention/${variant}_step_${redundant}_checkpoint.json" <<'PY'
import json, shutil, sys
from pathlib import Path
source, root, target, receipt = map(Path, sys.argv[1:])
resolved = target.resolve()
if not any(resolved.is_relative_to(base.resolve()) for base in (source, root)) or not target.name.startswith("global_step_"):
    raise RuntimeError(f"unsafe checkpoint prune: {target}")
size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file()) if target.exists() else 0
if target.exists(): shutil.rmtree(target)
receipt.write_text(json.dumps({"status":"PRUNED_REDUNDANT_REGULAR_CHECKPOINT","path":str(target),"bytes":size},indent=2)+"\n")
PY
}

evaluate() {
  local variant="$1" step="$2" config model out
  CURRENT_VARIANT="$variant"; CURRENT_TARGET="$step"; set_state EVAL "$variant" "$step"; disk_guard
  [[ "$variant" == baseline ]] && config="$BASELINE_CONFIG" || config="$OFT_CONFIG"
  out="$RUN_ROOT/evaluations/${variant}_step_${step}"
  if [[ ! -f "$out/PARALLEL_EVAL_COMPLETE" ]]; then
    model=$(export_checkpoint "$variant" "$step")
    rm -rf "$out"
    "$ROOT/scripts/run_parallel_vllm_eval_6gpu_20260712_v1.sh" "$config" "$model" "$out" \
      2>&1 | tee "$RUN_ROOT/logs/${variant}_eval_${step}.log"
    [[ -f "$out/PARALLEL_EVAL_COMPLETE" ]]
    ray stop --force >/dev/null 2>&1 || true
  fi
  prune_export "$variant" "$step"
  prune_regular_checkpoint "$variant" "$step"
}

data_order_receipt() {
  local step="$1" baseline_data oft_data output
  baseline_data="$(baseline_checkpoint "$step")/data.pt"
  oft_data="$(oft_checkpoint "$step")/data.pt"
  output="$RUN_ROOT/data_order/oft_vs_baseline_step_${step}.json"
  "$PY" - "$baseline_data" "$oft_data" "$DATA/train.parquet" "$step" "$output" <<'PY'
import hashlib, json, struct, sys
from pathlib import Path
import pyarrow.parquet as pq
import torch

baseline_path, oft_path, parquet_path, step_text, output_path = sys.argv[1:]
step = int(step_text)
def normalize(value):
    if isinstance(value, dict):
        return {str(k): normalize(v) for k,v in sorted(value.items(), key=lambda p:str(p[0])) if str(k)!="_base_seed"}
    if isinstance(value, (list,tuple)): return [normalize(v) for v in value]
    if isinstance(value, torch.Tensor): return value.detach().cpu().tolist()
    if isinstance(value,(str,int,float,bool)) or value is None: return value
    return repr(value)
states = {"baseline": torch.load(baseline_path,map_location="cpu",weights_only=False),
          "oft": torch.load(oft_path,map_location="cpu",weights_only=False)}
norm = {k: normalize(v) for k,v in states.items()}
if norm["baseline"] != norm["oft"]: raise RuntimeError(f"data order mismatch at step {step}")
rows = pq.ParquetFile(parquet_path).metadata.num_rows; batch=504; full=rows//batch
g=torch.Generator().manual_seed(20260707); remain=step; digest=hashlib.sha256()
while remain:
    perm=torch.randperm(rows,generator=g); take=min(remain,full)
    for row in perm[:take*batch].tolist(): digest.update(struct.pack("<q",row))
    remain-=take
payload={"status":"PASS","global_step":step,"variants_normalized_equal":True,
         "explicit_random_sampler_seed":20260707,"batch_size":504,
         "expected_prompt_index_ledger_sha256":digest.hexdigest()}
Path(output_path).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print("DATA_ORDER_RECEIPT_PASS",json.dumps(payload,sort_keys=True))
PY
}

prepare_oft_initial_model() {
  CURRENT_VARIANT=oft; CURRENT_TARGET=0; set_state OFT_EXPORT oft 0
  "$PY" "$ROOT/scripts/prepare_oft_fp32_exact_identity_export.py" \
    --base-model "$BASE_MODEL" --config "$OFT_CONFIG" --output "$OFT_MODEL" --project-root "$ROOT"
  [[ -f "$OFT_MODEL/EXPORT_COMPLETE" ]]; compgen -G "$OFT_MODEL/*.safetensors" >/dev/null
}

preflight_oft() {
  CURRENT_VARIANT=oft; CURRENT_TARGET=0; set_state OFT_PREFLIGHT oft 0
  CUDA_VISIBLE_DEVICES=0 "$PY" "$ROOT/scripts/run_oft_fp32_vllm_preflight.py" \
    --model "$OFT_MODEL" --output "$RUN_ROOT/preflight/oft_fp32_vllm_preflight.json" \
    2>&1 | tee "$RUN_ROOT/logs/oft_fp32_vllm_preflight.log"
  grep -q '"status": "PASS"' "$RUN_ROOT/preflight/oft_fp32_vllm_preflight.json"
  ray stop --force >/dev/null 2>&1 || true
}

freeze_oft_reference() {
  local model receipt
  model=$(export_checkpoint oft 98)
  receipt="$RUN_ROOT/references/oft_step98_reference.json"
  "$PY" - "$model" "$(oft_checkpoint 98)" "$receipt" <<'PY'
import hashlib, json, sys
from pathlib import Path
model, checkpoint, receipt = map(Path, sys.argv[1:])
files=[]
for path in sorted(p for p in model.rglob("*") if p.is_file()):
    files.append({"path":str(path.relative_to(model)),"bytes":path.stat().st_size,
                  "sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
if not any(row["path"].endswith(".safetensors") for row in files): raise RuntimeError("no OFT reference weights")
payload={"status":"FROZEN_OWN_STEP98_REFERENCE_READY","variant":"oft","reference_global_step":98,
         "reference_model_path":str(model.resolve()),"source_checkpoint":str(checkpoint.resolve()),"files":files}
receipt.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print("REFERENCE_POLICY_READY variant=oft step=98")
PY
}

CURRENT_VARIANT=none; CURRENT_TARGET=196; set_state PREFLIGHT none 196; rm -f "$RUN_ROOT/WAVE_FAILED"
[[ $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) -eq 6 ]]
[[ $(sha256sum "$DATA/train.parquet" | awk '{print $1}') == 16c145f165236a292140dd4fb86c4a0c4f7c6241a2390668cbf1d9364ded43d9 ]]
[[ $(sha256sum "$DATA/val.parquet" | awk '{print $1}') == b6a3e2c0538d686258736fc8d0c655b296b52433205bbab2665f89347de6a3b3 ]]
validate_checkpoint "$SOURCE_ROOT/triglu_6x5090_grpo_untunedbase_resume98_to196_seed20260707_v1/checkpoints/global_step_196"
[[ -f "$PRIORITY_ROOT/evaluations/triglu_step_196/PARALLEL_EVAL_COMPLETE" ]]
validate_checkpoint "$(baseline_checkpoint 128)"
[[ -f "$SOURCE_ROOT/evaluations/baseline_step_128/PARALLEL_EVAL_COMPLETE" ]]
third_tracker="$PRIORITY_ROOT/triglu_6x5090_grpo_resume196_to294_cosine5e7_to5e8_seed20260707_v1/checkpoints/latest_checkpointed_iteration.txt"
third_completed=196; [[ -f "$third_tracker" ]] && third_completed=$(tr -dc '0-9' <"$third_tracker")
(( third_completed <= 196 ))
disk_guard

for target in 158 196; do
  train_baseline_to "$target"
  evaluate baseline "$target"
done

prepare_oft_initial_model
preflight_oft
for target in 1 20 98; do
  train_oft_to "$target"
  evaluate oft "$target"
  data_order_receipt "$target"
done
freeze_oft_reference
for target in 128 158 196; do
  train_oft_to "$target"
  evaluate oft "$target"
  data_order_receipt "$target"
done

CURRENT_VARIANT=all; CURRENT_TARGET=196; set_state COMPLETE all 196
touch "$RUN_ROOT/WAVE_COMPLETE"; rm -f "$RUN_ROOT/WAVE_FAILED"; trap - ERR
echo "WAVE_COMPLETE $(date -Is)"
