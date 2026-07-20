#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL=/root/autodl-tmp/verl-v0.6.1-qwenpatch
PY="$ROOT/envs/vllm0102_verl061/bin/python"
RUN_ID=qwen3_1p7b_other_eval_6x5090_triglu_baseline_steps158_196_226_256_294_20260718_v1
OUT="$ROOT/runs/ood_eval/$RUN_ID"
SOURCE_ROOT="$ROOT/runs/grpo_interleaved/triglu_baseline_6x5090_grpo_retro30_60_then_98to196_interleaved_20260714_v1"
PRIORITY_ROOT="$ROOT/runs/grpo_priority/triglu_priority_6x5090_grpo_step128_to294_then_baseline128_to196_20260715_v1"
TRIGLU_TEMPLATE="$ROOT/runs/runtime_smokes/baseline_triglu_2x5090_grpo_prelaunch_gate_20260712_v1/triglu_exact_noop_export"
BASE_MODEL=/root/autodl-tmp/qwen3_single_layer_rl/models/Qwen3-1.7B-Base
HEPLUS_LEDGER_SOURCE="$ROOT/runs/ood_eval/qwen3_1p7b_ood_6x5090_step294_20260716_v1/untuned_base/code_heplus_mbpp/main/20260717_191648/predictions/qwen3-1p7b-single-layer-sft/humaneval_plus_default.jsonl"
STATE="$OUT/state.env"
CURRENT_CELL=none
CURRENT_BENCHMARK=none
CURRENT_STAGE=0
RUN_START_UNIX=""

mkdir -p "$OUT"/{cells,exports,export_staging,imports,logs,receipts,timings}
exec > >(tee -a "$OUT/logs/controller.log") 2>&1
exec 9>"$OUT/controller.lock"
flock -n 9 || { echo "OTHER_EVAL_CONTROLLER_ALREADY_RUNNING"; exit 9; }

export PYTHONPATH="$ROOT/src:$VERL${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_V1=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export TOKENIZERS_PARALLELISM=false

if [[ -s "$OUT/run_start_unix.txt" ]]; then
  RUN_START_UNIX=$(tr -dc '0-9' <"$OUT/run_start_unix.txt")
else
  RUN_START_UNIX=$(date +%s)
  printf '%s\n' "$RUN_START_UNIX" >"$OUT/run_start_unix.txt"
fi

write_state() {
  local phase="$1" cell="$2" benchmark="$3" stage="$4"
  printf 'PHASE=%s\nCELL=%s\nBENCHMARK=%s\nSTAGE_INDEX=%s\nSTAGE_COUNT=9\nRUN_START_UNIX=%s\nUPDATED_UNIX=%s\n' \
    "$phase" "$cell" "$benchmark" "$stage" "$RUN_START_UNIX" "$(date +%s)" >"$STATE.tmp"
  mv "$STATE.tmp" "$STATE"
}

old_root_mode=$(stat -c %a /root)
restore_root_mode() { chmod "$old_root_mode" /root 2>/dev/null || true; }
on_exit() {
  local rc=$?
  restore_root_mode
  if (( rc != 0 )); then
    write_state FAILED "$CURRENT_CELL" "$CURRENT_BENCHMARK" "$CURRENT_STAGE"
    printf 'WAVE_FAILED rc=%s cell=%s benchmark=%s stage=%s time=%s\n' \
      "$rc" "$CURRENT_CELL" "$CURRENT_BENCHMARK" "$CURRENT_STAGE" "$(date -Is)" \
      | tee "$OUT/WAVE_FAILED"
  fi
  exit "$rc"
}
trap on_exit EXIT INT TERM
chmod 701 /root

disk_guard() {
  local free
  free=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
  (( free >= 40 )) || { echo "OTHER_EVAL_DISK_GUARD_FAIL free=${free}G required=40G"; return 80; }
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

validate_checkpoint() {
  local checkpoint="$1" actor="$1/actor"
  [[ -s "$checkpoint/data.pt" ]]
  [[ -s "$actor/fsdp_config.json" ]]
  [[ $(find "$actor" -maxdepth 1 -type f -name 'model_world_size_6_rank_*.pt' | wc -l) -eq 6 ]]
  [[ $(find "$actor" -maxdepth 1 -type f -name 'optim_world_size_6_rank_*.pt' | wc -l) -eq 6 ]]
  [[ $(find "$actor" -maxdepth 1 -type f -name 'extra_state_world_size_6_rank_*.pt' | wc -l) -eq 6 ]]
}

existing_export() {
  local variant="$1" step="$2" path
  path="$PRIORITY_ROOT/exports/${variant}_step_${step}"
  if [[ -f "$path/EXPORT_COMPLETE" && -s "$path/config.json" ]] && compgen -G "$path/*.safetensors" >/dev/null; then
    echo "$path"
    return 0
  fi
  return 1
}

export_checkpoint() {
  local variant="$1" step="$2" label
  label="${variant}_step${step}"
  local checkpoint actor merge_actor config out staging="" reused=""
  if reused=$(existing_export "$variant" "$step"); then
    "$PY" - "$variant" "$step" "$reused" "$OUT/receipts/${label}_export.json" <<'PY'
import json
import sys
from pathlib import Path

variant, step, model, receipt = sys.argv[1:]
model = Path(model).resolve()
payload = {
    "status": "REUSED_EXISTING_EXPORT",
    "variant": variant,
    "global_step": int(step),
    "export_path": str(model),
    "config_present": (model / "config.json").is_file(),
    "safetensors": [path.name for path in sorted(model.glob("*.safetensors"))],
}
Path(receipt).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
    echo "$reused"
    return 0
  fi
  checkpoint=$(checkpoint_for "$variant" "$step")
  validate_checkpoint "$checkpoint"
  actor="$checkpoint/actor"
  merge_actor="$actor"
  config=$(config_for "$variant" "$step")
  out="$OUT/exports/$label"
  if [[ -f "$out/EXPORT_COMPLETE" && -s "$out/config.json" ]] && compgen -G "$out/*.safetensors" >/dev/null; then
    echo "$out"
    return 0
  fi

  rm -rf "$out"
  if [[ "$variant" == triglu ]]; then
    staging="$OUT/export_staging/$label/actor"
    rm -rf "$(dirname "$staging")"
    mkdir -p "$staging"
    for path in "$actor"/*; do
      [[ "$(basename "$path")" == huggingface ]] && continue
      ln -s "$path" "$staging/$(basename "$path")"
    done
    cp -a "$actor/huggingface" "$staging/huggingface"
    "$PY" - "$staging/huggingface/triglu_hf_model.py" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
needle = "from . import TRIGLU_ARCHITECTURE"
if text.count(needle) != 1:
    raise RuntimeError(f"expected one package constant import in {path}")
path.write_text(text.replace(needle, 'TRIGLU_ARCHITECTURE = "Qwen3TriGLUForCausalLM"'), encoding="utf-8")
root = path.parent
(root / "configuration_qwen3_triglu.py").write_text(
    "from qwen_single_layer_rl.vllm.triglu_hf_model import Qwen3TriGLUConfig\n", encoding="utf-8"
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
    HF_MODULES_CACHE="$staging/.hf_modules_preflight" "$PY" - "$staging/huggingface" <<'PY'
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
    --backend fsdp --trust-remote-code \
    --local_dir "$merge_actor" --target_dir "$out" >&2
  if [[ "$variant" == triglu ]]; then
    "$PY" - "$config" "$out" <<'PY'
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
  [[ -z "$staging" ]] || rm -rf "$(dirname "$staging")"
  "$PY" - "$variant" "$step" "$checkpoint" "$out" "$OUT/receipts/${label}_export.json" <<'PY'
import json
import sys
from pathlib import Path

variant, step, checkpoint, out, receipt = sys.argv[1:]
root = Path(out)
files = sorted(path for path in root.iterdir() if path.is_file())
payload = {
    "status": "EXPORT_COMPLETE",
    "variant": variant,
    "global_step": int(step),
    "source_checkpoint": str(Path(checkpoint).resolve()),
    "export_path": str(root.resolve()),
    "files": [{"name": path.name, "bytes": path.stat().st_size} for path in files],
}
Path(receipt).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  echo "$out"
}

write_state_for_stage() {
  CURRENT_BENCHMARK="$1"
  CURRENT_STAGE="$2"
  write_state EVAL "$CURRENT_CELL" "$CURRENT_BENCHMARK" "$CURRENT_STAGE"
}

run_stage() {
  local model="$1" config="$2" cell="$3" stage_index="$4" stage_name="$5"
  local dataset="$6" expected="$7" sandbox="$8" parser_v2="$9"
  local stage="$cell/$stage_name"
  if [[ -f "$stage/RANK_COMPLETE" ]]; then
    echo "STAGED6_ALREADY_COMPLETE cell=$CURRENT_CELL dataset=$dataset"
    return 0
  fi
  mkdir -p "$stage/shards"
  write_state_for_stage "$dataset" "$stage_index"
  echo "STAGED6_START cell=$CURRENT_CELL stage=$stage_index/9 dataset=$dataset expected=$expected time=$(date -Is)"

  run_shard() {
    local gpu="$1" shard_index="$2" shard shard_out latest
    shard=$(printf 'shard_%02d' "$shard_index")
    shard_out="$stage/shards/$shard"
    mkdir -p "$shard_out"
    if [[ -f "$shard_out/RANK_COMPLETE" ]]; then
      echo "STAGED6_SHARD_ALREADY_COMPLETE cell=$CURRENT_CELL dataset=$dataset shard=$shard"
      return 0
    fi
    local cache_args=() sandbox_args=() parser_args=()
    latest=$(find "$shard_out/main" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | sort | tail -1 || true)
    [[ -z "$latest" ]] || cache_args+=(--main-use-cache "$shard_out/main/$latest")
    [[ "$sandbox" == yes ]] && sandbox_args+=(--local-code-sandbox)
    [[ "$parser_v2" == yes ]] && parser_args+=(--humanevalplus-parser-v2)
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m qwen_single_layer_rl.eval.run_evalscope \
      --config "$config" --base-model-only --model-path "$model" --work-dir "$shard_out" \
      --backend vllm --vllm-model-impl auto --vllm-enforce-eager \
      --vllm-gpu-memory-utilization 0.85 --vllm-max-num-seqs 128 \
      --vllm-max-num-batched-tokens 32768 --microbatch-wait-seconds 0.25 \
      --seed 20260707 --max-tokens 3072 --eval-batch-size 128 \
      --amc-repeats 0 --skip-amc-greedy --datasets "$dataset" \
      --dataset-shard-count 6 --dataset-shard-index "$shard_index" \
      "${cache_args[@]}" "${sandbox_args[@]}" "${parser_args[@]}" \
      >"$stage/$shard.log" 2>&1
    touch "$shard_out/RANK_COMPLETE"
  }

  local pids=() status=0
  for gpu in 0 1 2 3 4 5; do
    run_shard "$gpu" "$gpu" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
  (( status == 0 )) || { echo "STAGED6_FAILED cell=$CURRENT_CELL dataset=$dataset status=$status"; return "$status"; }
  "$PY" "$ROOT/scripts/merge_generic_evalscope_shards.py" "$stage" \
    --dataset "$dataset" --shard-count 6 --expected-identities "$expected" | tee "$stage/merge.log"
  touch "$stage/RANK_COMPLETE"
  ray stop --force >/dev/null 2>&1 || true
  echo "STAGED6_COMPLETE cell=$CURRENT_CELL dataset=$dataset time=$(date -Is)"
}

run_primary_humanevalplus() {
  local variant="$1" step="$2" model="$3" cell="$4"
  local label heplus
  label="${variant}_step${step}"
  heplus="$cell/primary_humanevalplus"
  local generated_config="$cell/primary_humanevalplus_config.yaml"
  if [[ -f "$heplus/FULL164_COMPLETE" ]]; then
    echo "PRIMARY_HEPLUS_ALREADY_COMPLETE cell=$label"
    return 0
  fi
  write_state_for_stage primary_humanevalplus 9
  mkdir -p "$heplus"
  "$PY" - "$ROOT" "$label" "$variant" "$model" "$heplus" "$HEPLUS_LEDGER_SOURCE" "$generated_config" <<'PY'
import sys
from pathlib import Path
import yaml

root, label, variant, model, output, source, config_path = sys.argv[1:]
payload = {
    "run_id": f"qwen3_1p7b_other_eval_{label}_heplus_nochat_20260718_v1",
    "model_label": label,
    "protocol": "humanevalplus_evalscope_raw_instruction_nochat_full164",
    "completion_marker": "FULL164_COMPLETE",
    "model_path": str(Path(model).resolve()),
    "source_prediction_jsonl": str(Path(source).resolve()),
    "output_root": str(Path(output).resolve()),
    "expected_task_ledger_sha256": "7bb4ed06a3a4c725a9893f38e76087c9d6bf2c3caa8d0c880e061ffecc0a1baa",
    "cells": ["evalscope_raw_instruction_nochat"],
    "seed": 20260707,
    "sample_count": 164,
    "max_tokens": 3072,
    "sandbox_timeout_seconds": 300,
    "sandbox_scratch_root": f"/tmp/qwen_other_{label}_heplus",
    "vllm": {
        "gpu_memory_utilization": 0.85,
        "enforce_eager": True,
        "max_num_seqs": 64,
        "max_num_batched_tokens": 32768,
    },
}
if variant == "triglu":
    payload["vllm_plugin"] = "triglu"
Path(config_path).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
PY
  "$PY" "$ROOT/scripts/run_humanevalplus_prompt_protocol_matrix.py" \
    --root "$ROOT" --config "$generated_config" prepare
  local pids=() status=0
  for gpu in 0 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/scripts/run_humanevalplus_prompt_protocol_matrix.py" \
      --root "$ROOT" --config "$generated_config" worker \
      --cell evalscope_raw_instruction_nochat --shard-index "$gpu" --shard-count 6 \
      >"$heplus/evalscope_raw_instruction_nochat.shard${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || status=$?; done
  (( status == 0 )) || return "$status"
  "$PY" "$ROOT/scripts/run_humanevalplus_prompt_protocol_matrix.py" \
    --root "$ROOT" --config "$generated_config" merge --shard-count 6
  [[ -f "$heplus/FULL164_COMPLETE" && -s "$heplus/summary.json" ]]
  ray stop --force >/dev/null 2>&1 || true
}

prune_temporary_export() {
  local model="$1" label="$2" receipt
  receipt="$OUT/receipts/${label}_export_pruned.json"
  [[ "$model" == "$OUT/exports/"* ]] || return 0
  "$PY" - "$OUT" "$model" "$receipt" <<'PY'
import json
import shutil
import sys
from pathlib import Path

run_root, target, receipt = map(Path, sys.argv[1:])
resolved = target.resolve()
if not resolved.is_relative_to((run_root / "exports").resolve()):
    raise RuntimeError(f"refusing to prune outside temporary exports: {resolved}")
size = sum(path.stat().st_size for path in target.rglob("*") if path.is_file()) if target.exists() else 0
if target.exists():
    shutil.rmtree(target)
receipt.write_text(json.dumps({"status": "PRUNED", "path": str(resolved), "bytes": size}, indent=2) + "\n")
PY
}

run_cell() {
  local variant="$1" step="$2" label cell
  label="${variant}_step${step}"
  cell="$OUT/cells/${variant}_step_${step}"
  local config model started finished
  CURRENT_CELL="$label"
  CURRENT_BENCHMARK=prepare
  CURRENT_STAGE=0
  if [[ -f "$cell/CELL_COMPLETE" ]]; then
    echo "OTHER_EVAL_CELL_ALREADY_COMPLETE cell=$label"
    return 0
  fi
  disk_guard
  mkdir -p "$cell"
  started=$(date +%s)
  write_state PREPARE "$label" export 0
  config=$(config_for "$variant" "$step")
  model=$(export_checkpoint "$variant" "$step")
  [[ -s "$model/config.json" ]]
  compgen -G "$model/*.safetensors" >/dev/null

  run_stage "$model" "$config" "$cell" 1 reasoning_gpqa_staged6 gpqa_diamond 198 no no
  run_stage "$model" "$config" "$cell" 2 reasoning_mmlupro_staged6 mmlu_pro 12032 no no
  run_stage "$model" "$config" "$cell" 3 code_humanevalplus_staged6 humaneval_plus 164 yes yes
  run_stage "$model" "$config" "$cell" 4 code_mbpp_staged6 mbpp 500 yes no
  run_stage "$model" "$config" "$cell" 5 language_ceval_staged6 ceval 1346 no no
  run_stage "$model" "$config" "$cell" 6 language_ifeval_staged6 ifeval 541 no no
  run_stage "$model" "$config" "$cell" 7 language_mgsm_staged6 mgsm 2750 no no

  write_state_for_stage live_code_bench 8
  "$ROOT/scripts/run_livecodebench_parallel6_vllm_20260717_v1.sh" "$config" "$model" "$cell"
  touch "$cell/PARALLEL_OOD_EVAL_COMPLETE"
  run_primary_humanevalplus "$variant" "$step" "$model" "$cell"
  "$PY" "$ROOT/scripts/summarize_ood_eval.py" "$cell" --project-root "$ROOT" \
    --model-label "$label" --json-out "$cell/summary.json" | tee "$cell/summary.txt"
  touch "$cell/CELL_COMPLETE"
  finished=$(date +%s)
  "$PY" - "$label" "$started" "$finished" "$OUT/timings/${label}.json" <<'PY'
import json
import sys
from pathlib import Path

label, started, finished, output = sys.argv[1:]
Path(output).write_text(
    json.dumps(
        {"cell": label, "started_unix": int(started), "finished_unix": int(finished), "wall_seconds": int(finished) - int(started)},
        indent=2,
    ) + "\n",
    encoding="utf-8",
)
PY
  prune_temporary_export "$model" "$label"
  echo "OTHER_EVAL_CELL_COMPLETE cell=$label wall_seconds=$((finished-started)) time=$(date -Is)"
}

import_cell() {
  local label="$1" source="$2" corrected="$3" link receipt
  link="$OUT/cells/${label/_step/_step_}"
  receipt="$OUT/imports/${label}.json"
  [[ -f "$source/PARALLEL_OOD_EVAL_COMPLETE" ]]
  [[ -s "$source/summary.json" ]]
  [[ -f "$corrected/FULL164_COMPLETE" && -s "$corrected/summary.json" ]]
  if [[ -L "$link" ]]; then
    [[ "$(readlink -f "$link")" == "$(readlink -f "$source")" ]]
  elif [[ -e "$link" ]]; then
    echo "IMPORT_LINK_COLLISION label=$label path=$link" >&2
    return 73
  else
    ln -s "$source" "$link"
  fi
  "$PY" - "$label" "$source" "$corrected" "$receipt" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

label, source, corrected, receipt = sys.argv[1:]
source = Path(source).resolve()
corrected = Path(corrected).resolve()
def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
payload = {
    "status": "IMPORTED_COMPLETE",
    "label": label,
    "source_root": str(source),
    "source_summary_sha256": sha(source / "summary.json"),
    "corrected_humanevalplus_root": str(corrected),
    "corrected_humanevalplus_summary_sha256": sha(corrected / "summary.json"),
    "predictions_regenerated": False,
}
Path(receipt).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

write_state PREFLIGHT none validate 0
rm -f "$OUT/WAVE_FAILED" "$OUT/WAVE_COMPLETE"
[[ -x "$PY" ]]
[[ -s "$HEPLUS_LEDGER_SOURCE" ]]
[[ $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) -eq 6 ]]
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; then
  echo "OTHER_EVAL_GPU_BUSY_REFUSING_TO_CONTEND"
  exit 70
fi
disk_guard
for step in 158 196 226 256 294; do
  validate_checkpoint "$(checkpoint_for triglu "$step")"
  validate_checkpoint "$(checkpoint_for baseline "$step")"
done

import_cell triglu_step294 \
  "$ROOT/runs/ood_eval/qwen3_1p7b_ood_6x5090_step294_20260716_v1/triglu" \
  "$ROOT/runs/eval_protocol/qwen3_1p7b_heplus_prompt_matrix_full164_20260717_v1/models/triglu_step294"
import_cell baseline_step196 \
  "$ROOT/runs/ood_eval/qwen3_1p7b_ood_6x5090_baseline_step196_20260717_v1" \
  "$ROOT/runs/eval_protocol/qwen3_1p7b_heplus_prompt_matrix_full164_20260717_v1/models/baseline_step196"

run_cell triglu 158
run_cell baseline 158
run_cell triglu 196
run_cell triglu 226
run_cell baseline 226
run_cell triglu 256
run_cell baseline 256
run_cell baseline 294

CURRENT_CELL=all
CURRENT_BENCHMARK=complete
CURRENT_STAGE=9
write_state COMPLETE all all 9
touch "$OUT/WAVE_COMPLETE"
rm -f "$OUT/WAVE_FAILED"
bash "$ROOT/scripts/monitor_qwen3_1p7b_other_eval_majorsteps_6x5090_20260718_v1.sh" --embedded \
  >"$OUT/final_dashboard.txt"
trap - EXIT INT TERM
restore_root_mode
echo "OTHER_EVAL_WAVE_COMPLETE run_id=$RUN_ID time=$(date -Is)"
