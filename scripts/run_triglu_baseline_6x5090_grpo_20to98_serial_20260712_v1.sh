#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL=/root/autodl-tmp/verl-v0.6.1-qwenpatch
BASE_MODEL=/root/autodl-tmp/qwen3_single_layer_rl/models/Qwen3-1.7B-Base
TRIGLU_MODEL="$ROOT/runs/runtime_smokes/baseline_triglu_2x5090_grpo_prelaunch_gate_20260712_v1/triglu_exact_noop_export"
DATA="$ROOT/data/numina_math_cot_50k_decontam_v3_verl"
WAVE=triglu_baseline_6x5090_grpo_20to98_serial_20260712_v1
RUN_ROOT="$ROOT/runs/grpo_serial/$WAVE"
PY="$ROOT/envs/vllm0102_verl061/bin/python"
STATE="$RUN_ROOT/state.env"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/evaluations"
export PYTHONPATH="$ROOT/src:$VERL${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_V1=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

set_state() { printf 'PHASE=%q\nVARIANT=%q\nTARGET=%q\nSTART_UNIX=%q\n' "$1" "$2" "$3" "$(date +%s)" >"$STATE.tmp"; mv "$STATE.tmp" "$STATE"; }
disk_guard() { local free; free=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9'); (( free >= 100 )) || { echo "DISK_GUARD_FAIL free=${free}G"; exit 80; }; }
config_for() { echo "$ROOT/configs/runtime/$1_6x5090_grpo_untunedbase_b98_seed20260707_v1.yaml"; }
run_id_for() { echo "$1_6x5090_grpo_untunedbase_b98_seed20260707_v1"; }
model_for() { [[ "$1" == triglu ]] && echo "$TRIGLU_MODEL" || echo "$BASE_MODEL"; }

train_to() {
  local variant="$1" target="$2" config run_id model manifest cmd log tracker completed
  config=$(config_for "$variant"); run_id=$(run_id_for "$variant")
  model=$(model_for "$variant")
  manifest="$RUN_ROOT/${variant}_command_manifest.json"; log="$RUN_ROOT/logs/${variant}_to_${target}.log"
  tracker="$RUN_ROOT/$run_id/checkpoints/latest_checkpointed_iteration.txt"
  completed=0; [[ -f "$tracker" ]] && completed=$(tr -dc '0-9' <"$tracker")
  if (( completed >= target )) && [[ -f "$RUN_ROOT/$run_id/checkpoints/global_step_${target}/actor/fsdp_config.json" ]]; then
    echo "TRAIN_SEGMENT_ALREADY_COMPLETE variant=$variant target=$target checkpoint=$completed"
    return 0
  fi
  disk_guard; set_state TRAIN "$variant" "$target"
  cmd=$($PY -m qwen_single_layer_rl.training.verl_command --config "$config" --project-root "$ROOT" \
    --verl-root "$VERL" --model-path "$model" --data-dir "$DATA" --run-root "$RUN_ROOT" \
    --manifest-out "$manifest" --print-shell)
  echo "TRAIN_SEGMENT_START variant=$variant target=$target $(date -Is)" | tee -a "$log"
  set -o pipefail
  bash -lc "$cmd trainer.total_training_steps=$target" 2>&1 | tee -a "$log"
  local rc=${PIPESTATUS[0]}; (( rc == 0 )) || exit "$rc"
  test -f "$RUN_ROOT/$run_id/checkpoints/global_step_${target}/actor/fsdp_config.json"
  echo "TRAIN_SEGMENT_COMPLETE variant=$variant target=$target $(date -Is)" | tee -a "$log"
  "$PY" - "$RUN_ROOT/$run_id/audits/actor_rank0_model_surgery_audit.json" "$variant" <<'PY'
import json
import sys
d=json.load(open(sys.argv[1])); names=d['trainable_parameter_names']
expected=23 if sys.argv[2]=='triglu' else 11
assert len(names)==expected, (len(names), expected, names)
PY
  ray stop --force >/dev/null 2>&1 || true
}

export_checkpoint() {
  local variant="$1" step="$2" run_id actor merge_actor out config staging
  run_id=$(run_id_for "$variant"); config=$(config_for "$variant")
  actor="$RUN_ROOT/$run_id/checkpoints/global_step_${step}/actor"
  merge_actor="$actor"
  out="$RUN_ROOT/$run_id/exports/global_step_${step}"
  if [[ -f "$out/EXPORT_COMPLETE" ]] && { [[ ! -s "$out/config.json" ]] || ! compgen -G "$out/*.safetensors" >/dev/null; }; then
    echo "INVALID_EXPORT_MARKER_REMOVED variant=$variant step=$step" >&2
    rm -rf "$out"
  fi
  if [[ ! -f "$out/EXPORT_COMPLETE" ]]; then
    rm -rf "$out"; mkdir -p "$(dirname "$out")"
    if [[ "$variant" == triglu ]]; then
      staging="$RUN_ROOT/$run_id/export_staging/global_step_${step}/actor"
      rm -rf "$staging"; mkdir -p "$staging"
      for path in "$actor"/*; do
        [[ "$(basename "$path")" == huggingface ]] && continue
        ln -s "$path" "$staging/$(basename "$path")"
      done
      cp -a "$actor/huggingface" "$staging/huggingface"
      if ! "$PY" - "$staging/huggingface/triglu_hf_model.py" >&2 <<'PY'
import json, sys
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
receipt = {
    "status": "staging_only_repair",
    "source_checkpoint_code_unchanged": True,
    "replacement": "inline_constant_plus_auto_map_wrappers",
}
(path.parent / "export_staging_repair.json").write_text(json.dumps(receipt, indent=2) + "\n")
PY
      then
        echo "EXPORT_STAGING_REPAIR_FAILED variant=$variant step=$step" >&2
        return 1
      fi
      if ! HF_MODULES_CACHE="$staging/.hf_modules_preflight" "$PY" - "$staging/huggingface" >&2 <<'PY'
import json, sys
from pathlib import Path
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
root = Path(sys.argv[1])
config = AutoConfig.from_pretrained(root, trust_remote_code=True)
raw = json.loads((root / "config.json").read_text(encoding="utf-8"))
for key in ("AutoModel", "AutoModelForCausalLM"):
    get_class_from_dynamic_module(raw["auto_map"][key], root)
assert config.model_type == "qwen3_triglu"
print("TRIGLU_EXPORT_STAGING_PREFLIGHT_PASS")
PY
      then
        echo "EXPORT_STAGING_PREFLIGHT_FAILED variant=$variant step=$step" >&2
        return 1
      fi
      merge_actor="$staging"
    fi
    if ! HF_MODULES_CACHE="${staging:-$out/.hf_modules_cache}" "$PY" -m verl.model_merger merge \
      --backend fsdp --trust-remote-code --local_dir "$merge_actor" --target_dir "$out" >&2; then
      echo "EXPORT_MERGE_FAILED variant=$variant step=$step" >&2
      return 1
    fi
    if [[ "$variant" == triglu ]]; then
      if ! "$PY" - "$config" "$out" >&2 <<'PY'
import json, sys
from pathlib import Path
from qwen_single_layer_rl.config import load_config
from qwen_single_layer_rl.vllm.triglu_hf_model import Qwen3TriGLUConfig
cfg=load_config(Path(sys.argv[1])); out=Path(sys.argv[2]); raw=json.loads((out/'config.json').read_text())
for key in ('model_type','architectures','triglu_variant','auto_map'): raw.pop(key,None)
custom=Qwen3TriGLUConfig(triglu_variant=cfg['architecture_variant']['params'], **raw)
custom.save_pretrained(out)
(out/'configuration_qwen3_triglu.py').write_text('from qwen_single_layer_rl.vllm.triglu_hf_model import Qwen3TriGLUConfig\n')
(out/'modeling_qwen3_triglu.py').write_text('from qwen_single_layer_rl.vllm.triglu_hf_model import Qwen3TriGLUForCausalLM, Qwen3TriGLUModel\n')
PY
      then
        echo "EXPORT_CONFIG_PATCH_FAILED variant=$variant step=$step" >&2
        return 1
      fi
    fi
    [[ -s "$out/config.json" ]] || { echo "EXPORT_CONFIG_MISSING variant=$variant step=$step" >&2; return 1; }
    compgen -G "$out/*.safetensors" >/dev/null || { echo "EXPORT_WEIGHTS_MISSING variant=$variant step=$step" >&2; return 1; }
    touch "$out/EXPORT_COMPLETE"
    [[ -n "${staging:-}" ]] && rm -rf "$(dirname "$(dirname "$staging")")"
  fi
  echo "$out"
}

evaluate() {
  local variant="$1" step="$2" config model out
  config=$(config_for "$variant"); set_state EVAL "$variant" "$step"; disk_guard
  if ! model=$(export_checkpoint "$variant" "$step"); then
    echo "EVAL_EXPORT_FAILED variant=$variant step=$step" >&2
    exit 81
  fi
  out="$RUN_ROOT/evaluations/${variant}_step_${step}"
  if [[ -f "$out/PARALLEL_EVAL_COMPLETE" ]]; then
    echo "PARALLEL_EVAL_ALREADY_COMPLETE variant=$variant step=$step"
    return 0
  fi
  rm -rf "$out"
  "$ROOT/scripts/run_parallel_vllm_eval_6gpu_20260712_v1.sh" "$config" "$model" "$out" \
    2>&1 | tee "$RUN_ROOT/logs/${variant}_eval_${step}.log"
  ray stop --force >/dev/null 2>&1 || true
}

set_state PREFLIGHT none 0
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -eq 6
test -s "$BASE_MODEL/model.safetensors.index.json" || test -s "$BASE_MODEL/model.safetensors"
test -s "$TRIGLU_MODEL/config.json"
grep -q 'qwen3_triglu' "$TRIGLU_MODEL/config.json"
test "$(sha256sum "$DATA/train.parquet" | awk '{print $1}')" = 16c145f165236a292140dd4fb86c4a0c4f7c6241a2390668cbf1d9364ded43d9
test "$(sha256sum "$DATA/val.parquet" | awk '{print $1}')" = b6a3e2c0538d686258736fc8d0c655b296b52433205bbab2665f89347de6a3b3
disk_guard
"$PY" - "$ROOT" "$RUN_ROOT/data_order_contract.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
from qwen_single_layer_rl.config import load_config
root, output = Path(sys.argv[1]), Path(sys.argv[2])
configs = {
    name: load_config(root / 'configs/runtime' / f'{name}_6x5090_grpo_untunedbase_b98_seed20260707_v1.yaml')
    for name in ('triglu', 'baseline')
}
def contract(cfg):
    return {
        'train_parquet': str((root / cfg['dataset']['materialized_train']).resolve()),
        'val_parquet': str((root / cfg['dataset']['materialized_val']).resolve()),
        'selection_ledger_sha256': cfg['dataset']['selection_provenance_ledger_sha256'],
        'shuffle': cfg['dataset']['shuffle'],
        'dataloader_seed': cfg['dataset']['dataloader_seed'],
        'train_batch_size': cfg['grpo']['train_batch_size'],
        'ppo_mini_batch_size': cfg['grpo']['ppo_mini_batch_size'],
        'ppo_micro_batch_size': cfg['grpo']['ppo_micro_batch_size'],
        'group_size': cfg['grpo']['group_size'],
        'max_prompt_length': cfg['grpo']['max_prompt_length'],
        'max_response_length': cfg['grpo']['max_response_length'],
    }
observed = {name: contract(cfg) for name, cfg in configs.items()}
assert observed['triglu'] == observed['baseline'], observed
hashes = {}
for key in ('train_parquet', 'val_parquet'):
    path = Path(observed['triglu'][key]); h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''): h.update(block)
    hashes[key] = h.hexdigest()
payload = {'status': 'PASS', 'variants_identical': True, 'contract': observed['triglu'], 'file_sha256': hashes}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
print('DATA_ORDER_CONTRACT_PASS', json.dumps(payload['contract'], sort_keys=True))
PY

# Internal restart canary, followed by the owner-approved alternating schedule.
train_to triglu 1
train_to triglu 20
evaluate triglu 20
train_to baseline 1
train_to baseline 20
evaluate baseline 20
train_to triglu 98
evaluate triglu 98
train_to baseline 98
evaluate baseline 98
set_state COMPLETE all 98
touch "$RUN_ROOT/WAVE_COMPLETE"
echo "WAVE_COMPLETE $(date -Is)"
