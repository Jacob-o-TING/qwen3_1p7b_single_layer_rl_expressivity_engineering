#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="shs_vllm_full_eval_shadow_20260712_v1"
OUT="$ROOT/runs/evaluations/$RUN_ID"
PYTHON="$ROOT/envs/vllm0102_verl061/bin/python"
CONFIG="$ROOT/configs/sft/layer10_whole_layer_shs_sft.yaml"
CHECKPOINT="$ROOT/runs/sft_ordered_20260711_sft50k_v1/layer10_whole_layer_shs/checkpoints/step_00003916"
EXPORT="$ROOT/runs/runtime_smokes/shs_fullmodel_kernel_parity_20260712_v1/deployment_export"
SOURCE_MANIFEST="$ROOT/runs/runtime_smokes/shs_fullmodel_kernel_parity_20260712_v1/manifest.json"

mkdir -p "$OUT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_V1=1
cd "$ROOT"

if [[ -f "$OUT/full_eval_completion_receipt.json" ]]; then
  echo "FULL_EVAL_ALREADY_COMPLETE $OUT"
  exit 0
fi

"$PYTHON" - "$CHECKPOINT" "$EXPORT" "$SOURCE_MANIFEST" "$OUT/preflight_manifest.json" <<'PY'
import hashlib, json, platform, sys
from pathlib import Path
checkpoint, export, source_manifest, output = map(Path, sys.argv[1:])
source = json.loads(source_manifest.read_text())
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()
observed = {
    'checkpoint_trainable': sha(checkpoint / 'trainable_state.pt'),
    'export_model': sha(export / 'model.safetensors'),
}
expected = {
    'checkpoint_trainable': source['source_hashes']['checkpoint_trainable'],
    'export_model': source['export']['hashes']['model.safetensors'],
}
if observed != expected: raise SystemExit(f'hash binding failed: {observed=} {expected=}')
output.write_text(json.dumps({
    'run_id': 'shs_vllm_full_eval_shadow_20260712_v1',
    'protocol_label': 'SHADOW/NEW BACKEND',
    'strict_compatible_claim': False,
    'checkpoint': str(checkpoint), 'export': str(export),
    'hashes': observed, 'source_manifest': str(source_manifest),
    'engine': {'tp': 1, 'model_impl': 'transformers', 'enforce_eager': True,
               'shs_backend': 'reference', 'max_num_seqs': 128,
               'max_num_batched_tokens': 131072, 'gpu_memory_utilization': 0.85},
    'decode': {'seed': 20260707, 'max_tokens': 3072, 'main_temperature': 0.0,
               'amc_repeats': 32, 'amc_temperature': 1.0, 'amc_top_p': 1.0,
               'amc_greedy_repeats': 1},
    'python': platform.python_version(),
}, indent=2, sort_keys=True) + '\n')
PY

if [[ ! -f "$OUT/shell_start_unix.txt" ]]; then
  date +%s > "$OUT/shell_start_unix.txt"
fi
RESUME_ARGS=()
latest_cache() {
  local phase="$1"
  find "$OUT/$phase" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1
}
if MAIN_CACHE="$(latest_cache main)" && [[ -n "$MAIN_CACHE" ]]; then
  RESUME_ARGS+=(--main-use-cache "$MAIN_CACHE")
fi
if AMC_CACHE="$(latest_cache amc_average_at_32)" && [[ -n "$AMC_CACHE" ]]; then
  RESUME_ARGS+=(--amc-use-cache "$AMC_CACHE")
fi
if GREEDY_CACHE="$(latest_cache amc_greedy)" && [[ -n "$GREEDY_CACHE" ]]; then
  RESUME_ARGS+=(--amc-greedy-use-cache "$GREEDY_CACHE")
fi
if [[ -f "$OUT/shs_dispatch_receipts.jsonl" ]]; then
  mv "$OUT/shs_dispatch_receipts.jsonl" \
    "$OUT/shs_dispatch_receipts_attempt_$(date +%s).jsonl"
fi
"$PYTHON" -m qwen_single_layer_rl.eval.run_evalscope \
  --config "$CONFIG" --base-model-only --model-path "$EXPORT" --work-dir "$OUT" \
  --backend vllm --vllm-model-impl transformers --vllm-enforce-eager \
  --shs-backend reference --vllm-gpu-memory-utilization 0.85 \
  --vllm-max-num-seqs 128 --vllm-max-num-batched-tokens 131072 \
  --microbatch-wait-seconds 0.25 \
  --seed 20260707 --max-tokens 3072 --amc-repeats 32 \
  --amc-temperature 1.0 --amc-top-p 1.0 --eval-batch-size 128 \
  "${RESUME_ARGS[@]}"
WALL=$(($(date +%s) - $(<"$OUT/shell_start_unix.txt")))
"$PYTHON" - "$OUT" "$WALL" <<'PY'
import json, sys, time
from pathlib import Path
root, wall = Path(sys.argv[1]), int(sys.argv[2])
phases = {p: json.loads((root / f'{p}_completion_receipt.json').read_text())
          for p in ('main', 'amc', 'amc_greedy')}
summary = json.loads((root / 'model_summary.json').read_text())
receipts = [json.loads(x) for x in (root / 'generation_receipts.jsonl').read_text().splitlines()]
generations = [x for x in receipts if x.get('event') == 'generation_completed']
payload = {'status': 'complete', 'protocol_label': 'SHADOW/NEW BACKEND',
           'strict_compatible_claim': False, 'shell_wall_seconds': wall,
           'phase_wall_seconds': {k: v['elapsed_seconds'] for k, v in phases.items()},
           'generated_tokens': sum(int(x.get('generated_tokens') or 0) for x in generations),
           'rows': len(generations), 'summary': summary, 'completed_unix': time.time()}
(root / 'full_eval_completion_receipt.json').write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
PY
echo "RUN_SHELL_WALL_SECONDS $WALL"
