#!/usr/bin/env bash
set -euo pipefail

ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ID="baseline_triglu_2x5090_grpo_prelaunch_gate_20260712_v1"
OUT="$ROOT/runs/runtime_smokes/$RUN_ID"

echo "=== PRELAUNCH PROCESS ==="
screen -ls | grep -F "qwen_baseline_triglu_2x5090_prelaunch_20260712_v1" || true
echo "=== HUMAN-READABLE PROGRESS ==="
grep -E "RUN_|PRELAUNCH_|Traceback|Error|CUDA out of memory|failed" "$OUT/run.log" 2>/dev/null | tail -80 || true
echo "=== VARIANT RESULTS ==="
for variant in baseline triglu; do
  result="$OUT/$variant/result.json"
  if [[ -f "$result" ]]; then
    "$ROOT/envs/vllm0102_verl061/bin/python" - "$result" <<'PY'
import json
import sys
row = json.load(open(sys.argv[1], encoding="utf-8"))
on_policy = row["on_policy"]["rollout_vs_actor"]
reward = row["reward_audit"]
print(
    f"{row['variant']}: status={row['status']} ready_8gpu={row['ready_for_8gpu_production_shaped_canary']} "
    f"logprob_mean_abs={on_policy['mean_abs']:.6g} logprob_max_abs={on_policy['max_abs']:.6g} "
    f"reward={reward['reward_one']}/{reward['total']} resume_delta={row['resume']['resume_parameter_max_abs_delta']:.6g}"
)
PY
  else
    echo "$variant: pending"
  fi
done
echo "=== GPU ==="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "=== DISK ==="
df -h /root/autodl-tmp
