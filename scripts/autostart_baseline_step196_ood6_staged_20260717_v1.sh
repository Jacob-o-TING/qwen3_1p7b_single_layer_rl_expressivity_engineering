#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEGACY_OUT="$ROOT/runs/ood_eval/qwen3_1p7b_ood_6x5090_step294_20260716_v1"
RUN_ID=qwen3_1p7b_ood_6x5090_baseline_step196_20260717_v1
OUT="$ROOT/runs/ood_eval/$RUN_ID"
BARRIER="$LEGACY_OUT/PRE_BASELINE_OOD_COMPLETE"
RUNNER="$ROOT/scripts/run_baseline_step196_staged6_ood_20260717_v1.sh"
mkdir -p "$OUT/logs"
exec > >(tee -a "$OUT/logs/autostart.log") 2>&1

fail_safe_exit() {
  local rc=$?
  trap - EXIT HUP INT TERM
  if (( rc != 0 )); then
    echo "BASELINE_STEP196_AUTOSTART_FAILED rc=$rc time=$(date -Is)"
    printf 'rc=%s time=%s\n' "$rc" "$(date -Is)" >"$OUT/AUTOSTART_FAILED"
  fi
  exit "$rc"
}
trap fail_safe_exit EXIT HUP INT TERM

echo "BASELINE_STEP196_AUTOSTART_ARMED $(date -Is)"
while [[ ! -f "$LEGACY_OUT/untuned_base/PARALLEL_OOD_EVAL_COMPLETE" ]]; do
  sleep 3
done
echo "UNTUNED_BASE_OOD_COMPLETE $(date -Is)"

deadline=$(( $(date +%s) + 3600 ))
while pgrep -f '[r]un_parallel_ood_vllm_eval_6gpu_20260716_v1.*untuned_base' >/dev/null; do
  (( $(date +%s) < deadline )) || { echo "LEGACY_OOD_EXIT_TIMEOUT"; exit 81; }
  sleep 3
done
ray stop --force >/dev/null 2>&1 || true
while ! nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | \
    awk '$1 > 1024 {busy=1} END {exit busy}'; do
  (( $(date +%s) < deadline )) || { echo "GPU_IDLE_TIMEOUT"; exit 82; }
  sleep 5
done
echo "BASELINE_STEP196_GPU_HANDOFF_READY $(date -Is)"

bash "$RUNNER"
[[ -f "$OUT/OOD_COMPLETE" ]]
if [[ -d "$BARRIER" ]]; then
  rmdir "$BARRIER"
fi
touch "$BARRIER" "$LEGACY_OUT/BASELINE_STEP196_OOD_COMPLETE"
rm -f "$OUT/AUTOSTART_FAILED"
trap - EXIT HUP INT TERM
echo "BASELINE_STEP196_BARRIER_RELEASED $(date -Is)"
