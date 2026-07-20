#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/runs/ood_eval/qwen3_1p7b_ood_6x5090_step294_20260716_v1"
B196="$ROOT/runs/ood_eval/qwen3_1p7b_ood_6x5090_baseline_step196_20260717_v1"
PY="$ROOT/envs/vllm0102_verl061/bin/python"
MODE="${1:-full}"
if [[ "$MODE" != "--embedded" ]]; then
  echo "=== Qwen3 evaluation and OOD monitor ==="
  "$PY" "$ROOT/scripts/summarize_qwen_eval_dashboard.py" "$ROOT" 2>/dev/null || true
fi
echo "=== OOD execution progress ==="
if screen -ls 2>/dev/null | grep -Eq '[.](qwen_ood_step294_20260716_v1|qwen_baseline_step196_ood6_20260717_v1)[[:space:]]'; then
  echo "OOD screen: running"
else
  echo "OOD screen: stopped"
fi
[[ ! -f "$OUT/state.env" ]] || { source "$OUT/state.env"; echo "legacy phase=${PHASE:-unknown} model=${MODEL:-unknown}"; }
[[ ! -f "$B196/state.env" ]] || {
  source "$B196/state.env"
  echo "current phase=${PHASE:-unknown} model=${MODEL:-unknown} benchmark=${BENCHMARK:-unknown} stage=${STAGE_INDEX:-?}/${STAGE_COUNT:-?}"
}
[[ ! -f "$OUT/OOD_FAILED" ]] || echo "ALERT: $(cat "$OUT/OOD_FAILED")"
[[ ! -f "$B196/OOD_FAILED" ]] || echo "ALERT baseline_step196: $(cat "$B196/OOD_FAILED")"
echo "--- baseline_step196 staged detail ---"
echo "  staged six-GPU execution progress:"
cells=(reasoning_gpqa_staged6 reasoning_mmlupro_staged6 code_humanevalplus_staged6 code_mbpp_staged6 language_ceval_staged6 language_ifeval_staged6 language_mgsm_staged6 code_lcb)
complete_cells=0
for cell in "${cells[@]}"; do
  stage="$B196/$cell"
  status=pending
  [[ -f "$stage/RANK_COMPLETE" ]] && { status=complete; complete_cells=$((complete_cells + 1)); }
  detail=""
  if [[ -d "$stage/shards" ]]; then
    detail=$("$PY" - "$stage" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
merged = root / "merge_summary.json"
if merged.exists():
    value = json.loads(merged.read_text(encoding="utf-8"))
    print(
        f"final={100 * float(value['score']):.3f}% "
        f"n={int(value['report_samples'])}/{int(value['expected_identities'])}"
    )
else:
    reports = []
    completed = 0
    for shard in sorted((root / "shards").glob("shard_*")):
        completed += int((shard / "RANK_COMPLETE").exists())
        candidates = []
        for path in shard.glob("main/*/reports/*/*.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                candidates.append((path, int(value["num"]), float(value["score"])))
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
        if candidates:
            reports.append(sorted(candidates, key=lambda item: str(item[0]))[-1])
    samples = sum(value[1] for value in reports)
    if samples:
        score = sum(value[1] * value[2] for value in reports) / samples
        print(
            f"shards={completed}/6 partial={100 * score:.3f}% n={samples} "
            "(not final until exact merge)"
        )
PY
)
    [[ "$status" == pending && -n "$detail" ]] && status=partial
  fi
  progress=""
  if [[ "$status" != complete && -d "$stage" ]]; then
    latest_log=$(find "$stage" -maxdepth 1 -type f -name 'shard_*.log' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)
    if [[ -n "$latest_log" ]]; then
      progress=$(tr '\r' '\n' <"$latest_log" | grep -E 'Evaluating\[|Running\[eval\]|Downloading data:' | tail -1 | sed -E 's/\x1B\[[0-9;?]*[ -\/]*[@-~]//g' | tail -c 150 || true)
    fi
  fi
  printf '  %-24s %-8s %s %s\n' "$cell" "$status" "$detail" "$progress"
done
echo "  completed cells: $complete_cells/${#cells[@]}"
if [[ "$MODE" != "--embedded" ]]; then
  echo "GPU utilization / memory:"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
  echo "disk free: $(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')G"
fi
