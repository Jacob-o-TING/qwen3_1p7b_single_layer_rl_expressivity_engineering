#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/envs/vllm0102_verl061/bin/python"
RUN_ID=qwen3_1p7b_gpqa_diamond_freeform126_greedy3072_qwen3_4b_match_6x5090_steps158_196_226_256_294_20260718_v1
OUT="$ROOT/runs/freeform_eval/$RUN_ID"
MODE="${1:-full}"

if [[ "$MODE" != --embedded ]]; then
  echo "=== Qwen3 GPQA-Diamond-Freeform-126: TriGLU vs baseline ==="
fi
if screen -ls 2>/dev/null | grep -Eq '[.]qwen_gpqa_free126_majorsteps6_20260718_v1[[:space:]]'; then
  echo "controller: running"
else
  echo "controller: stopped"
fi

PHASE=NOT_STARTED CELL=none COMPLETED_CELLS=0 GENERATED_ROWS=0 MATCHED_ROWS=0 RUN_START_UNIX=0 UPDATED_UNIX=0
[[ ! -f "$OUT/state.env" ]] || source "$OUT/state.env"
now=$(date +%s)
elapsed=0
(( RUN_START_UNIX == 0 )) || elapsed=$((now - RUN_START_UNIX))
printf 'phase: %s | current cell: %s | elapsed: %.2f h\n' \
  "$PHASE" "$CELL" "$(awk -v s="$elapsed" 'BEGIN {print s/3600}')"
[[ ! -f "$OUT/WAVE_FAILED" ]] || echo "ALERT: $(tail -1 "$OUT/WAVE_FAILED")"

"$PY" - "$OUT" "$PHASE" "$CELL" "$elapsed" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
phase, current = sys.argv[2], sys.argv[3]
elapsed = int(sys.argv[4])
steps = (158, 196, 226, 256, 294)
variants = ("triglu", "baseline")


def read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


complete_cells = sum((out / "generation" / f"{variant}_step{step}" / "CELL_COMPLETE").exists() for step in steps for variant in variants)
complete_shards = sum(1 for marker in (out / "generation").glob("*/shards/rank_*/SHARD_COMPLETE"))
generated_rows = complete_cells * 126 + max(0, complete_shards - complete_cells * 6) * 21
matched_shards = sum(1 for marker in (out / "matching" / "shards").glob("rank_*/SHARD_COMPLETE"))
matched_rows = 1260 if (out / "matching" / "MATCHING_COMPLETE").exists() else matched_shards * 210
total_units = 1260 + 1260
done_units = generated_rows + matched_rows
width = 40
fill = min(width, done_units * width // total_units)
print(f"overall: [{'#' * fill}{'.' * (width-fill)}] {done_units}/{total_units} rows")
print(f"progress: generation={generated_rows}/1260 ({complete_cells}/10 cells) | matching={matched_rows}/1260 ({matched_shards}/6 shards)")

timing_paths = sorted((out / "timings").glob("*.json"), key=lambda path: path.stat().st_mtime)
timings = [read(path).get("wall_seconds") for path in timing_paths]
timings = [float(value) for value in timings if isinstance(value, (int, float)) and value > 0]
if timings and complete_cells < 10:
    recent = timings[-3:]
    seconds = sum(recent) / len(recent)
    print(f"recent speed: {seconds/60:.2f} min/generation cell (mean of latest {len(recent)})")
    print(f"ETA: generation {(10-complete_cells)*seconds/3600:.2f} h + matcher phase pending")
elif complete_cells == 10 and matched_rows < 1260:
    print("recent speed: all generation cells complete")
    matcher_times = [
        read(path).get("match_seconds")
        for path in (out / "matching" / "shards").glob("rank_*/summary.json")
    ]
    matcher_times = [float(value) for value in matcher_times if isinstance(value, (int, float)) and value > 0]
    if matcher_times:
        print(f"recent matcher shard speed: {sum(matcher_times)/len(matcher_times)/60:.2f} min/210 rows")
        print(f"ETA: up to {max(matcher_times)/60:.2f} min for the concurrent matcher phase")
    else:
        print(f"ETA: matcher {matched_rows}/1260; waiting for first complete matcher timing")
elif phase == "COMPLETE":
    print("ETA: complete")
else:
    print("recent speed: pending first completed generation cell")
    print("ETA: pending first timing signal")

final_summary = read(out / "summary.json")
match_summary = read(out / "matching" / "summary.json")
cells = final_summary.get("cells") or match_summary.get("cells") or {}

print("\nPRIMARY GPQA-Diamond-Freeform-126 paired comparison:")
print("  step  architecture  correct/126  accuracy    delta(TriGLU-baseline)  status")
for step in steps:
    left = cells.get(f"triglu_step{step}")
    right = cells.get(f"baseline_step{step}")
    delta = None if not left or not right else float(left["accuracy"]) - float(right["accuracy"])
    for variant, value in (("TriGLU", left), ("baseline", right)):
        cell_path = out / "generation" / f"{variant.lower()}_step{step}"
        if value:
            score = f"{int(value['correct']):>3}/126"
            accuracy = f"{100*float(value['accuracy']):7.3f}%"
            status = "complete"
        else:
            score = "pending"
            accuracy = "pending"
            status = "generated" if (cell_path / "CELL_COMPLETE").exists() else "pending"
        delta_text = "pending" if delta is None else f"{100*delta:+7.3f} pp"
        print(f"  {step:>4}  {variant:<12}  {score:<11}  {accuracy:<10}  {delta_text:<24}  {status}")

print("\nGeneration / matcher diagnostics:")
print("  step  architecture  cap_hits  missing_tags  matcher_failures  exact_string  generated_tokens")
for step in steps:
    for variant, label in (("triglu", "TriGLU"), ("baseline", "baseline")):
        generated = read(out / "generation" / f"{variant}_step{step}" / "summary.json")
        matched = cells.get(f"{variant}_step{step}", {})
        def field(value, key):
            return str(value[key]) if key in value else "pending"
        print(
            f"  {step:>4}  {label:<12}  {field(generated, 'cap_hits'):<8}  "
            f"{field(generated, 'missing_answer_tags'):<12}  {field(matched, 'matcher_failures'):<16}  "
            f"{field(matched, 'normalized_exact_matches'):<12}  {field(generated, 'generated_tokens')}"
        )

audit = final_summary.get("audit_queue_rows")
print(f"\nhuman audit queue: {'pending' if audit is None else audit} rows")
print("note: official MCQ GPQA-Diamond remains separate; no hard cross-protocol average is reported")
PY

if [[ "$MODE" != --embedded ]]; then
  echo
  echo "GPU utilization / memory:"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo "unavailable"
  echo "disk free: $(df -BG --output=avail "$ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')G"
  if [[ -f "$OUT/logs/controller.log" ]]; then
    recent=$(tail -n 250 "$OUT/logs/controller.log" | grep -E 'Traceback|OutOfMemory|CUDA error|WAVE_FAILED|FREEFORM_.*FAIL' | tail -1 || true)
    [[ -z "$recent" ]] || echo "recent actionable error: $recent"
  fi
fi
