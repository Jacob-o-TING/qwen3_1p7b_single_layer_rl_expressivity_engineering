#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/envs/vllm0102_verl061/bin/python"
RUN_ID=qwen3_1p7b_gpqa_diamond_freeform126_greedy3072_manual_agent_audit_6x5090_steps158_196_226_256_294_20260718_v1
SCREEN_NAME=qwen_gpqa_free126_manualaudit_gen6_20260718_v1
OUT="$ROOT/runs/freeform_eval/$RUN_ID"
PACKAGE="$OUT/audit_package/gpqa_diamond_freeform126_manual_agent_audit_20260718_v1"

echo "=== Qwen3 GPQA-Diamond-Freeform-126: manual agent audit ==="
if screen -ls 2>/dev/null | grep -Eq "[.]${SCREEN_NAME}[[:space:]]"; then
  echo "controller: running"
else
  echo "controller: stopped"
fi

PHASE=NOT_STARTED CELL=none COMPLETED_CELLS=0 GENERATED_ROWS=0 PACKAGED_ROWS=0 RUN_START_UNIX=0 UPDATED_UNIX=0
[[ ! -f "$OUT/state.env" ]] || source "$OUT/state.env"
now=$(date +%s)
elapsed=0
(( RUN_START_UNIX == 0 )) || elapsed=$((now - RUN_START_UNIX))
printf 'phase: %s | current cell: %s | elapsed: %.2f h\n' \
  "$PHASE" "$CELL" "$(awk -v s="$elapsed" 'BEGIN {print s/3600}')"
[[ ! -f "$OUT/WAVE_FAILED" ]] || echo "ALERT: $(tail -1 "$OUT/WAVE_FAILED")"

"$PY" - "$OUT" "$PACKAGE" "$PHASE" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

out, package = map(Path, sys.argv[1:3])
phase = sys.argv[3]
steps = (158, 196, 226, 256, 294)
variants = ("triglu", "baseline")


def read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, json.JSONDecodeError):
        return {}


complete_cells = sum((out / "generation" / f"{variant}_step{step}" / "CELL_COMPLETE").exists() for step in steps for variant in variants)
complete_shards = sum(1 for marker in (out / "generation").glob("*/shards/rank_*/SHARD_COMPLETE"))
generated_rows = complete_cells * 126 + max(0, complete_shards - complete_cells * 6) * 21
width = 40
fill = min(width, generated_rows * width // 1260)
print(f"generation: [{'#' * fill}{'.' * (width-fill)}] {generated_rows}/1260 rows ({complete_cells}/10 cells)")

timings = []
for path in sorted((out / "timings").glob("*.json"), key=lambda item: item.stat().st_mtime):
    value = read(path).get("wall_seconds")
    if isinstance(value, (int, float)) and value > 0:
        timings.append(float(value))
if timings and complete_cells < 10:
    recent = timings[-3:]
    seconds = sum(recent) / len(recent)
    print(f"recent speed: {seconds/60:.2f} min/cell (latest {len(recent)} mean)")
    print(f"ETA: {(10-complete_cells)*seconds/3600:.2f} h to agent-readable package")
elif complete_cells == 10 and phase in {"PACKAGE", "AWAITING_MANUAL_AUDIT"}:
    print("ETA: generation complete; package ready or packaging")
else:
    print("recent speed / ETA: pending first completed cell")

print("\nGeneration diagnostics (not correctness scores):")
print("  step  architecture  rows     cap_hits  missing_tags  generated_tokens  status")
for step in steps:
    for variant, display in (("triglu", "TriGLU"), ("baseline", "baseline")):
        cell = out / "generation" / f"{variant}_step{step}"
        summary = read(cell / "summary.json")
        if summary:
            print(
                f"  {step:>4}  {display:<12}  {summary.get('rows', 'pending')!s:<7} "
                f"{summary.get('cap_hits', 'pending')!s:<9} {summary.get('missing_answer_tags', 'pending')!s:<13} "
                f"{summary.get('generated_tokens', 'pending')!s:<17} complete"
            )
        else:
            shards = sum(1 for marker in (cell / "shards").glob("rank_*/SHARD_COMPLETE"))
            status = f"partial {shards}/6" if shards else "pending"
            print(f"  {step:>4}  {display:<12}  pending  pending   pending       pending           {status}")

progress = read(package / "progress.json")
summary = read(package / "final_summary.json")
print("\nManual audit:")
if summary:
    print("  step  architecture  correct  incorrect  uncertain  strict_accuracy")
    for step in steps:
        for variant, display in (("triglu", "TriGLU"), ("baseline", "baseline")):
            cell = summary.get("cells", {}).get(f"{variant}_step{step}", {})
            accuracy = cell.get("accuracy_strict")
            score = "pending" if accuracy is None else f"{100*float(accuracy):.3f}%"
            print(
                f"  {step:>4}  {display:<12}  {cell.get('correct', 'pending')!s:<7} "
                f"{cell.get('incorrect', 'pending')!s:<9} {cell.get('uncertain', 'pending')!s:<9} {score}"
            )
    print("  across checkpoints (population std):")
    for variant, value in summary.get("across_checkpoints", {}).items():
        print(
            f"    {variant:<8} mean={100*float(value['mean_accuracy_strict']):.3f}% "
            f"std={100*float(value['population_std_accuracy_strict']):.3f}% n={value['checkpoint_count']}"
        )
else:
    print(
        f"  status={progress.get('status', 'waiting_for_package')} "
        f"verdicts={progress.get('verdict_count', 0)}/1260 "
        f"questions={len(progress.get('completed_question_indices', []))}/126"
    )
    print("  correctness remains pending until complete row-by-row audit; no matcher score is substituted")
PY

echo
echo "GPU utilization / memory:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo "unavailable"
echo "disk free: $(df -BG --output=avail "$ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')G"
