#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/envs/vllm0102_verl061/bin/python"
RUN_ID=qwen3_1p7b_other_eval_6x5090_triglu_baseline_steps158_196_226_256_294_20260718_v1
OUT="$ROOT/runs/ood_eval/$RUN_ID"
FREEFORM_RUN_ID=qwen3_1p7b_gpqa_diamond_freeform126_greedy3072_manual_agent_audit_6x5090_steps158_196_226_256_294_20260718_v1
FREEFORM_OUT="$ROOT/runs/freeform_eval/$FREEFORM_RUN_ID"
FREEFORM_SUMMARY="$FREEFORM_OUT/audit_package/gpqa_diamond_freeform126_manual_agent_audit_20260718_v1/final_summary.json"
MODE="${1:-full}"

if [[ "$MODE" != --embedded ]]; then
  echo "=== Qwen3 major-checkpoint Other eval: TriGLU vs baseline ==="
fi
if screen -ls 2>/dev/null | grep -Eq '[.]qwen_other_majorsteps6_20260718_v1[[:space:]]'; then
  echo "controller: running"
else
  echo "controller: stopped"
fi

PHASE=not_started CELL=none BENCHMARK=none STAGE_INDEX=0 STAGE_COUNT=9 RUN_START_UNIX=0 UPDATED_UNIX=0
[[ ! -f "$OUT/state.env" ]] || source "$OUT/state.env"
now=$(date +%s)
elapsed=0
(( RUN_START_UNIX == 0 )) || elapsed=$((now - RUN_START_UNIX))
BENCHMARK_DISPLAY="$BENCHMARK"
[[ "$BENCHMARK" != gpqa_diamond ]] || BENCHMARK_DISPLAY=GPQA-Diamond
printf 'phase: %s | current cell: %s | benchmark: %s | stage: %s/%s | elapsed: %.2f h\n' \
  "$PHASE" "$CELL" "$BENCHMARK_DISPLAY" "$STAGE_INDEX" "$STAGE_COUNT" "$(awk -v s="$elapsed" 'BEGIN {print s/3600}')"
if [[ -f "$FREEFORM_OUT/state.env" ]]; then
  (
    FREEFORM_PHASE=NOT_STARTED FREEFORM_CELL=none
    source "$FREEFORM_OUT/state.env"
    printf 'GPQA-Freeform queue: phase=%s | current cell=%s\n' "$PHASE" "$CELL"
  )
elif screen -ls 2>/dev/null | grep -Eq '[.]qwen_gpqa_free126_manualaudit_gen6_20260718_v1[[:space:]]'; then
  echo "GPQA-Freeform queue: manual-audit generation controller starting"
else
  echo "GPQA-Freeform queue: not started"
fi

if [[ -f "$OUT/WAVE_FAILED" ]]; then
  echo "ALERT: $(tail -1 "$OUT/WAVE_FAILED")"
fi

"$PY" - "$ROOT" "$OUT" "$CELL" "$BENCHMARK" "$STAGE_INDEX" "$FREEFORM_OUT" "$FREEFORM_SUMMARY" <<'PY'
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
out = Path(sys.argv[2])
current_cell = sys.argv[3]
benchmark = sys.argv[4]
stage_index = int(sys.argv[5])
freeform_out = Path(sys.argv[6])
freeform_summary_path = Path(sys.argv[7])
sys.path.insert(0, str(root / "scripts"))
from summarize_ood_eval import build_summary  # noqa: E402

steps = (158, 196, 226, 256, 294)
variants = ("triglu", "baseline")
expected = {
    "gpqa_diamond": 198,
    "mmlu_pro": 12032,
    "humaneval_plus": 164,
    "mbpp": 500,
    "ceval": 1346,
    "ifeval": 541,
    "mgsm": 2750,
    "live_code_bench": 1055,
    "primary_humanevalplus": 164,
}
stage_names = {
    "gpqa_diamond": "reasoning_gpqa_staged6",
    "mmlu_pro": "reasoning_mmlupro_staged6",
    "humaneval_plus": "code_humanevalplus_staged6",
    "mbpp": "code_mbpp_staged6",
    "ceval": "language_ceval_staged6",
    "ifeval": "language_ifeval_staged6",
    "mgsm": "language_mgsm_staged6",
}


def cell_dir(variant: str, step: int) -> Path:
    return out / "cells" / f"{variant}_step_{step}"


def imported(variant: str, step: int) -> bool:
    return (out / "imports" / f"{variant}_step{step}.json").exists()


def status(variant: str, step: int, path: Path) -> str:
    if imported(variant, step):
        return "complete/import"
    if (path / "CELL_COMPLETE").exists():
        return "complete"
    if path.exists() and any(path.iterdir()):
        return "partial"
    return "pending"


def pct(value: float | None) -> str:
    return "pending" if value is None else f"{100.0 * value:.3f}%"


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, json.JSONDecodeError):
        return {}


freeform_summary = read_json(freeform_summary_path)
freeform_cells = freeform_summary.get("cells", {})


def summary_for(variant: str, step: int) -> dict:
    path = cell_dir(variant, step)
    if not path.exists():
        return build_summary(path, project_root=root, model_label=f"{variant}_step{step}")
    return build_summary(path, project_root=root, model_label=f"{variant}_step{step}")


def stage_progress() -> None:
    if current_cell in {"none", "all"} or stage_index <= 0:
        return
    try:
        variant, step_text = current_cell.split("_step", 1)
        step = int(step_text)
    except ValueError:
        return
    path = cell_dir(variant, step)
    print("current-stage detail:")
    if benchmark in stage_names:
        stage = path / stage_names[benchmark]
        shards = sum((item / "RANK_COMPLETE").exists() for item in (stage / "shards").glob("shard_*"))
        merged = stage / "merge_summary.json"
        if merged.exists():
            value = json.loads(merged.read_text(encoding="utf-8"))
            print(
                f"  shards=6/6 final={100.0 * float(value['score']):.3f}% "
                f"n={int(value['report_samples'])}/{int(value['expected_identities'])}"
            )
        else:
            samples = 0
            weighted = 0.0
            for report in (stage / "shards").glob("shard_*/main/*/reports/*/*.json"):
                try:
                    value = json.loads(report.read_text(encoding="utf-8"))
                    n = int(value["num"])
                    score = float(value["score"])
                except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
                samples += n
                weighted += n * score
            partial = "pending" if not samples else f"{100.0 * weighted / samples:.3f}%"
            print(
                f"  shards={shards}/6 partial={partial} n={samples}/{expected[benchmark]} "
                "(not final until exact merge)"
            )
    elif benchmark == "live_code_bench":
        lcb = path / "code_lcb_parallel6_release_latest_20260717_v1"
        workers = sum(1 for marker in lcb.rglob("RANK_COMPLETE")) if lcb.exists() else 0
        print(f"  LiveCodeBench completed shard markers={workers}/6 expected identities=1055")
    elif benchmark == "primary_humanevalplus":
        heplus = path / "primary_humanevalplus"
        workers = sum(1 for marker in (heplus / "workers").glob("*/WORKER_COMPLETE"))
        if (heplus / "summary.json").exists():
            value = json.loads((heplus / "summary.json").read_text(encoding="utf-8"))
            cell = value["cells"]["evalscope_raw_instruction_nochat"]
            print(f"  workers=6/6 final={cell['passed']}/{cell['rows']} ({100.0 * cell['score']:.3f}%)")
        else:
            print(f"  workers={workers}/6 n=pending/164 (not final until exact merge)")


stage_progress()

rows = []
for step in steps:
    for variant in variants:
        path = cell_dir(variant, step)
        summary = summary_for(variant, step)
        comparison = summary["parser_sensitive_code_comparison"]
        benchmarks = summary["benchmarks"]
        rows.append(
            {
                "step": step,
                "variant": "TriGLU" if variant == "triglu" else "baseline",
                "status": status(variant, step, path),
                "he_primary": comparison["humaneval_plus_prompt_corrected"],
                "he_heritage": comparison["humaneval_plus_post_parser"],
                "mbpp": (benchmarks.get("mbpp") or {}).get("score"),
                "lcb": comparison["live_code_bench_corrected"],
                "code_primary": comparison["code_avg_prompt_corrected"],
                "code_heritage": comparison["code_avg_post_parser"],
                "gpqa": (benchmarks.get("gpqa_diamond") or {}).get("score"),
                "gpqa_freeform": (freeform_cells.get(f"{variant}_step{step}") or {}).get("accuracy_strict"),
                "mmlu": (benchmarks.get("mmlu_pro") or {}).get("score"),
                "reasoning": summary["category_scores"]["reasoning"],
                "ceval": (benchmarks.get("ceval") or {}).get("score"),
                "ifeval": (benchmarks.get("ifeval") or {}).get("score"),
                "mgsm": (benchmarks.get("mgsm") or {}).get("score"),
                "language": summary["category_scores"]["language"],
            }
        )

print("\nPRIMARY corrected-protocol category comparison:")
print("  step  architecture  CodeAvg    Reasoning  Language   status")
for row in rows:
    print(
        f"  {row['step']:>4}  {row['variant']:<12}  {pct(row['code_primary']):<9}  "
        f"{pct(row['reasoning']):<9}  "
        f"{pct(row['language']):<9}  {row['status']}"
    )

print("\nPRIMARY individual benchmark detail:")
print("  step  architecture  HumanEval+  MBPP       LCB        GPQA-Diamond  GPQA-Freeform  MMLU-Pro   C-Eval     IFEval     MGSM")
for row in rows:
    print(
        f"  {row['step']:>4}  {row['variant']:<12}  {pct(row['he_primary']):<10} "
        f"{pct(row['mbpp']):<10} {pct(row['lcb']):<10} {pct(row['gpqa']):<13} "
        f"{pct(row['gpqa_freeform']):<14} {pct(row['mmlu']):<10} "
        f"{pct(row['ceval']):<10} {pct(row['ifeval']):<10} {pct(row['mgsm'])}"
    )

print("  note: GPQA-Diamond is official 4-choice accuracy; GPQA-Freeform is complete manual-agent audit and is never averaged into it")


def mean_std(values: list[float | None]) -> str:
    present = [100.0 * float(value) for value in values if value is not None]
    if not present:
        return "pending"
    return f"{statistics.fmean(present):.3f}+/-{statistics.pstdev(present):.3f}[n={len(present)}]"


print("\nACROSS-CHECKPOINT descriptive summary (population std; pending checkpoints excluded):")
print("  architecture       HumanEval+         MBPP               LCB                GPQA-Diamond       GPQA-Freeform      MMLU-Pro           C-Eval             IFEval             MGSM")
for variant in ("TriGLU", "baseline"):
    selected = [row for row in rows if row["variant"] == variant]
    print(
        f"  {variant + ' mean+/-std':<19} "
        f"{mean_std([row['he_primary'] for row in selected]):<18} "
        f"{mean_std([row['mbpp'] for row in selected]):<18} "
        f"{mean_std([row['lcb'] for row in selected]):<18} "
        f"{mean_std([row['gpqa'] for row in selected]):<18} "
        f"{mean_std([row['gpqa_freeform'] for row in selected]):<18} "
        f"{mean_std([row['mmlu'] for row in selected]):<18} "
        f"{mean_std([row['ceval'] for row in selected]):<18} "
        f"{mean_std([row['ifeval'] for row in selected]):<18} "
        f"{mean_std([row['mgsm'] for row in selected])}"
    )

print("\nHERITAGE chat-protocol code view (preserved, not PRIMARY):")
print("  step  architecture  HumanEval+  CodeAvg    status")
for row in rows:
    print(
        f"  {row['step']:>4}  {row['variant']:<12}  {pct(row['he_heritage']):<10}  "
        f"{pct(row['code_heritage']):<9}  {row['status']}"
    )

timings = []
for path in (out / "timings").glob("*.json"):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        timings.append(int(value["wall_seconds"]))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        pass
fresh_complete = sum((cell_dir(v, s) / "CELL_COMPLETE").exists() for s in steps for v in variants)
fresh_target = 8
if timings and fresh_complete < fresh_target:
    mean = sum(timings) / len(timings)
    print(f"\nETA: mean completed cell={mean/3600:.2f} h; remaining fresh cells={fresh_target-fresh_complete}; estimated={mean*(fresh_target-fresh_complete)/3600:.2f} h")
elif fresh_complete >= fresh_target:
    print("\nETA: all eight newly evaluated cells complete")
else:
    print("\nETA: pending until the first newly evaluated cell completes")
PY

if [[ "$MODE" != --embedded ]]; then
  echo
  echo "GPU utilization / memory:"
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
  echo "disk free: $(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')G"
fi
