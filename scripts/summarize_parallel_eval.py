from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


TRAINING_MIX_PROXY_WEIGHTS = {
    "paper_gsm8k": 0.51648,
    "paper_math500": 0.2004220113,
    "paper_olympiadbench": 0.1921370629,
    "paper_amc23_greedy": 0.0909609258,
}
COMPOSITE_LABEL = "whole_50k_training_mix_proxy_amc_greedy"
MATH_AVG_COMPONENTS = (
    "paper_gsm8k",
    "paper_math500",
    "paper_olympiadbench",
    "paper_amc23",
)
MATH_AVG_KEY = "math_avg_unweighted_amc_avg_at_32"
MATH_AVG_LABEL = "MathAvg (equal-weight; AMC Avg@32)"


def score(row: dict) -> float | None:
    value = (((row.get("sample_score") or {}).get("score") or {}).get("value") or {}).get("acc")
    return float(value) if value is not None else None


def label(path: Path) -> str:
    text = str(path).lower()
    # The protocol directory is more specific than the benchmark filename.
    # Check it first so greedy AMC is not folded into sampled Avg@32.
    if "amc_greedy" in text:
        return "paper_amc23_greedy"
    for name in ("paper_math500", "paper_gsm8k", "paper_olympiadbench", "paper_amc23"):
        if name in text:
            return name
    if "amc" in text:
        return "paper_amc23"
    return "unknown"


def display_label(cell: str) -> str:
    return {
        "paper_amc23": "paper_amc23_avg_at_32",
        "paper_amc23_greedy": "paper_amc23_greedy_pass_at_1",
    }.get(cell, cell)


def training_mix_composite(benchmarks: dict[str, dict[str, float]]) -> float | None:
    if any(
        name not in benchmarks or benchmarks[name].get("accuracy") is None
        for name in TRAINING_MIX_PROXY_WEIGHTS
    ):
        return None
    return 100.0 * sum(
        weight * float(benchmarks[name]["accuracy"])
        for name, weight in TRAINING_MIX_PROXY_WEIGHTS.items()
    )


def math_avg(benchmarks: dict[str, dict[str, float]]) -> float | None:
    if any(
        name not in benchmarks or benchmarks[name].get("accuracy") is None
        for name in MATH_AVG_COMPONENTS
    ):
        return None
    return 100.0 * sum(
        float(benchmarks[name]["accuracy"]) for name in MATH_AVG_COMPONENTS
    ) / len(MATH_AVG_COMPONENTS)


def build_summary(root: Path) -> dict:
    cells: dict[str, dict[str, float]] = defaultdict(lambda: {"rows": 0, "correct": 0.0})
    seen: set[tuple[str, str, int]] = set()
    for path in sorted(root.rglob("*.jsonl")):
        if "review" not in str(path).lower():
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            value = score(row)
            if value is None:
                continue
            cell = label(path)
            sample = row.get("sample_score") or {}
            key = (str(path), str(sample.get("sample_id")), line_no)
            if key in seen:
                continue
            seen.add(key)
            cells[cell]["rows"] += 1
            cells[cell]["correct"] += value
    receipts = 0
    tokens = 0
    for path in root.rglob("generation_receipts.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == "generation_completed":
                receipts += 1
                tokens += int(row.get("generated_tokens") or 0)
    benchmarks = {
        name: {**values, "accuracy": values["correct"] / values["rows"] if values["rows"] else None}
        for name, values in sorted(cells.items())
    }
    return {
        "status": "complete" if (root / "PARALLEL_EVAL_COMPLETE").exists() else "running",
        "generated_rows": receipts,
        "generated_tokens": tokens,
        "benchmarks": benchmarks,
        MATH_AVG_KEY: math_avg(benchmarks),
        "math_avg_uses_amc_avg_at_32": True,
        "whole_50k_training_mix_proxy_composite": training_mix_composite(benchmarks),
        "whole_50k_training_mix_proxy_uses_amc_greedy": True,
    }


def print_summary(summary: dict) -> None:
    for name, values in summary["benchmarks"].items():
        shown_name = display_label(name)
        print(f"{shown_name:34s} {values['correct']:7.1f}/{values['rows']:<5d} acc={values['accuracy']:.4f}")
    composite = summary["whole_50k_training_mix_proxy_composite"]
    if composite is None:
        print(f"{COMPOSITE_LABEL:34s} pending four complete components")
    else:
        print(f"{COMPOSITE_LABEL:34s} score={composite:.4f}")
    average = summary[MATH_AVG_KEY]
    if average is None:
        print(f"{MATH_AVG_LABEL:34s} pending four complete components")
    else:
        print(f"{MATH_AVG_LABEL:34s} score={average:.4f}")
    print(
        f"generated={summary['generated_rows']} tokens={summary['generated_tokens']} "
        f"status={summary['status']}"
    )


def collect_milestone_scores(
    root: Path,
    summary_key: str = "whole_50k_training_mix_proxy_composite",
    steps: set[int] | None = None,
) -> dict[int, dict[str, float | None]]:
    scores: dict[int, dict[str, float | None]] = defaultdict(dict)
    pattern = re.compile(r"^(triglu|baseline)_step_(\d+)$")
    for path in sorted(root.iterdir()) if root.exists() else ():
        if not path.is_dir():
            continue
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        variant, step = match.group(1), int(match.group(2))
        if steps is not None and step not in steps:
            continue
        scores[step][variant] = build_summary(path)[summary_key]
    return dict(scores)


def format_milestone_comparison(step: int, scores: dict[str, float | None]) -> str:
    triglu = scores.get("triglu")
    baseline = scores.get("baseline")
    if triglu is None or baseline is None:
        triglu_text = "pending" if triglu is None else f"{triglu:.4f}"
        baseline_text = "pending" if baseline is None else f"{baseline:.4f}"
        return f"step {step}: TriGLU={triglu_text}  baseline={baseline_text}  delta=pending"
    return (
        f"step {step}: TriGLU={triglu:.4f}  baseline={baseline:.4f}  "
        f"delta={triglu - baseline:+.4f} pp"
    )


def print_comparison(root: Path, steps: set[int] | None = None) -> None:
    print("whole-50K weighted comparison (AMC greedy pass@1):")
    scores = collect_milestone_scores(root, steps=steps)
    if not scores:
        print("  no milestone evaluations found")
    else:
        for step, step_scores in sorted(scores.items()):
            print(f"  {format_milestone_comparison(step, step_scores)}")

    print("MathAvg comparison (equal-weight; AMC Avg@32):")
    math_scores = collect_milestone_scores(root, MATH_AVG_KEY, steps=steps)
    if not math_scores:
        print("  no milestone evaluations found")
    else:
        for step, step_scores in sorted(math_scores.items()):
            print(f"  {format_milestone_comparison(step, step_scores)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--compare-subdirs", action="store_true")
    parser.add_argument("--steps", type=int, nargs="+")
    args = parser.parse_args()
    if args.compare_subdirs:
        print_comparison(args.root, set(args.steps) if args.steps else None)
        return
    summary = build_summary(args.root)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print_summary(summary)


if __name__ == "__main__":
    main()
