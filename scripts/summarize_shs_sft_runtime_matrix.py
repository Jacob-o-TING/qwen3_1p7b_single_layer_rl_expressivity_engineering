#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


CASES = ("reference_eager", "reference_compile", "triton_recompute_eager", "triton_recompute_compile")
PROJECTED_STEPS = 3916


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    results = {case: json.loads((args.root / case / "benchmark_result.json").read_text()) for case in CASES}
    failures = [case for case, result in results.items() if result["dispatch_receipt"]["fallback"]]
    summary = {
        "run_id": "shs_sft_runtime_matrix_20260712_v1",
        "status": "passed" if not failures else "failed",
        "cases": results,
        "dispatch_failures": failures,
        "projected_3916_step_seconds": {
            case: result["step_seconds_median"] * PROJECTED_STEPS for case, result in results.items()
        },
    }
    lines = [
        "# SHS Matched 50-Step SFT Runtime Matrix",
        "",
        f"Status: **{summary['status']}**",
        "",
        "| Cell | Median step (s) | p10 | p90 | Assistant tok/s | Peak GB | Projected 3916 steps (h) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case, result in results.items():
        lines.append(
            f"| {case} | {result['step_seconds_median']:.4f} | {result['step_seconds_p10']:.4f} | "
            f"{result['step_seconds_p90']:.4f} | {result['timed_assistant_tokens_per_second']:.1f} | "
            f"{result['max_memory_allocated_gb']:.2f} | {summary['projected_3916_step_seconds'][case] / 3600:.2f} |"
        )
    lines.extend(["", "Triton cells use reference-recompute backward and do not claim a custom backward kernel."])
    (args.root / "manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.root / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
