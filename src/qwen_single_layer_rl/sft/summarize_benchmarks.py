from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CASES = (
    "baseline_eager",
    "baseline_compile",
    "shs_eager",
    "shs_compile",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(root: Path) -> dict[str, Any]:
    results = {case: _load(root / case / "benchmark_result.json") for case in CASES}
    baseline_eager = float(results["baseline_eager"]["step_seconds_median"])
    baseline_compile = float(results["baseline_compile"]["step_seconds_median"])
    shs_eager = float(results["shs_eager"]["step_seconds_median"])
    shs_compile = float(results["shs_compile"]["step_seconds_median"])
    baseline_loss = float(results["baseline_eager"]["correctness_loss"])
    shs_eager_loss = float(results["shs_eager"]["correctness_loss"])
    return {
        "root": str(root),
        "cases": results,
        "comparisons": {
            "baseline_compile_speedup": baseline_eager / baseline_compile,
            "shs_compile_speedup": shs_eager / shs_compile,
            "shs_eager_overhead_ratio": shs_eager / baseline_eager,
            "shs_compile_overhead_ratio": shs_compile / baseline_compile,
            "baseline_compile_correctness_loss_relative_delta": abs(
                float(results["baseline_compile"]["correctness_loss"]) - baseline_loss
            ) / max(1.0e-12, abs(baseline_loss)),
            "shs_eager_initial_noop_loss_relative_delta": abs(shs_eager_loss - baseline_loss) /
            max(1.0e-12, abs(baseline_loss)),
            "shs_compile_correctness_loss_relative_delta": abs(
                float(results["shs_compile"]["correctness_loss"]) - shs_eager_loss
            ) / max(1.0e-12, abs(shs_eager_loss)),
            "baseline_compile_break_even_steps": max(
                0.0,
                float(results["baseline_compile"]["cold_step_seconds"]) /
                max(1.0e-12, baseline_eager - baseline_compile),
            ),
            "shs_compile_break_even_steps": max(
                0.0,
                float(results["shs_compile"]["cold_step_seconds"]) /
                max(1.0e-12, shs_eager - shs_compile),
            ),
        },
    }


def summarize_pair(root: Path, pair: str) -> dict[str, Any]:
    case_names = (f"{pair}_eager", f"{pair}_compile")
    results = {case: _load(root / case / "benchmark_result.json") for case in case_names}
    eager = float(results[case_names[0]]["step_seconds_median"])
    compiled = float(results[case_names[1]]["step_seconds_median"])
    cold = float(results[case_names[1]]["cold_step_seconds"])
    break_even = cold / (eager - compiled) if compiled < eager else None
    eager_loss = float(results[case_names[0]]["correctness_loss"])
    return {
        "root": str(root),
        "pair": pair,
        "cases": results,
        "comparisons": {
            "compile_speedup": eager / compiled,
            "compile_correctness_loss_relative_delta": abs(
                float(results[case_names[1]]["correctness_loss"]) - eager_loss
            ) / max(1.0e-12, abs(eager_loss)),
            "compile_break_even_steps": break_even,
        },
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Qwen3-1.7B SFT Eager/Compile Short Benchmark",
        "",
        "| Case | Compile | Initial loss | Cold step (s) | Median step (s) | Assistant tok/s | Peak allocated (GB) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for case in CASES:
        result = summary["cases"][case]
        lines.append(
            f"| {case} | {result['compile_mode']} | {result['correctness_loss']:.6f} | "
            f"{result['cold_step_seconds']:.3f} | {result['step_seconds_median']:.3f} | "
            f"{result['timed_assistant_tokens_per_second']:.1f} | "
            f"{result['max_memory_allocated_gb']:.2f} |"
        )
    comparisons = summary["comparisons"]
    lines.extend(
        [
            "",
            "## Comparisons",
            "",
            f"- Baseline compile speedup: `{comparisons['baseline_compile_speedup']:.3f}x`.",
            f"- SHS compile speedup: `{comparisons['shs_compile_speedup']:.3f}x`.",
            f"- SHS eager overhead: `{comparisons['shs_eager_overhead_ratio']:.3f}x` baseline eager.",
            f"- SHS compiled overhead: `{comparisons['shs_compile_overhead_ratio']:.3f}x` baseline compiled.",
            f"- Baseline compile initial-loss relative delta: `{comparisons['baseline_compile_correctness_loss_relative_delta']:.6f}`.",
            f"- SHS eager exact-noop initial-loss relative delta: `{comparisons['shs_eager_initial_noop_loss_relative_delta']:.6f}`.",
            f"- SHS compile initial-loss relative delta: `{comparisons['shs_compile_correctness_loss_relative_delta']:.6f}`.",
            f"- Baseline compile break-even: `{comparisons['baseline_compile_break_even_steps']:.1f}` steps.",
            f"- SHS compile break-even: `{comparisons['shs_compile_break_even_steps']:.1f}` steps.",
            "",
        ]
    )
    return "\n".join(lines)


def render_pair_markdown(summary: dict[str, Any]) -> str:
    pair = str(summary["pair"])
    case_names = (f"{pair}_eager", f"{pair}_compile")
    lines = [
        f"# Qwen3-1.7B SFT {pair.upper()} Production-Shape Gate",
        "",
        "| Case | Compile | Initial loss | Cold step (s) | Median step (s) | Assistant tok/s | Peak allocated (GB) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for case in case_names:
        result = summary["cases"][case]
        lines.append(
            f"| {case} | {result['compile_mode']} | {result['correctness_loss']:.6f} | "
            f"{result['cold_step_seconds']:.3f} | {result['step_seconds_median']:.3f} | "
            f"{result['timed_assistant_tokens_per_second']:.1f} | "
            f"{result['max_memory_allocated_gb']:.2f} |"
        )
    comparisons = summary["comparisons"]
    break_even = comparisons["compile_break_even_steps"]
    lines.extend(
        [
            "",
            "## Comparisons",
            "",
            f"- Compile speedup: `{comparisons['compile_speedup']:.3f}x`.",
            f"- Compile initial-loss relative delta: `{comparisons['compile_correctness_loss_relative_delta']:.6f}`.",
            (
                f"- Compile break-even: `{break_even:.1f}` optimizer steps."
                if break_even is not None
                else "- Compile break-even: none; compiled steady-state was not faster."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pair", choices=("baseline", "shs"))
    args = parser.parse_args()
    summary = summarize_pair(args.root, args.pair) if args.pair else summarize(args.root)
    markdown = render_pair_markdown(summary) if args.pair else render_markdown(summary)
    (args.root / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.root / "benchmark_summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
