from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _benchmark_label(path: Path) -> str | None:
    if path.name.startswith("paper_math500_"):
        return "math500"
    if path.name.startswith("paper_gsm8k_"):
        return "gsm8k"
    if path.name.startswith("paper_olympiadbench_"):
        return "olympiadbench"
    if path.name.startswith("paper_amc23_"):
        return "amc_greedy_pass_at_1" if "amc_greedy" in path.parts else "amc_average_at_32"
    return None


def collect_benchmark_status(root: Path) -> dict[str, dict[str, Any]]:
    scores: dict[str, list[float]] = {
        "math500": [],
        "gsm8k": [],
        "olympiadbench": [],
        "amc_average_at_32": [],
        "amc_greedy_pass_at_1": [],
    }
    for path in sorted(root.rglob("reviews/**/*.jsonl")):
        label = _benchmark_label(path)
        if label is None:
            continue
        for row in _jsonl_rows(path):
            value = row.get("sample_score", {}).get("score", {}).get("value", {}).get("acc")
            if isinstance(value, (int, float)):
                scores[label].append(float(value))
    return {
        label: {
            "available": bool(values),
            "completed": len(values),
            "accuracy": sum(values) / len(values) if values else None,
        }
        for label, values in scores.items()
    }


def build_model_summary(root: Path) -> dict[str, Any]:
    benchmarks = collect_benchmark_status(root)
    primary = ("math500", "gsm8k", "olympiadbench", "amc_average_at_32")
    primary_values = [benchmarks[name]["accuracy"] for name in primary]
    complete = all(value is not None for value in primary_values)
    return {
        "benchmarks": benchmarks,
        "four_benchmark_math_average": (
            sum(float(value) for value in primary_values) / len(primary_values) if complete else None
        ),
        "four_benchmark_components": list(primary),
        "amc_greedy_excluded_from_four_benchmark_average": True,
    }


def collect_live_status(root: Path) -> dict[str, Any]:
    cells: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("generation_receipts.jsonl")):
        rows = _jsonl_rows(path)
        completions = [row for row in rows if row.get("event") == "generation_completed"]
        loads = [row for row in rows if row.get("event") == "engine_loaded"]
        label = str(path.parent.relative_to(root)) or "."
        cells[label] = {
            "actual_backend": loads[-1].get("actual_backend") if loads else None,
            "engine_load_seconds": loads[-1].get("engine_load_seconds") if loads else None,
            "completed": len(completions),
            "generated_tokens": sum(int(row.get("generated_tokens") or 0) for row in completions),
        }
    benchmarks = collect_benchmark_status(root)
    reviewed = sum(cell["completed"] for cell in benchmarks.values())
    weighted_correct = sum(
        float(cell["accuracy"]) * int(cell["completed"])
        for cell in benchmarks.values()
        if cell["accuracy"] is not None
    )
    expected = {
        "math500": 500,
        "gsm8k": 1319,
        "olympiadbench": 675,
        "amc_average_at_32": 1280,
        "amc_greedy_pass_at_1": 40,
    }
    for label, value in expected.items():
        benchmarks[label]["expected"] = value
    all_completions = sum(cell["completed"] for cell in cells.values())
    all_tokens = sum(cell["generated_tokens"] for cell in cells.values())
    started_path = root / "shell_start_unix.txt"
    elapsed = time.time() - float(started_path.read_text().strip()) if started_path.exists() else None
    rate = all_tokens / elapsed if elapsed and elapsed > 0 else None
    expected_total = sum(expected.values())
    eta = elapsed * (expected_total - all_completions) / all_completions if elapsed and all_completions else None
    try:
        gpu_memory = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=2,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        gpu_memory = "unavailable"
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cells": cells,
        "reviewed": reviewed,
        "partial_accuracy": weighted_correct / reviewed if reviewed else None,
        "benchmarks": benchmarks,
        "generated_tokens": all_tokens,
        "tokens_per_second_shell": rate,
        "elapsed_seconds": elapsed,
        "eta_seconds": eta,
        "gpu_memory_mib": gpu_memory,
    }


def format_live_status(status: dict[str, Any]) -> str:
    lines = [
        f"EVAL_STATUS {status['timestamp']} reviewed={status['reviewed']} "
        f"partial_accuracy={status['partial_accuracy']} generated_tokens={status['generated_tokens']} "
        f"tok_s={status['tokens_per_second_shell']} gpu_memory_mib={status['gpu_memory_mib']} "
        f"elapsed_s={status['elapsed_seconds']} eta_s={status['eta_seconds']}"
    ]
    for label, cell in sorted(status["cells"].items()):
        lines.append(
            f"EVAL_CELL {label} backend={cell['actual_backend']} "
            f"engine_load_s={cell['engine_load_seconds']} completed={cell['completed']} "
            f"generated_tokens={cell['generated_tokens']}"
        )
    for label, cell in status["benchmarks"].items():
        lines.append(
            f"EVAL_BENCHMARK {label} available={cell['available']} "
            f"completed={cell['completed']}/{cell['expected']} accuracy={cell['accuracy']}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(format_live_status(collect_live_status(args.root)))


if __name__ == "__main__":
    main()
