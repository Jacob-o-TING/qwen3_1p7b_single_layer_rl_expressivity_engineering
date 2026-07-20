from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .progress import summarize_run


VARIANTS = (
    ("layer10_whole_layer_shs", "SHS"),
    ("layer10_whole_layer_baseline", "Baseline"),
    ("layer10_whole_layer_triglu_side_ffn", "TriGLU"),
    ("layer10_whole_layer_oft", "OFT"),
)
BENCHMARKS = (
    ("paper_math500", "MATH-500", 500),
    ("paper_gsm8k", "GSM8K", 1319),
    ("paper_olympiadbench", "Olympiad", 675),
    ("paper_amc23", "AMC@32", 1280),
)
GREEDY_AMC_PREFIXES = {
    "layer10_whole_layer_shs": "amc_greedy_modal_path_shs",
    "layer10_whole_layer_baseline": "amc_greedy_modal_path_baseline",
    "layer10_whole_layer_triglu_side_ffn": "amc_greedy_modal_path_triglu",
    "layer10_whole_layer_oft": "amc_greedy_modal_path_oft",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest(paths: list[Path]) -> Path | None:
    return max(paths, key=lambda path: path.stat().st_mtime) if paths else None


def _main_result_root(eval_dir: Path) -> Path:
    manifest_path = eval_dir / "evaluation_manifest.json"
    if not manifest_path.is_file():
        return eval_dir
    cache_path = _read_json(manifest_path).get("main_use_cache")
    if not cache_path:
        return eval_dir
    resolved = Path(str(cache_path))
    return resolved if resolved.is_dir() else eval_dir


def _report(eval_dir: Path, dataset: str) -> tuple[float, int] | None:
    result_root = eval_dir if dataset == "paper_amc23" else _main_result_root(eval_dir)
    path = _latest(list(result_root.glob(f"**/reports/*/{dataset}.json")))
    if path is None:
        return None
    payload = _read_json(path)
    return 100.0 * float(payload["score"]), int(payload.get("num", 0))


def _partial(eval_dir: Path, dataset: str) -> tuple[int, int, float] | None:
    result_root = eval_dir if dataset == "paper_amc23" else _main_result_root(eval_dir)
    path = _latest(list(result_root.glob(f"**/reviews/*/{dataset}_main.jsonl")))
    if path is None:
        return None
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return None
    correct = sum(
        float(row["sample_score"]["score"]["value"]["acc"])
        for row in rows
    )
    return len(rows), round(correct), 100.0 * correct / len(rows)


def _duration(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, _ = divmod(remainder, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m"


def _training_line(run_dir: Path) -> tuple[str, bool]:
    summary = summarize_run(run_dir, window=100)
    if summary is None:
        return "Training: queued", False
    completed = summary["latest_step"] == summary["total_steps"]
    state = "DONE" if completed else f"{summary['progress']:.1%}"
    validation = summary["validations"][-1]["loss"] if summary["validations"] else None
    result_path = run_dir / "train_result.json"
    wall = _read_json(result_path).get("wall_seconds") if result_path.is_file() else None
    parts = [
        f"Training: {state} ({summary['latest_step']}/{summary['total_steps']})",
        f"last100 loss {summary['recent_window_loss_mean']:.4f}",
    ]
    if validation is not None:
        parts.append(f"val {validation:.4f}")
    parts.append(f"step {summary['recent_step_seconds_median']:.3f}s")
    if wall is not None:
        parts.append(f"wall {_duration(float(wall))}")
    if not completed:
        parts.append(f"train ETA {_duration(summary['eta_train_seconds'])}")
    return " | ".join(parts), completed


def _evaluation_lines(eval_dir: Path) -> tuple[list[str], bool, str | None]:
    cells: list[str] = []
    scores: list[float] = []
    active: str | None = None
    complete = True
    for dataset, label, total in BENCHMARKS:
        report = _report(eval_dir, dataset)
        if report is not None:
            score, count = report
            scores.append(score)
            cells.append(f"{label} {score:.2f}% [n={count}]")
            continue
        complete = False
        partial = _partial(eval_dir, dataset)
        if partial is not None:
            rows, correct, accuracy = partial
            cells.append(
                f"{label} {correct}/{rows} correct ({accuracy:.2f}%) | "
                f"generated {rows}/{total} | PARTIAL"
            )
            active = f"{label} {rows}/{total} ({100.0 * rows / total:.1f}% generated)"
        else:
            cells.append(f"{label} pending")
    lines = ["Eval: " + " | ".join(cells)]
    if complete and len(scores) == len(BENCHMARKS):
        lines.append(f"Math average: {sum(scores) / len(scores):.2f}%")
    return lines, complete, active


def _greedy_amc_cell(run_root: Path, variant_directory: str) -> str:
    prefix = GREEDY_AMC_PREFIXES[variant_directory]
    diagnostic_root = run_root / "diagnostics"
    if not diagnostic_root.is_dir():
        return "AMC greedy pending"
    candidates = [
        path
        for path in diagnostic_root.iterdir()
        if path.is_dir() and path.name.startswith(prefix) and "preflight" not in path.name
    ]
    directory = _latest(candidates)
    if directory is None:
        return "AMC greedy pending"
    report = _report(directory, "paper_amc23")
    if report is not None and (directory / "diagnostic_complete.json").is_file():
        score, count = report
        return f"AMC greedy {score:.2f}% [n={count}]"
    partial = _partial(directory, "paper_amc23")
    if partial is None:
        return "AMC greedy pending"
    rows, correct, accuracy = partial
    return f"AMC greedy {correct}/{rows} correct ({accuracy:.2f}%) | generated {rows}/40 | PARTIAL"


def _diagnostic_lines(run_root: Path) -> list[str]:
    diagnostic_root = run_root / "diagnostics"
    rows: list[str] = []
    if not diagnostic_root.is_dir():
        return rows
    preferred_order = (
        "amc_greedy_modal_path_shs",
        "amc_greedy_modal_path_baseline",
        "amc_greedy_modal_path_untuned",
        "amc_average_at_32_untuned",
        "amc_greedy_modal_path_triglu",
        "amc_greedy_modal_path_oft",
    )

    def order_key(directory: Path) -> tuple[int, str]:
        return (
            next(
                (index for index, prefix in enumerate(preferred_order) if directory.name.startswith(prefix)),
                len(preferred_order),
            ),
            directory.name,
        )

    for directory in sorted(diagnostic_root.iterdir(), key=order_key):
        if not directory.is_dir() or "preflight" in directory.name:
            continue
        receipt_path = directory / "diagnostic_complete.json"
        label_map = {
            "shs": "SHS",
            "whole_layer_baseline": "Baseline",
            "untuned_qwen3_1p7b_base": "Untuned base",
            "layer10_whole_layer_triglu_side_ffn": "TriGLU",
            "layer10_whole_layer_oft": "OFT",
        }
        if receipt_path.is_file():
            receipt = _read_json(receipt_path)
            report = _report(directory, "paper_amc23")
            if report is None:
                continue
            score, count = report
            decode = receipt.get("decode", {})
            mode = "greedy" if not decode.get("do_sample") else f"T={decode.get('temperature')} Avg@{decode.get('repeats')}"
            variant = str(receipt.get("variant", directory.name))
            label = label_map.get(variant, variant)
            rows.append(f"  {label}: {score:.2f}% | {mode} | {count} rows | FINAL")
            continue

        sampled = "average_at_32" in directory.name
        mode = "T=1.0 Avg@32" if sampled else "greedy"
        total = 1280 if sampled else 40
        partial = _partial(directory, "paper_amc23")
        if partial is None:
            continue
        generated, correct, accuracy = partial
        label = next(
            (
                friendly
                for key, friendly in (
                    ("triglu", "TriGLU"),
                    ("baseline", "Baseline"),
                    ("shs", "SHS"),
                    ("oft", "OFT"),
                    ("untuned", "Untuned base"),
                )
                if key in directory.name
            ),
            directory.name,
        )
        rows.append(
            f"  {label}: {correct}/{generated} = {accuracy:.2f}% | {mode} | "
            f"{generated}/{total} generated | PARTIAL"
        )
    return rows


def format_dashboard(run_root: Path) -> str:
    lines = [f"Run: {run_root.name}", "", "MODEL SUMMARY"]
    current = "All variants complete"
    current_found = False
    for directory_name, label in VARIANTS:
        run_dir = run_root / directory_name
        eval_dir = run_root / "evaluations" / directory_name
        greedy_amc = _greedy_amc_cell(run_root, directory_name)
        training_line, training_complete = _training_line(run_dir)
        lines.extend(("", f"[{label}]", training_line))
        if not training_complete:
            lines.append(f"Eval: pending | {greedy_amc}")
            if not current_found:
                current = f"{label} training"
                current_found = True
            continue
        eval_lines, eval_complete, active = _evaluation_lines(eval_dir)
        eval_lines[0] = f"{eval_lines[0]} | {greedy_amc}"
        lines.extend(eval_lines)
        if not eval_complete and not current_found:
            current = f"{label} evaluating {active or 'startup'}"
            current_found = True

    diagnostics = _diagnostic_lines(run_root)
    lines.extend(("", "AMC DECODE CONTROLS"))
    lines.extend(diagnostics or ["  none completed"])
    lines.extend(("", f"CURRENT PHASE: {current}"))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    print(format_dashboard(args.run_root.resolve()))


if __name__ == "__main__":
    main()
