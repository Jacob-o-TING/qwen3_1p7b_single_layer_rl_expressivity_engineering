from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


CATEGORIES = {
    "code": ("humaneval_plus", "mbpp", "live_code_bench"),
    "reasoning": ("gpqa_diamond", "mmlu_pro"),
    "language": ("ceval", "ifeval", "mgsm"),
}
PROMPT_PROTOCOL_RUN = "qwen3_1p7b_heplus_prompt_matrix_full164_20260717_v1"
PROMPT_PROTOCOL_CELL = "evalscope_raw_instruction_nochat"


def _read_summary(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _infer_prompt_protocol_model(root: Path) -> str | None:
    value = str(root).lower()
    if root.name == "untuned_base":
        return "untuned_base"
    if root.name == "triglu" or "triglu_step294" in value:
        return "triglu_step294"
    if "baseline_step196" in value:
        return "baseline_step196"
    return None


def _prompt_protocol_summary(
    root: Path,
    *,
    project_root: Path | None = None,
    model_label: str | None = None,
) -> tuple[float | None, str | None]:
    local_summary_path = root / "primary_humanevalplus" / "summary.json"
    local_summary = _read_summary(local_summary_path)
    if local_summary is not None and int(local_summary.get("sample_count") or 0) == 164:
        local_cell = (local_summary.get("cells") or {}).get(PROMPT_PROTOCOL_CELL)
        if isinstance(local_cell, dict) and int(local_cell.get("rows") or 0) == 164:
            score = local_cell.get("score")
            if score is not None:
                return float(score), str(local_summary_path)

    label = model_label or _infer_prompt_protocol_model(root)
    if label is None:
        return None, None
    base = (project_root or Path(__file__).resolve().parents[1]) / "runs" / "eval_protocol" / PROMPT_PROTOCOL_RUN
    summary_path = base / "summary.json" if label == "untuned_base" else base / "models" / label / "summary.json"
    summary = _read_summary(summary_path)
    if summary is None or int(summary.get("sample_count") or 0) != 164:
        return None, None
    cell = (summary.get("cells") or {}).get(PROMPT_PROTOCOL_CELL)
    if not isinstance(cell, dict) or int(cell.get("rows") or 0) != 164:
        return None, None
    score = cell.get("score")
    return (float(score), str(summary_path)) if score is not None else (None, None)


def parser_sensitive_code_comparison(
    root: Path,
    benchmarks: dict,
    *,
    project_root: Path | None = None,
    model_label: str | None = None,
) -> dict:
    humaneval_pre = (benchmarks.get("humaneval_plus") or {}).get("score")
    mbpp = (benchmarks.get("mbpp") or {}).get("score")

    humaneval_post = None
    humaneval_path = None
    candidates = sorted(
        root.glob("**/diagnostics/humanevalplus_parser_v2_reviewonly_*/summary.json")
    )
    for path in reversed(candidates):
        summary = _read_summary(path)
        if summary is None or int(summary.get("rows") or 0) != 164:
            continue
        value = summary.get("parser_only_score")
        if value is not None:
            humaneval_post = float(value)
            humaneval_path = str(path)
            break
    if humaneval_post is None and humaneval_pre is not None:
        humaneval_post = float(humaneval_pre)

    livecodebench = None
    livecodebench_path = None
    candidates = sorted(
        root.glob(
            "**/diagnostics/*livecodebench*reviewonly*sandbox_output_contract*/summary.json"
        )
    )
    for path in reversed(candidates):
        summary = _read_summary(path)
        if summary is None or int(summary.get("rows") or 0) != 1055:
            continue
        value = summary.get("corrected_score")
        if value is not None and summary.get("source_predictions_unchanged") is True:
            livecodebench = float(value)
            livecodebench_path = str(path)
            break
    if livecodebench is None:
        raw_livecodebench = (benchmarks.get("live_code_bench") or {}).get("score")
        if raw_livecodebench is not None:
            livecodebench = float(raw_livecodebench)

    ready = None not in (humaneval_pre, humaneval_post, mbpp, livecodebench)
    humaneval_prompt_corrected, humaneval_prompt_path = _prompt_protocol_summary(
        root,
        project_root=project_root,
        model_label=model_label,
    )
    prompt_ready = None not in (humaneval_prompt_corrected, mbpp, livecodebench)
    return {
        "humaneval_plus_pre_parser": humaneval_pre,
        "humaneval_plus_post_parser": humaneval_post,
        "live_code_bench_corrected": livecodebench,
        "code_avg_pre_parser": (
            (humaneval_pre + mbpp + livecodebench) / 3 if ready else None
        ),
        "code_avg_post_parser": (
            (humaneval_post + mbpp + livecodebench) / 3 if ready else None
        ),
        "humaneval_plus_prompt_corrected": humaneval_prompt_corrected,
        "code_avg_prompt_corrected": (
            (humaneval_prompt_corrected + mbpp + livecodebench) / 3
            if prompt_ready
            else None
        ),
        "definition": (
            "equal-weight HumanEval+ + MBPP + corrected LiveCodeBench; only HumanEval+ "
            "changes between pre and post"
        ),
        "humaneval_recovery_summary": humaneval_path,
        "humaneval_prompt_protocol_summary": humaneval_prompt_path,
        "livecodebench_recovery_summary": livecodebench_path,
    }


def build_summary(
    root: Path,
    *,
    project_root: Path | None = None,
    model_label: str | None = None,
) -> dict:
    weighted: dict[str, list[tuple[float, int]]] = defaultdict(list)
    report_files: list[str] = []
    for path in sorted(root.rglob("*.json")):
        if "reports" not in path.parts:
            continue
        if any("_failed_" in part for part in path.parts):
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            dataset = str(report["dataset_name"])
            value = float(report["score"])
            count = int(report["num"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if dataset not in {name for names in CATEGORIES.values() for name in names}:
            continue
        weighted[dataset].append((value, count))
        report_files.append(str(path))

    benchmarks = {}
    for dataset, values in sorted(weighted.items()):
        samples = sum(count for _, count in values)
        score = sum(value * count for value, count in values) / samples if samples else None
        benchmarks[dataset] = {"score": score, "samples": samples, "reports": len(values)}

    parser_comparison = parser_sensitive_code_comparison(
        root,
        benchmarks,
        project_root=project_root,
        model_label=model_label,
    )
    effective_benchmarks = {
        name: {**cell, "score_source": "raw_report"}
        for name, cell in benchmarks.items()
    }
    corrections = {
        "humaneval_plus": (
            parser_comparison["humaneval_plus_post_parser"],
            (
                "parser_sandbox_recovery"
                if parser_comparison["humaneval_recovery_summary"]
                else "raw_report"
            ),
        ),
        "live_code_bench": (
            parser_comparison["live_code_bench_corrected"],
            (
                "sandbox_output_contract_recovery"
                if parser_comparison["livecodebench_recovery_summary"]
                else "raw_report"
            ),
        ),
    }
    for name, (score, source) in corrections.items():
        if score is None or name not in effective_benchmarks:
            continue
        effective_benchmarks[name]["raw_report_score"] = effective_benchmarks[name]["score"]
        effective_benchmarks[name]["score"] = score
        effective_benchmarks[name]["score_source"] = source

    category_scores = {}
    for category, names in CATEGORIES.items():
        scores = [
            effective_benchmarks[name]["score"]
            for name in names
            if name in effective_benchmarks
        ]
        category_scores[category] = sum(scores) / len(scores) if len(scores) == len(names) else None
    all_scores = [
        effective_benchmarks[name]["score"]
        for names in CATEGORIES.values()
        for name in names
        if name in effective_benchmarks
    ]
    complete = all(value is not None for value in category_scores.values())
    return {
        "status": "complete" if (root / "PARALLEL_OOD_EVAL_COMPLETE").exists() and complete else "running",
        "protocol": "vllm_greedy_pass_at_1_max_tokens_3072",
        "benchmarks": effective_benchmarks,
        "raw_report_benchmarks": benchmarks,
        "category_scores": category_scores,
        "ood_benchmark_mean": sum(all_scores) / len(all_scores) if len(all_scores) == 8 else None,
        "ood_category_mean": (
            sum(category_scores.values()) / 3 if complete else None
        ),
        "report_files": report_files,
        "parser_sensitive_code_comparison": parser_comparison,
    }


def print_summary(summary: dict) -> None:
    for category, names in CATEGORIES.items():
        print(f"{category}:")
        for name in names:
            cell = summary["benchmarks"].get(name)
            if cell is None:
                print(f"  {name:20s} pending")
            else:
                source = cell.get("score_source", "raw_report")
                suffix = "" if source == "raw_report" else "  [corrected]"
                print(
                    f"  {name:20s} {100 * cell['score']:7.3f}%  "
                    f"n={cell['samples']}{suffix}"
                )
        score = summary["category_scores"][category]
        print(f"  {category + '_mean':20s} {'pending' if score is None else f'{100 * score:.3f}%'}")
        if category == "code":
            comparison = summary["parser_sensitive_code_comparison"]

            def shown(value: float | None) -> str:
                return "pending" if value is None else f"{100 * value:.3f}%"

            print(
                "  HumanEval+ pre/post parser "
                f"{shown(comparison['humaneval_plus_pre_parser'])} -> "
                f"{shown(comparison['humaneval_plus_post_parser'])} "
                "(post prefers parser recovery when available)"
            )
            print(
                "  CodeAvg pre/post parser    "
                f"{shown(comparison['code_avg_pre_parser'])} -> "
                f"{shown(comparison['code_avg_post_parser'])} "
                "(LCB corrected in both)"
            )
            print(
                "  corrected LiveCodeBench   "
                f"{shown(comparison['live_code_bench_corrected'])}"
            )
            print(
                "  HumanEval+ chat/parser -> prompt-corrected "
                f"{shown(comparison['humaneval_plus_post_parser'])} -> "
                f"{shown(comparison['humaneval_plus_prompt_corrected'])}"
            )
            print(
                "  CodeAvg chat/parser -> prompt-corrected    "
                f"{shown(comparison['code_avg_post_parser'])} -> "
                f"{shown(comparison['code_avg_prompt_corrected'])}"
            )
    print(f"status={summary['status']} protocol={summary['protocol']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--compare-models", action="store_true")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--model-label")
    args = parser.parse_args()
    if args.compare_models:
        for model in ("triglu", "baseline", "untuned_base"):
            print(f"=== {model} ===")
            print_summary(
                build_summary(
                    args.root / model,
                    project_root=args.project_root,
                    model_label=(model if model == "untuned_base" else None),
                )
            )
        return
    summary = build_summary(
        args.root,
        project_root=args.project_root,
        model_label=args.model_label,
    )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print_summary(summary)


if __name__ == "__main__":
    main()
