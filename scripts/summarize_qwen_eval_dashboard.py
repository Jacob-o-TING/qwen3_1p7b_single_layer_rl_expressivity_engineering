from __future__ import annotations

import argparse
import json
from pathlib import Path

from summarize_ood_eval import build_summary


PROTOCOL_RUN = "qwen3_1p7b_heplus_prompt_matrix_full164_20260717_v1"
PROTOCOL_CELL = "evalscope_raw_instruction_nochat"


def _read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _percent(value: float | None) -> str:
    return "pending" if value is None else f"{100.0 * value:.3f}%"


def _protocol_summary(project_root: Path, label: str) -> dict | None:
    root = project_root / "runs" / "eval_protocol" / PROTOCOL_RUN
    path = root / "summary.json" if label == "untuned_base" else root / "models" / label / "summary.json"
    summary = _read_json(path)
    if summary is None or int(summary.get("sample_count") or 0) != 164:
        return None
    cell = (summary.get("cells") or {}).get(PROTOCOL_CELL)
    return cell if isinstance(cell, dict) and int(cell.get("rows") or 0) == 164 else None


def build_dashboard(project_root: Path) -> list[dict]:
    ood = project_root / "runs" / "ood_eval"
    shared = ood / "qwen3_1p7b_ood_6x5090_step294_20260716_v1"
    specs = (
        ("TriGLU-294", "triglu_step294", shared / "triglu"),
        (
            "baseline-196",
            "baseline_step196",
            ood / "qwen3_1p7b_ood_6x5090_baseline_step196_20260717_v1",
        ),
        ("untuned-base", "untuned_base", shared / "untuned_base"),
        ("baseline-294", "baseline_step294", shared / "baseline"),
    )
    rows = []
    for display, label, root in specs:
        summary = build_summary(root, project_root=project_root, model_label=label)
        comparison = summary["parser_sensitive_code_comparison"]
        benchmarks = summary["benchmarks"]
        reasoning = summary["category_scores"]["reasoning"]
        language = summary["category_scores"]["language"]
        rows.append(
            {
                "display": display,
                "label": label,
                "status": summary["status"] if summary["benchmarks"] else "pending",
                "humaneval_corrected": comparison["humaneval_plus_prompt_corrected"],
                "mbpp": (benchmarks.get("mbpp") or {}).get("score"),
                "lcb": comparison["live_code_bench_corrected"],
                "code_corrected": comparison["code_avg_prompt_corrected"],
                "humaneval_heritage": comparison["humaneval_plus_post_parser"],
                "code_heritage": comparison["code_avg_post_parser"],
                "gpqa_diamond": (benchmarks.get("gpqa_diamond") or {}).get("score"),
                "mmlu_pro": (benchmarks.get("mmlu_pro") or {}).get("score"),
                "ceval": (benchmarks.get("ceval") or {}).get("score"),
                "ifeval": (benchmarks.get("ifeval") or {}).get("score"),
                "mgsm": (benchmarks.get("mgsm") or {}).get("score"),
                "reasoning": reasoning,
                "language": language,
                "protocol": _protocol_summary(project_root, label),
            }
        )
    return rows


def print_dashboard(rows: list[dict]) -> None:
    print("=== Evaluation dashboard: current results first ===")
    print("PRIMARY corrected code protocol (raw no-chat HumanEval+; all scores pass@1):")
    print("  model             HumanEval+       MBPP        LiveCodeBench  corrected CodeAvg")
    for row in rows:
        print(
            f"  {row['display']:17s} {_percent(row['humaneval_corrected']):15s} "
            f"{_percent(row['mbpp']):11s} {_percent(row['lcb']):14s} "
            f"{_percent(row['code_corrected'])}"
        )
    print("  paper base anchor  44.500%         52.900%     7.400%         34.900%")
    print("  note: paper anchors are orientation only; exact evaluator parity is not claimed")

    print("PRIMARY reasoning OOD benchmarks (greedy pass@1):")
    print("  model             GPQA-Diamond  MMLU-Pro     reasoning avg")
    for row in rows:
        print(
            f"  {row['display']:17s} {_percent(row['gpqa_diamond']):13s} "
            f"{_percent(row['mmlu_pro']):12s} {_percent(row['reasoning'])}"
        )

    print("PRIMARY language OOD benchmarks (greedy pass@1):")
    print("  model             C-Eval        IFEval        MGSM          language avg")
    for row in rows:
        print(
            f"  {row['display']:17s} {_percent(row['ceval']):13s} "
            f"{_percent(row['ifeval']):13s} {_percent(row['mgsm']):13s} "
            f"{_percent(row['language'])}"
        )

    print("Corrected HumanEval+ diagnostics:")
    for row in rows:
        protocol = row["protocol"]
        if protocol is None:
            print(f"  {row['display']:17s} pending")
            continue
        print(
            f"  {row['display']:17s} {protocol['passed']}/{protocol['rows']} "
            f"syntax={protocol['syntax_valid_completions']} cap={protocol['cap_hits']} "
            f"timeouts={protocol.get('execution_timeouts', 'unreported')} "
            f"collapse_loops={protocol['collapse_loops']}"
        )

    print("PRIMARY current corrected-protocol summary:")
    print("  model             HumanEval+       CodeAvg     reasoning    language     status")
    for row in rows:
        print(
            f"  {row['display']:17s} {_percent(row['humaneval_corrected']):15s} "
            f"{_percent(row['code_corrected']):11s} {_percent(row['reasoning']):12s} "
            f"{_percent(row['language']):12s} {row['status']}"
        )
    print("  note: current OOD aggregates use prompt-corrected HumanEval+ and corrected LiveCodeBench")

    print("HERITAGE chat-protocol results (preserved; not the primary HumanEval+ view):")
    print("  model             HumanEval+       CodeAvg     reasoning    language     status")
    for row in rows:
        print(
            f"  {row['display']:17s} {_percent(row['humaneval_heritage']):15s} "
            f"{_percent(row['code_heritage']):11s} {_percent(row['reasoning']):12s} "
            f"{_percent(row['language']):12s} {row['status']}"
        )
    print("  note: heritage OOD means remain immutable and are not mixed with corrected HumanEval+")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", type=Path)
    args = parser.parse_args()
    print_dashboard(build_dashboard(args.project_root.resolve()))


if __name__ == "__main__":
    main()
