from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def score(row: dict[str, Any]) -> float:
    return float(row["sample_score"]["score"]["value"]["acc"])


def score_payload(row: dict[str, Any]) -> dict[str, Any]:
    return row["sample_score"]["score"]


def key(row: dict[str, Any], *, sampled: bool) -> tuple[int, int]:
    sample = row["sample_score"]
    group_id = int(sample["group_id"])
    sample_id = int(sample["sample_id"])
    return group_id, sample_id % 32 if sampled else 0


def indexed(path: Path, *, sampled: bool) -> dict[tuple[int, int], dict[str, Any]]:
    result = {key(row, sampled=sampled): row for row in rows(path)}
    if len(result) != len(rows(path)):
        raise RuntimeError(f"duplicate review identity in {path}")
    return result


def percentile(values: list[int], fraction: float) -> int:
    return sorted(values)[min(len(values) - 1, int((len(values) - 1) * fraction))]


def compare(
    old: dict[tuple[int, int], dict[str, Any]],
    new: dict[tuple[int, int], dict[str, Any]],
    tokenizer: Any,
) -> dict[str, Any]:
    if old.keys() != new.keys():
        raise RuntimeError(
            f"identity mismatch: old_only={sorted(old.keys() - new.keys())[:10]} "
            f"new_only={sorted(new.keys() - old.keys())[:10]}"
        )
    old_lengths: list[int] = []
    new_lengths: list[int] = []
    old_missing = new_missing = extracted_equal = 0
    correct_to_wrong = wrong_to_correct = 0
    for identity in sorted(old):
        old_score, new_score = score(old[identity]), score(new[identity])
        correct_to_wrong += old_score > 0.5 and new_score <= 0.5
        wrong_to_correct += old_score <= 0.5 and new_score > 0.5
        old_payload, new_payload = score_payload(old[identity]), score_payload(new[identity])
        old_extracted = old_payload.get("extracted_prediction")
        new_extracted = new_payload.get("extracted_prediction")
        old_missing += old_extracted in (None, "")
        new_missing += new_extracted in (None, "")
        extracted_equal += old_extracted == new_extracted
        old_lengths.append(len(tokenizer.encode(old_payload.get("prediction", ""), add_special_tokens=False)))
        new_lengths.append(len(tokenizer.encode(new_payload.get("prediction", ""), add_special_tokens=False)))
    count = len(old)
    old_correct = sum(score(row) for row in old.values())
    new_correct = sum(score(row) for row in new.values())
    return {
        "count": count,
        "old_correct": old_correct,
        "new_correct": new_correct,
        "old_accuracy_percent": 100.0 * old_correct / count,
        "new_accuracy_percent": 100.0 * new_correct / count,
        "delta_points": 100.0 * (new_correct - old_correct) / count,
        "correct_to_wrong": correct_to_wrong,
        "wrong_to_correct": wrong_to_correct,
        "extracted_answer_exact_agreement": extracted_equal / count,
        "old_missing_extraction": old_missing,
        "new_missing_extraction": new_missing,
        "old_cap_hits_3072": sum(value >= 3072 for value in old_lengths),
        "new_cap_hits_3072": sum(value >= 3072 for value in new_lengths),
        "old_response_tokens": {
            "mean": statistics.fmean(old_lengths),
            "p50": percentile(old_lengths, 0.50),
            "p90": percentile(old_lengths, 0.90),
        },
        "new_response_tokens": {
            "mean": statistics.fmean(new_lengths),
            "p50": percentile(new_lengths, 0.50),
            "p90": percentile(new_lengths, 0.90),
        },
    }


def generation_timing(paths: list[Path]) -> dict[str, Any]:
    generated_tokens = 0
    phase_batches: dict[str, set[tuple[float, int]]] = defaultdict(set)
    rows_count = 0
    for path in paths:
        for row in rows(path):
            if row.get("event") != "generation_completed":
                continue
            rows_count += 1
            generated_tokens += int(row.get("generated_tokens") or 0)
            identity = row["identity"]
            if identity["benchmark"] != "paper_amc23":
                phase = "main"
            elif row["generation"]["do_sample"]:
                phase = "amc_average_at_32"
            else:
                phase = "amc_greedy"
            phase_batches[phase].add(
                (float(row["batch_elapsed_seconds"]), int(row["batch_size"]))
            )
    phase_seconds = {
        phase: sum(elapsed for elapsed, _ in batches) for phase, batches in phase_batches.items()
    }
    return {
        "generated_rows_including_repair": rows_count,
        "generated_tokens_including_repair": generated_tokens,
        "generation_call_wall_seconds": sum(phase_seconds.values()),
        "generation_call_wall_by_phase_seconds": phase_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--old-primary", type=Path, required=True)
    parser.add_argument("--old-greedy", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    old_main = args.old_primary / "main" / "20260711_050132" / "reviews" / "qwen3-1p7b-single-layer-sft"
    new_main = args.root / "main" / "20260712_144252" / "reviews" / "qwen3-1p7b-single-layer-sft"
    old_amc = args.old_primary / "amc_average_at_32" / "20260711_095148" / "reviews" / "qwen3-1p7b-single-layer-sft" / "paper_amc23_main.jsonl"
    new_amc = args.root / "_repair_cache" / "amc_1279_clean" / "reviews" / "qwen3-1p7b-single-layer-sft" / "paper_amc23_main.jsonl"
    old_greedy = args.old_greedy / "main" / "20260711_185923" / "reviews" / "qwen3-1p7b-single-layer-sft" / "paper_amc23_main.jsonl"
    new_greedy = next((args.root / "amc_greedy").rglob("reviews/**/*.jsonl"))
    files = {
        "math500": (old_main / "paper_math500_main.jsonl", new_main / "paper_math500_main.jsonl", False),
        "gsm8k": (old_main / "paper_gsm8k_main.jsonl", new_main / "paper_gsm8k_main.jsonl", False),
        "olympiadbench": (old_main / "paper_olympiadbench_main.jsonl", new_main / "paper_olympiadbench_main.jsonl", False),
        "amc_average_at_32": (old_amc, new_amc, True),
        "amc_greedy_pass_at_1": (old_greedy, new_greedy, False),
    }
    comparisons: dict[str, Any] = {}
    indexed_rows: dict[str, tuple[dict[tuple[int, int], Any], dict[tuple[int, int], Any]]] = {}
    for name, (old_path, new_path, sampled) in files.items():
        old_rows = indexed(old_path, sampled=sampled)
        new_rows = indexed(new_path, sampled=sampled)
        indexed_rows[name] = old_rows, new_rows
        comparisons[name] = compare(old_rows, new_rows, tokenizer)

    amc_old, amc_new = indexed_rows["amc_average_at_32"]
    per_item = []
    for item_id in range(40):
        per_item.append(
            {
                "item_id": item_id,
                "old_correct_count_32": int(sum(score(amc_old[(item_id, sample)]) for sample in range(32))),
                "new_correct_count_32": int(sum(score(amc_new[(item_id, sample)]) for sample in range(32))),
            }
        )

    primary = ("math500", "gsm8k", "olympiadbench", "amc_average_at_32")
    old_average = statistics.fmean(comparisons[name]["old_accuracy_percent"] for name in primary)
    new_average = statistics.fmean(comparisons[name]["new_accuracy_percent"] for name in primary)
    first_receipt = json.loads((args.root / "full_eval_completion_receipt_incomplete_1277_attempt1.json").read_text())
    start_unix = float((args.root / "shell_start_unix.txt").read_text().strip())
    final_unix = (args.root / "_repair_cache" / "amc_1279_clean" / "reports" / "report.html").stat().st_mtime
    final_wall = final_unix - start_unix
    timing = generation_timing(
        [
            args.root / "generation_receipts.jsonl",
            args.root / "repair_attempt_3keys" / "generation_receipts.jsonl",
            args.root / "repair_attempt_1key" / "generation_receipts.jsonl",
        ]
    )
    timing.update(
        {
            "first_full_attempt_shell_wall_seconds": first_receipt["shell_wall_seconds"],
            "artifact_complete_shell_wall_seconds": final_wall,
            "old_hf_serial_shell_wall_seconds": 8 * 3600 + 10 * 60,
            "artifact_complete_speedup_vs_old_hf": (8 * 3600 + 10 * 60) / final_wall,
            "generated_token_throughput_artifact_complete": timing["generated_tokens_including_repair"] / final_wall,
            "long_decode_anchor_tokens_per_second": 2262.8,
            "throughput_fraction_of_long_decode_anchor": (
                timing["generated_tokens_including_repair"] / final_wall / 2262.8
            ),
        }
    )
    payload = {
        "run_id": "shs_vllm_full_eval_shadow_20260712_v1",
        "protocol_label": "SHADOW/NEW BACKEND",
        "strict_compatible": False,
        "compatibility_boundary_points": 0.5,
        "comparisons": comparisons,
        "four_task_average": {
            "old_percent": old_average,
            "new_percent": new_average,
            "delta_points": new_average - old_average,
        },
        "amc_per_item_correct_count": per_item,
        "timing": timing,
        "repair": {
            "malformed_original_rows_preserved": True,
            "missing_ids_regenerated": [[11, 13], [25, 2], [32, 28]],
            "single_writer_final_repair_id": [25, 2],
            "final_amc_review_rows": len(amc_new),
        },
        "preflight": json.loads((args.root / "preflight_manifest.json").read_text()),
        "source_hashes": {
            str(path): sha256(path)
            for path in (
                Path("src/qwen_single_layer_rl/eval/evalscope_custom_model.py"),
                Path("src/qwen_single_layer_rl/eval/generator_backend.py"),
                Path("src/qwen_single_layer_rl/eval/live_status.py"),
                Path("src/qwen_single_layer_rl/eval/run_evalscope.py"),
                Path("scripts/run_shs_vllm_full_eval_shadow_20260712_v1.sh"),
                Path("scripts/summarize_shs_vllm_full_eval_shadow.py"),
            )
        },
    }
    (args.root / "compact_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# SHS vLLM Full Evaluation Shadow Summary",
        "",
        "Protocol: **SHADOW/NEW BACKEND**. This result is not strict-compatible.",
        "",
        "| Benchmark | Old HF % | New vLLM % | Delta pt | C->W | W->C | Extract agreement |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in (*primary, "amc_greedy_pass_at_1"):
        item = comparisons[name]
        lines.append(
            f"| {name} | {item['old_accuracy_percent']:.4f} | {item['new_accuracy_percent']:.4f} | "
            f"{item['delta_points']:+.4f} | {item['correct_to_wrong']} | {item['wrong_to_correct']} | "
            f"{100 * item['extracted_answer_exact_agreement']:.2f}% |"
        )
    lines.extend(
        [
            "",
            f"Four-task average: {old_average:.4f}% -> {new_average:.4f}% ({new_average-old_average:+.4f} pt).",
            f"Artifact-complete wall: {final_wall:.1f}s; speedup versus 8h10m: {timing['artifact_complete_speedup_vs_old_hf']:.3f}x.",
            f"Generated tokens including repair: {timing['generated_tokens_including_repair']}; artifact-complete throughput: {timing['generated_token_throughput_artifact_complete']:.1f} tok/s.",
            "",
            "See `compact_summary.json` for length, cap, extraction, timing, and per-item AMC details.",
        ]
    )
    (args.root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    final_receipt = {
        "status": "complete",
        "run_id": payload["run_id"],
        "protocol_label": payload["protocol_label"],
        "strict_compatible": False,
        "verified_unique_review_rows": {
            name: comparison["count"] for name, comparison in comparisons.items()
        },
        "scores": {
            name: comparison["new_accuracy_percent"] for name, comparison in comparisons.items()
        },
        "four_task_average_percent": new_average,
        "timing": timing,
        "repair": payload["repair"],
        "compact_summary": str((args.root / "compact_summary.json").resolve()),
    }
    (args.root / "full_eval_completion_receipt.json").write_text(
        json.dumps(final_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
