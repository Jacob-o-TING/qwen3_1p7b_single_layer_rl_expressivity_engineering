from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from qwen_single_layer_rl.eval.gpqa_freeform import load_jsonl, write_json_atomic, write_jsonl_atomic


EXPECTED_QUESTIONS = 126
RESPONSES_PER_QUESTION = 10
EXPECTED_ROWS = EXPECTED_QUESTIONS * RESPONSES_PER_QUESTION
REQUIRED = {
    "audit_row_id",
    "response_read_completely",
    "semantic_correctness",
    "answer_extraction",
    "failure_category",
    "confidence",
    "evidence_note",
}
VERDICTS = {"correct", "incorrect", "uncertain"}
EXTRACTION = {
    "tag_consistent",
    "tag_missing_but_answer_present",
    "tag_wrong_or_conflicting",
    "no_answer",
    "uncertain",
}
FAILURES = {
    "correct",
    "factual_error",
    "reasoning_error",
    "contradictory_answer",
    "insufficient_or_vague",
    "truncated",
    "malformed_or_no_answer",
    "reference_or_question_ambiguity",
    "other",
}
CONFIDENCE = {"high", "medium", "low"}


def validate_verdict(row: dict[str, Any], canonical_ids: set[str]) -> None:
    missing = REQUIRED - row.keys()
    if missing:
        raise ValueError(f"verdict is missing fields {sorted(missing)}: {row.get('audit_row_id')}")
    if row["audit_row_id"] not in canonical_ids:
        raise ValueError(f"unknown audit_row_id: {row['audit_row_id']}")
    if row["response_read_completely"] is not True:
        raise ValueError(f"response_read_completely must be true: {row['audit_row_id']}")
    if row["semantic_correctness"] not in VERDICTS:
        raise ValueError(f"invalid semantic_correctness: {row['audit_row_id']}")
    if row["answer_extraction"] not in EXTRACTION:
        raise ValueError(f"invalid answer_extraction: {row['audit_row_id']}")
    if row["failure_category"] not in FAILURES:
        raise ValueError(f"invalid failure_category: {row['audit_row_id']}")
    if row["confidence"] not in CONFIDENCE:
        raise ValueError(f"invalid confidence: {row['audit_row_id']}")
    if not isinstance(row["evidence_note"], str) or not row["evidence_note"].strip():
        raise ValueError(f"evidence_note must be non-empty: {row['audit_row_id']}")
    if row["semantic_correctness"] == "correct" and row["failure_category"] != "correct":
        raise ValueError(f"correct verdict must use correct failure category: {row['audit_row_id']}")
    if row["semantic_correctness"] == "incorrect" and row["failure_category"] == "correct":
        raise ValueError(f"incorrect verdict cannot use correct failure category: {row['audit_row_id']}")


def collect(package: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    canonical = load_jsonl(package / "audit_rows.jsonl")
    canonical_ids = {str(row["audit_row_id"]) for row in canonical}
    if len(canonical) != EXPECTED_ROWS or len(canonical_ids) != EXPECTED_ROWS:
        raise RuntimeError("canonical audit rows failed identity gate")
    verdicts: list[dict[str, Any]] = []
    for path in sorted((package / "audit_staging").glob("question_*.verdicts.jsonl")):
        rows = load_jsonl(path)
        if len(rows) != RESPONSES_PER_QUESTION:
            raise ValueError(f"{path} has {len(rows)} verdicts, expected ten")
        for row in rows:
            validate_verdict(row, canonical_ids)
        verdicts.extend(rows)
    ids = [str(row["audit_row_id"]) for row in verdicts]
    duplicates = sorted(row_id for row_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate staged verdict IDs: {duplicates[:5]}")
    order = {str(row["audit_row_id"]): index for index, row in enumerate(canonical)}
    verdicts.sort(key=lambda row: order[str(row["audit_row_id"])])
    return canonical, verdicts


def build_summary(canonical: list[dict[str, Any]], verdicts: list[dict[str, Any]]) -> dict[str, Any]:
    source = {str(row["audit_row_id"]): row for row in canonical}
    cells: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"correct": 0, "incorrect": 0, "uncertain": 0, "total": 0}
    )
    failure_counts: Counter[str] = Counter()
    enriched: list[dict[str, Any]] = []
    for verdict in verdicts:
        original = source[str(verdict["audit_row_id"])]
        label = f"{original['variant']}_step{original['global_step']}"
        cells[label][str(verdict["semantic_correctness"])] += 1
        cells[label]["total"] += 1
        failure_counts[str(verdict["failure_category"])] += 1
        enriched.append({**original, **verdict})
    for cell in cells.values():
        cell["accuracy_strict"] = cell["correct"] / cell["total"] if cell["total"] else None
        decided = cell["correct"] + cell["incorrect"]
        cell["accuracy_decided_only"] = cell["correct"] / decided if decided else None
    by_variant: dict[str, list[float]] = defaultdict(list)
    for label, cell in cells.items():
        if cell["total"] != EXPECTED_QUESTIONS:
            raise RuntimeError(f"cell denominator does not close: {label}={cell['total']}")
        by_variant[label.split("_step", 1)[0]].append(float(cell["accuracy_strict"]))
    across = {
        variant: {
            "checkpoint_count": len(values),
            "mean_accuracy_strict": statistics.fmean(values),
            "population_std_accuracy_strict": statistics.pstdev(values),
        }
        for variant, values in by_variant.items()
    }
    uncertain = [row for row in enriched if row["semantic_correctness"] == "uncertain"]
    grouped = {(row["question_id"], row["global_step"], row["variant"]): row for row in enriched}
    disagreements: list[dict[str, Any]] = []
    for question_id in sorted({str(row["question_id"]) for row in canonical}):
        for step in sorted({int(row["global_step"]) for row in canonical}):
            left = grouped.get((question_id, step, "triglu"))
            right = grouped.get((question_id, step, "baseline"))
            if left and right and left["semantic_correctness"] != right["semantic_correctness"]:
                disagreements.extend((left, right))
    return {
        "status": "MANUAL_AUDIT_COMPLETE",
        "verdict_count": len(verdicts),
        "unique_verdict_ids": len({row["audit_row_id"] for row in verdicts}),
        "cells": dict(sorted(cells.items())),
        "across_checkpoints": across,
        "failure_category_counts": dict(sorted(failure_counts.items())),
        "uncertain_rows": len(uncertain),
        "disagreement_rows": len(disagreements),
        "_uncertain_payload": uncertain,
        "_disagreement_payload": disagreements,
    }


def sync(package: Path) -> dict[str, Any]:
    canonical, verdicts = collect(package)
    write_jsonl_atomic(package / "verdicts.jsonl", verdicts)
    source = {str(row["audit_row_id"]): row for row in canonical}
    counts = Counter(int(source[str(row["audit_row_id"])]["question_index"]) for row in verdicts)
    completed = sorted(index for index, count in counts.items() if count == RESPONSES_PER_QUESTION)
    if any(count != RESPONSES_PER_QUESTION for count in counts.values()):
        raise RuntimeError("a staged question is not an exact ten-row unit")
    progress = {
        "status": "COMPLETE" if len(verdicts) == EXPECTED_ROWS else "IN_PROGRESS",
        "expected_questions": EXPECTED_QUESTIONS,
        "expected_verdicts": EXPECTED_ROWS,
        "completed_question_indices": completed,
        "verdict_count": len(verdicts),
    }
    write_json_atomic(package / "progress.json", progress)
    if len(verdicts) == EXPECTED_ROWS:
        summary = build_summary(canonical, verdicts)
        uncertain = summary.pop("_uncertain_payload")
        disagreements = summary.pop("_disagreement_payload")
        write_jsonl_atomic(package / "review_uncertain.jsonl", uncertain)
        write_jsonl_atomic(package / "review_disagreements.jsonl", disagreements)
        write_json_atomic(package / "final_summary.json", summary)
        (package / "AUDIT_COMPLETE").write_text("complete\n", encoding="utf-8")
    return progress


def validate(package: Path) -> dict[str, Any]:
    canonical, verdicts = collect(package)
    progress = json.loads((package / "progress.json").read_text(encoding="utf-8"))
    if progress.get("verdict_count") != len(verdicts):
        raise RuntimeError("progress verdict count drift")
    if (package / "AUDIT_COMPLETE").is_file():
        if len(verdicts) != EXPECTED_ROWS:
            raise RuntimeError("AUDIT_COMPLETE exists before exact verdict closure")
        summary = build_summary(canonical, verdicts)
        summary.pop("_uncertain_payload")
        summary.pop("_disagreement_payload")
        recorded = json.loads((package / "final_summary.json").read_text(encoding="utf-8"))
        if recorded != summary:
            raise RuntimeError("final summary drift")
    return progress


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("action", choices=("sync", "validate"))
    args = parser.parse_args()
    result = sync(args.package.resolve()) if args.action == "sync" else validate(args.package.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
