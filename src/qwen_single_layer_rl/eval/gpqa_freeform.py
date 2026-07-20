"""Deterministic contracts for the GPQA-Diamond free-form evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


EXPECTED_ROWS = 126
GENERATION_SHARDS = 6
MATCHER_ROWS = 1260
MATCHER_SHARDS = 6
ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
MATCHER_TAG = re.compile(r"<answer>\s*([01])\s*</answer>", re.IGNORECASE)


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def normalize_dataset_rows(source_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in source_rows:
        missing = {key for key in ("question_id", "question", "answer") if key not in source}
        if missing:
            raise ValueError(f"GPQA source row is missing fields: {sorted(missing)}")
        question_id = str(source["question_id"]).strip()
        question = str(source["question"]).strip()
        answer = str(source["answer"]).strip()
        if not question_id or not question or not answer:
            raise ValueError("GPQA question_id, question, and answer must be non-empty")
        if question_id in seen:
            raise ValueError(f"duplicate GPQA question_id: {question_id}")
        if re.search(r"\banswer choices?\s*:", question, re.IGNORECASE):
            raise ValueError(f"choice scaffold leaked into free-form question: {question_id}")
        seen.add(question_id)
        normalized.append(
            {
                "question_id": question_id,
                "question": question,
                "reference_answer": answer,
                "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
                "reference_answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
            }
        )
    normalized.sort(key=lambda row: row["question_id"])
    if len(normalized) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} GPQA rows, observed {len(normalized)}")
    for ordered_index, row in enumerate(normalized):
        row["ordered_index"] = ordered_index
        row["generation_rank"] = ordered_index % GENERATION_SHARDS
        row["row_sha256"] = canonical_json_sha256(row)
    return normalized


def select_shard(rows: Iterable[dict[str, Any]], *, rank: int, shard_count: int) -> list[dict[str, Any]]:
    if not 0 <= rank < shard_count:
        raise ValueError(f"rank {rank} is outside shard_count {shard_count}")
    selected = [row for row in rows if int(row["ordered_index"]) % shard_count == rank]
    return sorted(selected, key=lambda row: int(row["ordered_index"]))


def render_generation_prompt(question: str) -> str:
    return (
        "Answer the following science question without relying on answer choices. "
        "Reason carefully, then place only your final answer inside "
        "<answer>...</answer> tags.\n\nQuestion:\n"
        f"{question}\n"
    )


def extract_answer_tag(response: str) -> str | None:
    matches = ANSWER_TAG.findall(response)
    if not matches:
        return None
    answer = matches[-1].strip()
    return answer or None


def normalize_answer_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = re.sub(r"\s+", " ", value.strip().casefold())
    normalized = normalized.rstrip(". ;,")
    return normalized or None


def render_matcher_prompt(*, question: str, reference_answer: str, response: str) -> str:
    return (
        "You are given a question, a reference answer, and a candidate response. "
        "Decide whether the candidate response matches the reference answer in the context of the question. "
        "The candidate must contain at least as much correct information as the reference, but it may be "
        "more specific or use an equivalent paraphrase. For numeric answers, accept a relative error below "
        "1 percent, computed as absolute(candidate-reference) divided by mean(candidate, reference). "
        "Use the supplied reference; do not independently replace it with a different solution. "
        "Return exactly <answer>1</answer> for a match or <answer>0</answer> for a non-match.\n\n"
        f"Question:\n{question}\n\nReference answer:\n{reference_answer}\n\n"
        f"Candidate response:\n{response}\n"
    )


def parse_matcher_decision(response: str) -> int | None:
    matches = MATCHER_TAG.findall(response)
    if len(matches) != 1:
        return None
    residue = MATCHER_TAG.sub("", response).strip()
    if residue:
        return None
    return int(matches[0])


def merge_exact_shards(
    shard_paths: Iterable[Path],
    *,
    expected_rows: int,
    expected_ids: Iterable[str],
    identity_key: str,
) -> list[dict[str, Any]]:
    rows = [row for path in shard_paths for row in load_jsonl(path)]
    identities = [str(row[identity_key]) for row in rows]
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} merged rows, observed {len(rows)}")
    if len(set(identities)) != len(identities):
        raise ValueError(f"duplicate {identity_key} in merged shards")
    expected = list(expected_ids)
    if set(identities) != set(expected):
        missing = sorted(set(expected) - set(identities))
        extra = sorted(set(identities) - set(expected))
        raise ValueError(f"shard coverage mismatch: missing={missing[:5]}, extra={extra[:5]}")
    return sorted(rows, key=lambda row: int(row["ordered_index"]))


def build_matcher_ledger(generation_cells: Iterable[tuple[str, int, Path]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, global_step, path in generation_cells:
        cell_rows = load_jsonl(path)
        if len(cell_rows) != EXPECTED_ROWS:
            raise ValueError(f"{variant}-{global_step} has {len(cell_rows)} rows, expected {EXPECTED_ROWS}")
        for row in cell_rows:
            candidate_id = f"{variant}:{global_step}:{row['question_id']}"
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "variant": variant,
                    "global_step": global_step,
                    "question_id": row["question_id"],
                    "question": row["question"],
                    "reference_answer": row["reference_answer"],
                    "candidate_response": row["raw_response"],
                    "parsed_candidate_answer": row.get("parsed_answer"),
                    "normalized_exact_match": (
                        normalize_answer_text(row.get("parsed_answer"))
                        == normalize_answer_text(row["reference_answer"])
                    ),
                    "generation_finish_reason": row.get("finish_reason"),
                    "generation_cap_hit": bool(row.get("cap_hit", False)),
                }
            )
    if len(rows) != MATCHER_ROWS:
        raise ValueError(f"expected {MATCHER_ROWS} matcher rows, observed {len(rows)}")
    rows.sort(key=lambda row: (int(row["global_step"]), row["variant"], row["question_id"]))
    for ordered_index, row in enumerate(rows):
        row["ordered_index"] = ordered_index
        row["matcher_rank"] = ordered_index % MATCHER_SHARDS
        row["row_sha256"] = canonical_json_sha256(row)
    return rows


def summarize_matches(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    cells: dict[str, dict[str, Any]] = {}
    by_pair: dict[tuple[int, str], dict[str, Any]] = {}
    for row in materialized:
        key = (int(row["global_step"]), str(row["variant"]))
        cell = by_pair.setdefault(
            key,
            {
                "correct": 0,
                "incorrect": 0,
                "matcher_failures": 0,
                "cap_hits": 0,
                "missing_tags": 0,
                "normalized_exact_matches": 0,
            },
        )
        decision = row.get("matcher_decision")
        if decision == 1:
            cell["correct"] += 1
        elif decision == 0:
            cell["incorrect"] += 1
        else:
            cell["matcher_failures"] += 1
        cell["cap_hits"] += int(bool(row.get("generation_cap_hit")))
        cell["missing_tags"] += int(row.get("parsed_candidate_answer") is None)
        cell["normalized_exact_matches"] += int(bool(row.get("normalized_exact_match", False)))
    for (step, variant), cell in sorted(by_pair.items()):
        total = sum(cell[key] for key in ("correct", "incorrect", "matcher_failures"))
        if total != EXPECTED_ROWS:
            raise ValueError(f"denominator mismatch for {variant}-{step}: {total}")
        cell["total"] = total
        cell["accuracy"] = cell["correct"] / total
        cells[f"{variant}_step{step}"] = cell
    return {"rows": len(materialized), "cells": cells}
