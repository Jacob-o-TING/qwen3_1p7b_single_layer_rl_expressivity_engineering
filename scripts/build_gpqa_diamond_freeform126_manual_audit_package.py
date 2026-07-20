from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from qwen_single_layer_rl.eval.gpqa_freeform import (
    canonical_json_sha256,
    file_sha256,
    load_jsonl,
    normalize_answer_text,
    write_json_atomic,
    write_jsonl_atomic,
)


EXPECTED_QUESTIONS = 126
RESPONSES_PER_QUESTION = 10
EXPECTED_ROWS = EXPECTED_QUESTIONS * RESPONSES_PER_QUESTION


def load_config(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("manual audit config must be a mapping")
    return value


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_generation_cells(config: dict[str, Any], run_root: Path) -> dict[str, list[dict[str, Any]]]:
    cells: dict[str, list[dict[str, Any]]] = {}
    for cell in config["cells"]:
        label = str(cell["label"])
        directory = run_root / "generation" / label
        responses = directory / "responses.jsonl"
        summary_path = directory / "summary.json"
        if not (directory / "CELL_COMPLETE").is_file() or not responses.is_file() or not summary_path.is_file():
            raise RuntimeError(f"generation cell is incomplete: {label}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows = load_jsonl(responses)
        if len(rows) != EXPECTED_QUESTIONS or summary.get("responses_sha256") != file_sha256(responses):
            raise RuntimeError(f"generation cell integrity failed: {label}")
        cells[label] = rows
    return cells


def build_audit_rows(
    config: dict[str, Any],
    ledger: list[dict[str, Any]],
    generation_cells: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if len(ledger) != EXPECTED_QUESTIONS or len({row["question_id"] for row in ledger}) != EXPECTED_QUESTIONS:
        raise ValueError("GPQA ledger must contain exactly 126 unique questions")
    by_cell = {
        label: {str(row["question_id"]): row for row in rows}
        for label, rows in generation_cells.items()
    }
    results: list[dict[str, Any]] = []
    for question_index, question in enumerate(ledger):
        question_id = str(question["question_id"])
        for response_index, cell in enumerate(config["cells"]):
            label = str(cell["label"])
            response = by_cell[label].get(question_id)
            if response is None:
                raise ValueError(f"missing question {question_id} from {label}")
            audit_row_id = f"gpqa_freeform126_q{question_index:03d}_{label}"
            row = {
                "audit_row_id": audit_row_id,
                "question_index": question_index,
                "response_index": response_index,
                "question_id": question_id,
                "variant": str(cell["variant"]),
                "global_step": int(cell["global_step"]),
                "question": str(question["question"]),
                "reference_answer": str(question["reference_answer"]),
                "raw_response": str(response["raw_response"]),
                "parsed_candidate_answer": response.get("parsed_answer"),
                "normalized_exact_match_auxiliary": (
                    normalize_answer_text(response.get("parsed_answer"))
                    == normalize_answer_text(question["reference_answer"])
                ),
                "generated_tokens": int(response["generated_tokens"]),
                "finish_reason": str(response["finish_reason"]),
                "cap_hit": bool(response["cap_hit"]),
                "request_seed": int(response["request_seed"]),
                "source_generation_receipt_sha256": str(response["generation_receipt_sha256"]),
            }
            row["audit_payload_sha256"] = canonical_json_sha256(row)
            results.append(row)
    if len(results) != EXPECTED_ROWS or len({row["audit_row_id"] for row in results}) != EXPECTED_ROWS:
        raise ValueError("manual audit rows do not close to 1,260 unique identities")
    return results


def indented(value: str) -> str:
    return "\n".join(f"    {line}" for line in value.splitlines()) or "    <empty>"


def write_question_chunks(output: Path, rows: list[dict[str, Any]]) -> None:
    chunks = output / "question_chunks"
    chunks.mkdir(parents=True)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["question_index"]), []).append(row)
    if sorted(grouped) != list(range(EXPECTED_QUESTIONS)):
        raise ValueError("question chunk indices are not exact 0..125")
    for question_index, question_rows in grouped.items():
        if len(question_rows) != RESPONSES_PER_QUESTION:
            raise ValueError(f"question {question_index} has {len(question_rows)} responses")
        first = question_rows[0]
        lines = [
            f"# GPQA-Freeform Manual Audit - Question {question_index:03d}",
            "",
            f"Question ID: `{first['question_id']}`",
            "",
            "## Question",
            "",
            indented(first["question"]),
            "",
            "## Reference Answer",
            "",
            indented(first["reference_answer"]),
        ]
        for row in question_rows:
            lines.extend(
                [
                    "",
                    f"## {row['variant']} - global step {row['global_step']}",
                    "",
                    f"Audit row: `{row['audit_row_id']}`",
                    f"Tokens: `{row['generated_tokens']}` | finish: `{row['finish_reason']}` | cap hit: `{row['cap_hit']}`",
                    f"Parsed tag (auxiliary only): `{html.escape(str(row['parsed_candidate_answer']))}`",
                    "",
                    "### Complete Candidate Response",
                    "",
                    indented(row["raw_response"]),
                ]
            )
        (chunks / f"question_{question_index:03d}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def instructions() -> str:
    return """# GPQA-Diamond-Freeform-126 Full Manual Audit Instructions

This package contains 1,260 responses: 126 questions x two architectures x five checkpoints.
`audit_rows.jsonl` is canonical. Read all 126 `question_chunks/question_NNN.md` files in order.

## Hard Procedure

Read every candidate response completely and compare it with the question and reference answer. Do not use
the extracted tag, normalized exact-match flag, regex, or another model as a substitute for reading. For each
question, write exactly ten verdict rows to `audit_staging/question_NNN.verdicts.jsonl`, then run the repository
merge/validation helper so `verdicts.jsonl` and `progress.json` remain resumable.

Every verdict must contain:

```json
{
  "audit_row_id": "gpqa_freeform126_q000_triglu_step158",
  "response_read_completely": true,
  "semantic_correctness": "correct | incorrect | uncertain",
  "answer_extraction": "tag_consistent | tag_missing_but_answer_present | tag_wrong_or_conflicting | no_answer | uncertain",
  "failure_category": "correct | factual_error | reasoning_error | contradictory_answer | insufficient_or_vague | truncated | malformed_or_no_answer | reference_or_question_ambiguity | other",
  "confidence": "high | medium | low",
  "evidence_note": "Concise row-specific reason grounded in the response and reference."
}
```

Use `uncertain` when equivalence, domain correctness, or question/reference validity cannot be established with
high confidence. Never force uncertainty into correct/incorrect. A response may be semantically correct despite
a missing `<answer>` tag; record correctness and extraction status separately.

## Completion Gates

- exactly 1,260 verdicts and 1,260 unique canonical IDs;
- exactly ten verdicts per question and all 126 question IDs completed;
- no missing required fields and only approved enums;
- separate `review_uncertain.jsonl` and `review_disagreements.jsonl`;
- category totals and every per-cell denominator close exactly;
- source generations and `audit_rows.jsonl` remain immutable.
"""


def validate_package(output: Path) -> dict[str, Any]:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    rows = load_jsonl(output / "audit_rows.jsonl")
    counts = Counter(int(row["question_index"]) for row in rows)
    if len(rows) != EXPECTED_ROWS or len({row["audit_row_id"] for row in rows}) != EXPECTED_ROWS:
        raise RuntimeError("audit package row/identity gate failed")
    if counts != Counter({index: RESPONSES_PER_QUESTION for index in range(EXPECTED_QUESTIONS)}):
        raise RuntimeError("audit package question balance gate failed")
    if manifest.get("audit_rows_sha256") != file_sha256(output / "audit_rows.jsonl"):
        raise RuntimeError("audit package hash gate failed")
    if len(list((output / "question_chunks").glob("question_*.md"))) != EXPECTED_QUESTIONS:
        raise RuntimeError("audit package chunk-count gate failed")
    return manifest


def build(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    config = load_config(args.config.resolve())
    run_root = args.run_root.resolve()
    output = args.output.resolve()
    if (output / "PACKAGE_READY").is_file():
        print(json.dumps(validate_package(output), indent=2, sort_keys=True))
        return
    if output.exists():
        raise RuntimeError(f"refusing to overwrite incomplete audit package: {output}")
    staging = output.with_name(output.name + ".building")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    ledger_path = resolve(root, config["dataset"]["ledger"])
    ledger = load_jsonl(ledger_path)
    generation_cells = load_generation_cells(config, run_root)
    rows = build_audit_rows(config, ledger, generation_cells)
    write_jsonl_atomic(staging / "audit_rows.jsonl", rows)
    write_question_chunks(staging, rows)
    (staging / "audit_staging").mkdir()
    (staging / "verdicts.jsonl").write_text("", encoding="utf-8")
    (staging / "review_uncertain.jsonl").write_text("", encoding="utf-8")
    (staging / "review_disagreements.jsonl").write_text("", encoding="utf-8")
    (staging / "README_AUDIT_INSTRUCTIONS.md").write_text(instructions(), encoding="utf-8")
    progress = {
        "status": "AWAITING_MANUAL_AUDIT",
        "expected_questions": EXPECTED_QUESTIONS,
        "expected_verdicts": EXPECTED_ROWS,
        "completed_question_indices": [],
        "verdict_count": 0,
    }
    write_json_atomic(staging / "progress.json", progress)
    source_cells = {
        label: file_sha256(run_root / "generation" / label / "responses.jsonl")
        for label in generation_cells
    }
    manifest = {
        "status": "AGENT_READ_READY",
        "run_id": config["run_id"],
        "rows": len(rows),
        "unique_audit_row_ids": len({row["audit_row_id"] for row in rows}),
        "questions": EXPECTED_QUESTIONS,
        "responses_per_question": RESPONSES_PER_QUESTION,
        "ledger_path": str(ledger_path),
        "ledger_sha256": file_sha256(ledger_path),
        "audit_rows_sha256": file_sha256(staging / "audit_rows.jsonl"),
        "source_cell_sha256": source_cells,
        "question_ids_sha256": hashlib.sha256(
            json.dumps([row["question_id"] for row in ledger], separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    write_json_atomic(staging / "manifest.json", manifest)
    (staging / "PACKAGE_READY").write_text("ready\n", encoding="utf-8")
    staging.replace(output)
    print(json.dumps(validate_package(output), indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("action", choices=("build", "validate"))
    args = parser.parse_args()
    if args.action == "build":
        build(args)
    else:
        print(json.dumps(validate_package(args.output.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
