from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def message_content(record: dict[str, Any], role: str) -> str:
    for message in record.get("messages") or []:
        if message.get("role") == role:
            return str(message.get("content") or "")
    return ""


def normalize_row(
    review: dict[str, Any],
    cap_metadata: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    index = int(review["index"])
    sample_score = review.get("sample_score") or {}
    score = sample_score.get("score") or {}
    metadata = sample_score.get("sample_metadata") or {}
    group_id = int(sample_score["group_id"])
    sample_id = int(sample_score["sample_id"])
    cap = cap_metadata.get(index, {})
    return {
        "schema_version": 1,
        "audit_row_id": f"amc23_shs_g{group_id:02d}_i{index:04d}",
        "index": index,
        "sample_id": sample_id,
        "group_id": group_id,
        "problem_id": metadata.get("id", group_id),
        "source_url": metadata.get("url"),
        "prompt": message_content(review, "user"),
        "reference_target": review.get("target"),
        "response": message_content(review, "assistant"),
        "evalscope_extracted_prediction": score.get("extracted_prediction"),
        "evalscope_accuracy": ((score.get("value") or {}).get("acc")),
        "generated_tokens_retokenized": cap.get("generated_tokens"),
        "cap_hit_proxy": cap.get("cap_hit_proxy"),
        "boxed_answer_present_proxy": cap.get("boxed_answer_present"),
        "cap_provenance": cap.get("token_count_provenance"),
        "agent_verdict_template": {
            "response_read_completely": None,
            "final_answer_present": None,
            "independent_final_answer": None,
            "mathematical_correctness": None,
            "extraction_verdict": None,
            "failure_category": None,
            "confidence": None,
            "evidence_note": None,
        },
    }


def markdown_chunk(group_id: int, rows: list[dict[str, Any]]) -> str:
    first = rows[0]
    lines = [
        f"# AMC23 SHS Audit - Question {group_id:02d}",
        "",
        f"Source: {first['source_url']}",
        "",
        "## Problem",
        "",
        first["prompt"],
        "",
        "## Reference Target",
        "",
        f"`{first['reference_target']}`",
        "",
        "## Sampled Responses",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"### {row['audit_row_id']}",
                "",
                f"- `index`: {row['index']}",
                f"- `sample_id`: {row['sample_id']}",
                f"- `evalscope_extracted_prediction`: {json.dumps(row['evalscope_extracted_prediction'], ensure_ascii=False)}",
                f"- `evalscope_accuracy`: {row['evalscope_accuracy']}",
                f"- `generated_tokens_retokenized`: {row['generated_tokens_retokenized']}",
                f"- `cap_hit_proxy`: {row['cap_hit_proxy']}",
                "",
                "<response>",
                row["response"],
                "</response>",
                "",
                "<verdict-template>",
                json.dumps(row["agent_verdict_template"], ensure_ascii=False, indent=2),
                "</verdict-template>",
                "",
            ]
        )
    return "\n".join(lines)


def instructions_text() -> str:
    return """# AMC23 SHS Full Manual Audit Instructions

This package contains all 1,280 SHS AMC responses: 40 unique questions with 32
sampled responses per question. `audit_rows.jsonl` is canonical. The 40 files in
`question_chunks/` present the same rows in a more readable form.

## Required Procedure

Read every response completely. For each canonical `audit_row_id`, append
exactly one JSON object to a separate `verdicts.jsonl`. Never rewrite
`audit_rows.jsonl` and never change the primary EvalScope score.

Each verdict must contain:

```json
{
  "audit_row_id": "amc23_shs_g00_i0000",
  "response_read_completely": true,
  "final_answer_present": true,
  "independent_final_answer": "27",
  "mathematical_correctness": "correct | incorrect | uncertain",
  "extraction_verdict": "correct | missed_present_answer | wrong_span | no_answer_existed | uncertain",
  "failure_category": "correct | math_error | truncated_before_answer | malformed_or_missing_final_answer | extractor_miss | contradictory_multiple_answers | incoherent_or_other",
  "confidence": "high | medium | low",
  "evidence_note": "One concise, row-specific reason."
}
```

Process one question chunk at a time and checkpoint `verdicts.jsonl` after every
row. Maintain `progress.json` with completed question IDs and verdict count so
the audit is resumable. Do not use regex or a heuristic parser as a substitute
for reading the full response.

## Judgment Rules

- Compare the response with the problem and reference target, not only with
  EvalScope's extracted prediction.
- Treat mathematically equivalent forms as equivalent. Use `uncertain` rather
  than casually overruling symbolic equivalence.
- A response may contain a correct intermediate value but end with a different
  final answer. Record the final committed answer and note the contradiction.
- `cap_hit_proxy` is based on retokenized decoded text, not an exact generation
  finish reason. Use it as supporting metadata only.
- A missing extraction is not automatically a truncation. Distinguish a present
  answer missed by the extractor from a response that never states an answer.
- Do not infer correctness from `evalscope_accuracy`; independently read first,
  then record agreement or disagreement.

## Completion Gates

Before reporting completion, verify:

- exactly 1,280 verdicts;
- exactly 1,280 unique `audit_row_id` values;
- all 40 groups represented with 32 verdicts each;
- no null required verdict fields;
- category totals sum to 1,280;
- disagreements and `uncertain` rows are listed separately for review.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-jsonl", type=Path, required=True)
    parser.add_argument("--diagnostic-rows-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=1280)
    parser.add_argument("--expected-groups", type=int, default=40)
    parser.add_argument("--expected-repeats", type=int, default=32)
    args = parser.parse_args()

    cap_metadata = {
        int(row["index"]): row
        for row in read_jsonl(args.diagnostic_rows_jsonl)
        if row.get("benchmark") == "paper_amc23"
    }
    rows = [normalize_row(review, cap_metadata) for review in read_jsonl(args.review_jsonl)]
    rows.sort(key=lambda row: (row["group_id"], row["index"]))

    indices = [row["index"] for row in rows]
    row_ids = [row["audit_row_id"] for row in rows]
    group_counts = Counter(row["group_id"] for row in rows)
    if len(rows) != args.expected_rows:
        raise SystemExit(f"Expected {args.expected_rows} rows, found {len(rows)}")
    if len(set(indices)) != len(rows) or len(set(row_ids)) != len(rows):
        raise SystemExit("Duplicate index or audit_row_id detected")
    if len(group_counts) != args.expected_groups:
        raise SystemExit(f"Expected {args.expected_groups} groups, found {len(group_counts)}")
    bad_groups = {group: count for group, count in group_counts.items() if count != args.expected_repeats}
    if bad_groups:
        raise SystemExit(f"Unexpected per-group counts: {bad_groups}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir = args.output_dir / "question_chunks"
    chunks_dir.mkdir(exist_ok=True)
    canonical_path = args.output_dir / "audit_rows.jsonl"
    canonical_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["group_id"])].append(row)
    for group_id, group_rows in sorted(grouped.items()):
        (chunks_dir / f"question_{group_id:02d}.md").write_text(
            markdown_chunk(group_id, group_rows), encoding="utf-8"
        )

    (args.output_dir / "README_AUDIT_INSTRUCTIONS.md").write_text(
        instructions_text(), encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "dataset": "paper_amc23",
        "variant": "layer10_whole_layer_shs",
        "row_count": len(rows),
        "group_count": len(group_counts),
        "responses_per_group": args.expected_repeats,
        "canonical_jsonl": canonical_path.name,
        "canonical_jsonl_sha256": sha256(canonical_path),
        "source_review_jsonl": str(args.review_jsonl.resolve()),
        "source_review_sha256": sha256(args.review_jsonl),
        "source_diagnostic_rows_jsonl": str(args.diagnostic_rows_jsonl.resolve()),
        "source_diagnostic_rows_sha256": sha256(args.diagnostic_rows_jsonl),
        "chunk_files": [f"question_chunks/question_{group_id:02d}.md" for group_id in sorted(grouped)],
        "required_verdict_count": len(rows),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
