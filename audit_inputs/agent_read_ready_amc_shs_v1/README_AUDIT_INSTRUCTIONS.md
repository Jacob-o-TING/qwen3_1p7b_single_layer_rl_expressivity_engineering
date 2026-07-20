# AMC23 SHS Full Manual Audit Instructions

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
