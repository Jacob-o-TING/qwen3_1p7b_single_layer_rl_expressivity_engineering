# 2026-07-11 SHS AMC Full Semantic Audit Record

## Scope And Integrity

A Terra agent completed a secondary semantic audit of all 1,280 SHS AMC23
responses (40 questions times 32 samples). The canonical input and row-level
verdicts remain outside Git under:

```text
audit_inputs/agent_read_ready_amc_shs_v1/
```

An independent integrity check confirmed:

- 1,280 input rows and 1,280 verdict rows;
- 1,280 unique input IDs and 1,280 unique verdict IDs;
- zero missing IDs, extra IDs, or duplicate verdicts;
- all 40 question groups completed with 32 responses each;
- zero null required verdict fields;
- all rows marked `response_read_completely=true`.

## Semantic Results

| Mathematical correctness | Count |
|---|---:|
| Correct | 184 |
| Incorrect | 1,094 |
| Uncertain | 2 |

The original EvalScope scorer marked 172 of 1,280 responses correct (13.4375%).
The semantic audit has 184 definite-correct responses (14.3750%). Depending on
the two uncertain rows, its possible accuracy range is 14.3750% to 14.5313%, a
gain of only 0.9375 to 1.0938 percentage points over the original score.

The row-level comparison found 15 definite score disagreements:

- 14 rows that Terra judged correct but EvalScope scored incorrect;
- one row that Terra judged incorrect but EvalScope scored correct;
- two additional rows remained uncertain, one originally correct and one
  originally incorrect.

These are secondary semantic judgments, not replacements for the primary
paper-faithful EvalScope result. Symbolic or problem-interpretation disputes in
the disagreement queue still warrant targeted human review before any corrected
score is reported as authoritative.

## Extraction Review

| Extraction verdict | Count |
|---|---:|
| Correct | 999 |
| Wrong span | 201 |
| Missed present answer | 14 |
| No answer existed | 64 |
| Uncertain | 2 |

Failure categories assigned by the audit were:

| Failure category | Count |
|---|---:|
| Math error | 826 |
| Correct | 172 |
| Extractor miss | 8 |
| Malformed or missing final answer | 82 |
| Incoherent or other | 114 |
| Contradictory multiple answers | 48 |
| Truncated before answer | 30 |

The extraction/failure labels are not mutually identical to mathematical
correctness: a mathematically correct response can still be categorized by an
extractor failure, and a present extracted span can be mathematically wrong.

## Conclusion

Full semantic re-reading recovers only about one percentage point of AMC
accuracy. Extraction imperfections are real, but they do not explain the large
gap between the SHS SFT AMC result and the paper's reported base-model AMC
score. The whole-layer baseline AMC evaluation remains the more decisive
control for separating evaluator/protocol effects from architecture/training
effects.
