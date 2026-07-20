# Cap-Hit Continuation And Extraction Diagnostics Plan

Date: 2026-07-11

Status: approved planning artifact. This document does not modify or interrupt
the active paper-faithful evaluation wave.

## Objectives

Add two post-hoc analyses to the Qwen3-1.7B SFT evaluation pipeline:

1. Diagnose the relationship between responses that reach the 3072-token cap
   and responses for which EvalScope cannot extract a final answer.
2. Selectively continue cap-hit responses by prefilling their existing trace
   and generating an additional bounded segment, then measure how often the
   continuation rescues an otherwise incorrect or incomplete answer.

The original `accuracy_at_3072` remains the primary paper-faithful result. All
continuation results are secondary diagnostics and must never replace or be
mixed into the primary score.

## Required Source Artifacts

For every benchmark response, retain or reconstruct:

- Architecture variant and checkpoint identity.
- Benchmark, subset, problem ID, and repeat ID.
- Exact prompt text or its stable source-row identity.
- Full generated response text.
- EvalScope extracted prediction, reference answer, and score.
- Generated-token count under the pinned Qwen3 tokenizer.
- Whether generation ended by EOS or by the configured token cap.
- Whether a syntactically plausible boxed final answer occurs in the trace.

The current SHS evaluation stores full response text but not generated token
IDs or an explicit finish reason. Its length analysis may reconstruct tokens
from text and must label the prefix provenance as `retokenized_text`. Future
evaluations should additionally save exact generated token IDs, token count,
and finish reason so their provenance can be labeled `exact_generation`.

## Diagnostic 1: Cap Hit Versus Answer Extraction

Construct the following 2 x 2 table independently for each benchmark and
variant:

| | Extracted answer present | Extracted answer missing |
|---|---:|---:|
| Token cap hit | A | B |
| Token cap not hit | C | D |

Interpretation:

- A: an answer was extractable before generation reached the cap; this may
  indicate post-answer rambling and an opportunity for safe early stopping.
- B: the strongest candidate set for reasoning truncated before the final
  answer and for selective continuation.
- C: ordinary completed generations.
- D: likely formatting failures, refusal/rambling followed by EOS, empty
  output, or answer-extractor failure rather than insufficient token budget.

Report at least:

- Cap-hit rate.
- Missing-extraction rate.
- `P(missing extraction | cap hit)`.
- `P(cap hit | missing extraction)`.
- Accuracy for cap-hit and non-cap responses.
- Missing-box rate for cap-hit and non-cap responses.
- Generated-token p50, p90, p95, p99, and maximum.
- Accuracy by fixed response-length bucket.

Manually inspect a deterministic sample from cells B and D. Record whether each
trace ends in an unfinished sentence/formula, contains a malformed answer,
contains a valid answer missed by the extractor, or exhibits another failure
mode. Preserve selected row IDs and the sampling seed in the analysis manifest.

## Diagnostic 2: Selective Cap Continuation

The continuation experiment is named
`cap_continuation_3072_plus_3072_v1`.

Eligibility is determined only by generation termination/length:

```text
eligible = finish_reason == length at 3072 tokens
```

Do not select based on correctness, missing extraction, or the contents of the
reference answer. Outcome-conditioned selection would bias the rescue result.

For each eligible response:

1. Reconstruct the original prompt and exact assistant prefix where available.
2. Prefill `prompt + existing assistant trace` into the same variant and final
   checkpoint that produced the original trace.
3. Continue for at most another 3072 generated tokens with the benchmark's
   original greedy or sampling protocol.
4. Concatenate the original and continuation text without rewriting either
   segment.
5. Apply the same answer extractor and verifier used by the primary run.

For greedy MATH-500, GSM8K, and OlympiadBench, continuation conditioned on an
exact prefix should closely match a one-shot longer decode, subject to possible
small numerical differences caused by re-prefill. For sampled AMC responses,
derive a deterministic continuation seed from the experiment seed, problem ID,
repeat ID, and continuation round. This is a valid conditional sample but is
not claimed to be bitwise identical to a hypothetical one-shot 6144-token
sample.

## Continuation Metrics

Report separately for every benchmark and variant:

- Number and fraction eligible for continuation.
- Accuracy before continuation among eligible responses.
- Accuracy after continuation among eligible responses.
- Rescue count and rescue rate.
- Additional generated-token p50, p90, p99, and maximum.
- Fraction still reaching the second cap without an extracted answer.
- Fraction with an answer before the first cap that changes after continuation.
- Total continuation GPU time and generated tokens.

Define rescue rate as:

```text
rescued original cap-hit responses that were incorrect and become correct
--------------------------------------------------------------------------
all original cap-hit responses that were incorrect
```

Also preserve the original answer, continued answer, original score, continued
score, and both trace segments for every eligible response.

## Cross-Variant Fairness

Never compare variants only on each variant's independently filtered non-cap
subset. Different architectures may hit the cap on different problem
distributions, making those conditional sets incomparable.

Alongside per-variant diagnostics, construct a common-prompt intersection for
which all compared variants are non-cap. Report this only as a secondary
controlled subset and keep the full primary benchmark scores visible.

For AMC, preserve the 32-repeat structure. Report sample-level continuation
statistics and question-level aggregates; do not silently change the number of
effective samples per problem when comparing Average@32 results.

## Validation Gates

Before accepting the analysis:

- Verify tokenization against the pinned local Qwen3 tokenizer revision.
- Quantify decode-then-retokenize round-trip mismatches for current text-only
  traces.
- Confirm that no non-cap response enters the continuation set.
- Confirm that selection never reads correctness or reference-answer fields.
- Re-run extraction/scoring on unmodified primary traces and reproduce the
  original report scores exactly.
- Unit-test nested LaTeX boxes, malformed boxes, multiple boxes, empty output,
  EOS termination, and exact-cap termination.
- Save a machine-readable manifest with code revision, checkpoint hashes,
  tokenizer identity, cap values, seeds, and source artifact hashes.

## Deliverables

- `response_length_diagnostics.json` and a concise Markdown report.
- The cap-hit/extraction 2 x 2 tables by benchmark and variant.
- A row-level manifest for deterministic manual inspection samples.
- Continuation predictions and reviews under a separate artifact root.
- `cap_continuation_3072_plus_3072_v1` JSON and Markdown summaries.
- Cross-variant common-prompt diagnostics after all four primary evaluations
  finish.

Large prediction traces remain outside Git. Commit only source code, tests,
manifests, compact summaries, and experiment records.
