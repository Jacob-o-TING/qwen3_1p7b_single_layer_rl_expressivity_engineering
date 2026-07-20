# 2026-07-11 SHS Response-Length Audit Record

## Scope

The approved Phase A post-hoc audit was run against the completed SHS final
evaluation under:

```text
runs/sft_ordered_20260711_sft50k_v1/evaluations/layer10_whole_layer_shs
```

The audit was CPU-only (`CUDA_VISIBLE_DEVICES` empty, `OMP_NUM_THREADS=2`,
`nice -n 15`). It loaded the pinned Qwen3 tokenizer locally, retokenized saved
response text, and read EvalScope prediction/review JSONL files. It did not load
a model or perform generation. Immediately after the audit, training GPU load
was 95% with 12,695 MiB allocated, while the whole-layer baseline had advanced
to step 1305/3916. No training interruption or slowdown was observed.

Source and tests were committed as `dce3da4` (`Add CPU-only response length
diagnostics`). Four remote focused tests passed before the accepted audit.

## SHS Results

| Benchmark | Samples | Cap proxy A | Cap proxy B | Non-cap C | Non-cap D | Cap-proxy rate | Missing extraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| AMC23 Average@32 | 1,280 | 40 | 16 | 1,158 | 66 | 4.38% | 6.41% |
| GSM8K | 1,319 | 10 | 1 | 1,306 | 2 | 0.83% | 0.23% |
| MATH-500 | 500 | 34 | 0 | 466 | 0 | 6.80% | 0.00% |
| OlympiadBench | 675 | 81 | 0 | 594 | 0 | 12.00% | 0.00% |

Cell definitions follow the approved plan: A is cap hit with an extracted
answer, B is cap hit without one, C is non-cap with an extracted answer, and D
is non-cap without one.

Across all 3,774 saved responses, 182 are cap-proxy hits. Of those, 165 already
have an extracted answer and 17 do not. Those 17 B-cell rows are the strongest
continuation candidates: 16 AMC samples and one GSM8K sample. The 68 D-cell
rows, predominantly AMC, support a separate formatting/extractor audit rather
than a token-budget explanation.

Accuracy among cap-proxy rows was 0% for AMC and GSM8K, 2.94% for MATH-500,
and 1.23% for OlympiadBench. This shows that merely extracting an answer before
the cap does not imply that the answer is correct. The high A-cell count may
represent wrong boxed answers followed by continued text, or a correct-looking
intermediate answer superseded by later reasoning; deterministic trace review
is required before assigning a failure mode.

## Length Distribution

| Benchmark | p50 | p90 | p95 | p99 | Max retokenized tokens |
|---|---:|---:|---:|---:|---:|
| AMC23 Average@32 | 629.0 | 1,556.7 | 2,851.8 | 3,072.0 | 4,607 |
| GSM8K | 247.0 | 375.2 | 429.1 | 724.2 | 3,093 |
| MATH-500 | 435.0 | 1,093.3 | 3,072.0 | 3,072.0 | 3,072 |
| OlympiadBench | 650.0 | 3,072.0 | 3,072.0 | 3,072.0 | 3,072 |

## Provenance Limitation

The current custom EvalScope adapter stores decoded text but not exact generated
token IDs or a trustworthy finish reason. Its `ModelOutput.from_content` path
reports `stop` even when generation may have exhausted `max_new_tokens`.
Therefore this audit labels cap status as `retokenized_text` and defines the cap
proxy as retokenized length greater than or equal to 3,072.

Decoded sampled AMC text can retokenize to more than 3,072 tokens; the 4,607
maximum must not be interpreted as the generator violating its 3,072-token
limit. Exact cap classification requires future evaluations to persist generated
token IDs, generated-token count, and finish reason at generation time.

The complete machine-readable outputs remain with the evaluation artifacts:

```text
diagnostics/response_length_diagnostics.json
diagnostics/response_length_diagnostics.md
diagnostics/response_length_rows.jsonl
```

## Evaluation Backend Finding

The current final-evaluation path is not vLLM. EvalScope dispatches up to eight
synchronous requests, and `QwenSingleLayerSFTModel` collects compatible requests
for 10 ms before running one padded `transformers.generate()` micro-batch. This
provides fixed microbatch concurrency but not vLLM continuous batching or
iteration-level scheduling. The baseline checkpoint therefore uses the same
evaluation implementation as SHS for protocol fairness; it may be somewhat
faster because its architecture is simpler, but it should not be expected to
receive a vLLM-scale speedup in the active wave.
