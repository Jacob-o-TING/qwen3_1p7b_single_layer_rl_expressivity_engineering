# 2026-07-11 Baseline Completion, AMC Decode Controls, And TriGLU Record

Date: 2026-07-11

## Whole-Layer Baseline Completion

The whole-layer baseline completed its full primary evaluation at
2026-07-11 18:59:07 +08:00 with a verified completion receipt. The primary
comparison is:

| Benchmark | SHS SFT | Whole-layer baseline SFT | Baseline minus SHS |
|---|---:|---:|---:|
| MATH-500 | 59.00 | 57.60 | -1.40 pp |
| GSM8K | 76.95 | 78.32 | +1.37 pp |
| OlympiadBench | 22.96 | 23.11 | +0.15 pp |
| AMC Average@32, T=1.0 | 13.44 | 13.67 | +0.23 pp |
| Unweighted math average | 43.0875 | 43.1750 | +0.0875 pp |

The aggregate difference is effectively a tie. SHS redistributes performance
toward MATH-500, while the naive whole-layer baseline leads slightly on GSM8K,
OlympiadBench, and sampled AMC. The completed OlympiadBench comparison differs
by approximately one item and does not support a meaningful architecture gap.

## AMC Decode Controls

All four pre-TriGLU controls completed with hash-bound receipts:

| Model/control | Decode | Rows | AMC score |
|---|---|---:|---:|
| SHS SFT | Greedy pass@1 | 40 | 27.50% |
| Whole-layer baseline SFT | Greedy pass@1 | 40 | 32.50% |
| Untuned Qwen3-1.7B-Base | Greedy pass@1 | 40 | 30.00% |
| Untuned Qwen3-1.7B-Base | T=1.0 Average@32 | 1,280 | 18.83% |
| Whole-layer baseline SFT | T=1.0 Average@32 | 1,280 | 13.67% |
| SHS SFT | T=1.0 Average@32 | 1,280 | 13.44% |

The controls provide evidence for a decoding-distribution effect. The three
greedy scores remain in a relatively narrow 27.5% to 32.5% range, while both
SFT models lose approximately five points relative to the untuned base under
the local temperature-1.0 sampled protocol. The naive SFT baseline is best
under greedy decoding, whereas SHS is lowest; SHS therefore does not remedy the
AMC reliability issue under either registered decode in this wave.

This result is consistent with modal answer knowledge remaining partially
available while probability mass under sampling becomes less reliable after
SFT. It does not by itself establish flatter token-level entropy: the current
evidence is output-level accuracy under two decode policies. Trace-level answer
frequency, entropy, cap-hit, and semantic failure analyses remain necessary.

The local untuned-base Average@32 score of 18.83% is below the paper's reported
26.1% base AMC score. This reinforces the existing protocol caveat: the paper
does not publish a fully matching sampling and evaluator recipe, so local
within-harness comparisons are primary.

## TriGLU Milestone

After the controls, TriGLU completed all 3,916 SFT steps in 3,876.889 seconds
and saved the exact final checkpoint. Its final validation loss is 0.622954,
between SHS at 0.620852 and the whole-layer baseline at 0.636781. Median training
step time is approximately 0.987 seconds, close to the baseline's 0.948 seconds
and substantially faster than SHS at 1.483 seconds.

At the bounded status read, TriGLU had entered MATH-500 evaluation and completed
280 of 500 rows, with a provisional 164 correct answers or 58.57%. This partial
score is contiguous and must not be treated as final.

The primary ordered screen and the terminal OFT greedy waiter remained detached
and healthy. GPU memory was approximately 12.1 GB, the data volume retained
approximately 215 GB free, and no traceback or out-of-memory error was present.

## 2026-07-12 TriGLU Greedy Modal-Path Signal

The completed dashboard exposes a decoding-dependent TriGLU result that must
not be hidden by the four-benchmark aggregate:

| Model | AMC greedy pass@1 | Correct rows | AMC Average@32 |
|---|---:|---:|---:|
| TriGLU | 37.50% | 15/40 | 13.44% |
| Whole-layer baseline | 32.50% | 13/40 | 13.67% |
| Untuned base | 30.00% | 12/40 | 18.83% |
| SHS | 27.50% | 11/40 | 13.44% |

TriGLU has the strongest observed greedy/modal-path result despite not
improving the sampled Average@32 score or the four-task aggregate. A plausible
working hypothesis is that TriGLU makes a correct reasoning path more likely to
be the tokenwise mode without concentrating enough total probability mass on
correct trajectories under temperature-1.0 sampling. This may reflect improved
local preference with weak sequence-level calibration or robustness.

The 40-item greedy set is small: TriGLU exceeds baseline by only two correct
items, so this is architecture signal rather than a statistical win. Required
follow-up is a paired item audit plus a preregistered temperature sweep such as
`T=0.1/0.3/0.6/1.0`, recording answer entropy, response length, cap hits,
extraction failures, and correct-answer frequency. Architecture selection must
retain this greedy evidence separately from the paper-primary sampled average.
