# 2026-07-11 Baseline MATH-500 Partial Comparison Record

Date: 2026-07-11

## Snapshot Scope

This is a read-only progress snapshot of the actively growing whole-layer
baseline MATH-500 review file. It is not the final score. The snapshot contained
352 complete rows with contiguous dataset indices 0 through 351. Because this
is dataset order rather than a random sample, confidence intervals are
descriptive and must not be used to declare a final winner.

## Results

| Comparator | Correct | Rows | Accuracy |
|---|---:|---:|---:|
| Whole-layer baseline partial | 197 | 352 | 55.97% |
| SHS on the same 352 indices | 207 | 352 | 58.81% |
| SHS completed MATH-500 | 295 | 500 | 59.00% |
| Paper-reported Qwen3-1.7B untuned base | - | 500 | 57.40% |

The baseline partial Wilson 95% interval is 50.74% to 61.06%. Its partial
score is 2.84 percentage points below SHS on the exact same indices and 1.43
points below the paper-reported untuned base score. Completed SHS is 1.60
points above the paper base row.

## Paired Breakdown

| Outcome on the same item | Count |
|---|---:|
| Both correct | 178 |
| Both wrong | 126 |
| Baseline only correct | 19 |
| SHS only correct | 29 |

All 352 baseline rows had an extracted prediction. The current result is
consistent with baseline and SHS being close, with SHS provisionally ahead; it
does not yet establish a significant or final difference. The paper comparison
is orienting only because the paper does not publish a fully matching evaluator
recipe, and its 57.4 value is for the untuned base rather than this SFT run.

## Source

The live snapshot was read from:

```text
runs/sft_ordered_20260711_sft50k_v1/evaluations/
  layer10_whole_layer_baseline/main/20260711_160850/reviews/
  qwen3-1p7b-single-layer-sft/paper_math500_main.jsonl
```

No training, evaluation, checkpoint, prediction, or review artifact was
modified by this audit.

## Final MATH-500 Result

The completed hash-ready EvalScope report subsequently recorded 288 correct
answers from 500 rows, or 57.60% accuracy, with zero missing extractions. The
paired primary-score comparison is therefore:

| Model | Correct | Accuracy | Difference from baseline |
|---|---:|---:|---:|
| Whole-layer baseline SFT | 288/500 | 57.60% | - |
| SHS SFT | 295/500 | 59.00% | +1.40 pp |
| Paper-reported untuned base | - | 57.40% | -0.20 pp |
| Paper full-parameter GRPO | - | 64.00% | +6.40 pp |
| Paper Layer-10 single-layer GRPO | - | 68.60% | +11.00 pp |

The provisional 352-row ordering persisted directionally but narrowed from a
2.84-point paired gap to a 1.40-point aggregate gap. SHS leads by seven items;
the whole-layer baseline is effectively aligned with the paper's untuned-base
MATH-500 score under the available cross-protocol comparison. The paper value
remains orienting rather than controlled because its complete evaluator recipe
is unpublished.

For the requested RL comparison, the paper's Table 2 reports 64.0% MATH-500
for full-parameter GRPO and 68.6% for Layer-10 single-layer GRPO. The latter's
51.8 value is the unweighted average over MATH-500, GSM8K, OlympiadBench, and
AMC, not its MATH-500 score. Relative to the paper's Layer-10 GRPO result, the
local SHS SFT result is 9.6 points lower and the local whole-layer SFT baseline
is 11.0 points lower. These gaps should be interpreted as SFT-versus-GRPO
orientation, not as architecture-only effects.

After writing the MATH-500 report, the same serial evaluator advanced normally
to GSM8K with no reported runtime error.

## Final GSM8K Result

The baseline subsequently completed GSM8K with 1,033 correct answers from
1,319 rows, or 78.32% accuracy, with two missing extractions. The comparison is:

| Model | GSM8K accuracy | Difference from baseline |
|---|---:|---:|
| Whole-layer baseline SFT | 78.32% | - |
| SHS SFT | 76.95% | -1.37 pp |
| Paper-reported untuned base | 74.40% | -3.92 pp |
| Paper full-parameter GRPO | 82.00% | +3.68 pp |
| Paper Layer-10 single-layer GRPO | 80.50% | +2.18 pp |

Unlike MATH-500, the local whole-layer SFT baseline leads SHS on GSM8K. It also
closes most of the orienting gap from the paper base to the paper Layer-10 GRPO
result, while remaining 2.18 points below that RL result. The serial evaluator
advanced normally to OlympiadBench with no reported runtime error.

## Working Interpretation

The whole-layer SFT result of 57.60% is close to the paper-reported untuned-base
result of 57.40%, while SHS reaches 59.00% under the same local SFT data, seed,
ordering, schedule, and evaluator as the whole-layer baseline. SHS also reaches
a lower final validation loss, 0.620852 versus 0.636781 for the baseline.

A plausible working hypothesis is that the pretrained Layer-10 parameterization
is relatively insensitive to additional imitation-style SFT under this recipe,
whereas SHS supplies an alternative, more expressive adaptation geometry that
can absorb useful behavior without perturbing the initial function: its exact
no-op initialization preserves the base SwiGLU before training. The observed
seven-item MATH-500 gain is consistent with this hypothesis.

This is not yet evidence that Layer 10 is uniquely a "thinking layer," that the
pretrained model is at a capability ceiling, or that raw parameter count alone
causes the gain. SHS simultaneously changes function class, optimization
geometry, additive and multiplicative update pathways, and inductive bias. The
large gap to the paper's 68.60% Layer-10 GRPO result instead suggests a possible
SFT-signal ceiling: outcome-based RL can still extract substantially more from
the model than imitation SFT. TriGLU and OFT provide the next controlled tests
of expressivity versus update geometry.
