# Training-Mix-Weighted Evaluation Composite Record / 中英混搭版

<!-- bilingual-companion-v1 -->

> **English original / 英文原件：** [2026-07-13_training-mix-weighted-eval-composite-record.md](../2026-07-13_training-mix-weighted-eval-composite-record.md)
> **中文导读 / Bilingual guide：** 这份 companion 的主题是 **Training-Mix-Weighted Evaluation Composite Record**。下面完整保留 original report 的 metrics、tables、commands、paths、hashes 与 scientific claims，同时加入中文 narrative layer 和 bilingual section handles，方便项目负责人阅读与后续 Agent 检索。
> **Evidence contract / 证据契约：** English original 保持 immutable；如果 companion 与原件出现 wording tension，以原件中的 exact numbers、machine-readable artifacts 和 source paths 为准，companion 不得 retroactively 改写实验事实。


Date: 2026-07-13

Status: ANALYSIS COMPLETE

## 目的 / Purpose
This record defines a single diagnostic composite for comparing the current
Qwen3-1.7B variants against the capability mix represented by the materialized
NuminaMath-CoT 50K training set. It preserves the raw benchmark scores as the
primary evidence. The composite is a model-selection aid, not a replacement
for per-benchmark reporting and not a claim of universal mathematical ability.

The active GRPO wave is unchanged by this analysis. No training data, data
order, checkpoint, generation, review, or evaluation protocol was modified.

## Authoritative Training Ledger

The counts come from the 50,000 `train` rows in:

```text
data_manifests/numina_math_cot_50k_decontam_v3/selected_rows.tsv
```

The ledger SHA-256 pinned by its manifest is:

```text
1097fdde429daf60eca6cbfb9b4e9f2f49ca5386133bd6618155087069a3968d
```

The exact training-source counts are:

| Source category | Rows | Share of 50K |
|---|---:|---:|
| `amc_aime` | 227 | 0.454% |
| `aops_forum` | 1,882 | 3.764% |
| `cn_k12` | 16,691 | 33.382% |
| `gsm8k` | 215 | 0.430% |
| `math` | 454 | 0.908% |
| `olympiads` | 8,859 | 17.718% |
| `orca_math` | 8,918 | 17.836% |
| `synthetic_amc` | 3,967 | 7.934% |
| `synthetic_math` | 8,787 | 17.574% |
| **Total** | **50,000** | **100.000%** |

The source-family mapping used for the four direct benchmark proxies is:

| Benchmark proxy | Training source family | Rows | Share of 50K |
|---|---|---:|---:|
| AMC | `amc_aime + synthetic_amc` | 4,194 | 8.388% |
| MATH-500 | `math + synthetic_math` | 9,241 | 18.482% |
| GSM8K | `gsm8k` | 215 | 0.430% |
| OlympiadBench | `olympiads` | 8,859 | 17.718% |
| **Directly mapped total** |  | **22,509** | **45.018%** |

The remaining 27,491 rows, or 54.982%, are:

| Remaining category | Rows | Share of 50K | Share of remainder | Interpretation |
|---|---:|---:|---:|---|
| `cn_k12` | 16,691 | 33.382% | 60.715% | Curriculum/K12-heavy; includes high-school exercises and is not uniformly elementary. |
| `orca_math` | 8,918 | 17.836% | 32.439% | Grade-school and elementary-style mathematical word problems. |
| `aops_forum` | 1,882 | 3.764% | 6.846% | Competition-leaning but heterogeneous; not equivalent to a pure Olympiad set. |

The taxonomy is a source-level approximation. The official NuminaMath
description spans Chinese high-school exercises, online mathematics forums,
and US/international Olympiad problems. The source names do not provide a
per-problem difficulty label.

Reference URLs:

- <https://huggingface.co/datasets/AI-MO/NuminaMath-CoT/blob/main/README.md>
- <https://github.com/project-numina/aimo-progress-prize#datasets>
- <https://huggingface.co/papers/2402.14830>
- <https://artofproblemsolving.com/wiki/index.php/Mathematics_forums>

## AMC Protocol Decision

The paper-primary AMC result is Average@32, but the paper does not publish
enough sampling detail to reconstruct the temperature, top-p, or equivalent
decode distribution confidently. A sampled Average@32 score can therefore mix
model capability with an underspecified sampling protocol.

For this local composite, AMC uses deterministic greedy pass@1. This aligns its
decode class with the greedy MATH-500, GSM8K, and OlympiadBench evaluations and
measures the model's modal answer path more directly. AMC Average@32 remains a
separately reported paper-aligned result and must not be deleted or silently
substituted in the raw evaluation table. It is excluded only from this local
composite.

## Two-Level Capability Taxonomy

The complete 50K mix is grouped into two broad capability domains.

Foundation/curriculum-heavy:

```text
cn_k12 + orca_math + gsm8k
= 16,691 + 8,918 + 215
= 25,824 rows
= 51.648% of the training set
```

Competition-heavy or competition-leaning:

```text
AMC family + MATH family + olympiads + aops_forum
= 4,194 + 9,241 + 8,859 + 1,882
= 24,176 rows
= 48.352% of the training set
```

This yields an approximately balanced 51.648% foundation and 48.352%
competition training mix.

## Proxy-Weight Construction

GSM8K is the only current benchmark that directly probes the foundation-style
word-problem domain. It therefore proxies the complete foundation bucket:

```text
w_GSM8K = 25,824 / 50,000 = 0.51648
```

The three mapped competition families contain:

```text
N_mapped_competition = 4,194 + 9,241 + 8,859 = 22,294
```

There is no dedicated `aops_forum` evaluation cell. Its 1,882-row weight is
therefore distributed across AMC, MATH-500, and OlympiadBench in proportion to
their existing mapped competition-family counts. Equivalently, each internal
competition weight is multiplied by the complete competition-domain mass:

```text
w_AMC = (24,176 / 50,000) * (4,194 / 22,294)
      = 0.0909609258

w_MATH = (24,176 / 50,000) * (9,241 / 22,294)
       = 0.2004220113

w_Olympiad = (24,176 / 50,000) * (8,859 / 22,294)
           = 0.1921370629
```

The final effective weights are:

| Benchmark score | Effective weight |
|---|---:|
| GSM8K greedy | 51.6480% |
| MATH-500 greedy | 20.0422% |
| OlympiadBench greedy | 19.2137% |
| AMC greedy pass@1 | 9.0961% |
| **Total** | **100.0000%** |

The composite for model `m` is:

```text
S_m = 0.51648 * GSM8K_m
    + 0.2004220113 * MATH500_m
    + 0.1921370629 * OlympiadBench_m
    + 0.0909609258 * AMC_greedy_m
```

All benchmark inputs are percentage accuracies on a 0-to-100 scale.

## Step-20 Calculation

The completed six-GPU step-20 evaluations provide:

| Model | GSM8K greedy | MATH-500 greedy | OlympiadBench greedy | AMC greedy pass@1 |
|---|---:|---:|---:|---:|
| TriGLU-20 | 83.5481 | 64.0000 | 25.4815 | 37.5000 |
| Baseline-20 | 82.8658 | 63.0000 | 25.7778 | 37.5000 |

TriGLU-20:

```text
S_TriGLU
= 0.51648      * 83.5481
+ 0.2004220113 * 64.0000
+ 0.1921370629 * 25.4815
+ 0.0909609258 * 37.5000
= 64.2849
```

Baseline-20:

```text
S_Baseline
= 0.51648      * 82.8658
+ 0.2004220113 * 63.0000
+ 0.1921370629 * 25.7778
+ 0.0909609258 * 37.5000
= 63.7890
```

Current difference:

```text
64.2849 - 63.7890 = +0.4959 percentage points for TriGLU-20
```

For transparency, two additional diagnostics are retained:

| Aggregation | TriGLU-20 | Baseline-20 | Difference |
|---|---:|---:|---:|
| Unweighted four-benchmark mean, AMC greedy | 52.6324 | 52.2859 | +0.3465 |
| Direct-family-only weighted mean, 22,509 mapped rows | 44.0891 | 43.7887 | +0.3004 |
| Whole-50K two-level proxy composite | 64.2849 | 63.7890 | +0.4959 |

The difference is at the previously discussed approximately 0.5-point boundary
and is not robust evidence by itself. The step-98 evaluation is required before
using this diagnostic in architecture selection.

## 假设与边界 / Assumptions And Limits
1. Every training row contributes equal mass; token count, solution quality,
   difficulty, and reward informativeness are not used as weights.
2. GSM8K proxies `cn_k12` and `orca_math` even though those sources cover a
   broader curriculum and are not identical to GSM8K.
3. `aops_forum` is treated as competition-leaning and proportionally allocated
   across three competition benchmarks. The source itself is heterogeneous.
4. Source categories are mutually exclusive metadata labels, but benchmark
   capability domains overlap conceptually.
5. Greedy AMC improves decode comparability but does not reproduce the paper's
   underspecified Average@32 protocol.
6. A single composite can hide regressions. Every report must keep all four raw
   scores visible next to the composite.
7. This weighting measures alignment with the selected training mix. It is not
   necessarily the right utility weighting for deployment or general-purpose
   mathematical reasoning.

## 报告契约 / Reporting Contract
Future milestone summaries may report this value as:

```text
Whole-50K training-mix proxy composite (AMC greedy pass@1)
```

They must also report the four component scores, identify the checkpoint and
decode protocol, and keep AMC Average@32 visible as a separate paper-aligned
diagnostic when available. Any change to the category mapping or proxy
allocation creates a new metric version and must not overwrite this record.

## Human-readable Monitor 集成 / Human-Readable Monitor Integration
The six-GPU GRPO monitor now computes this composite directly from each
milestone evaluation directory and prints a matched-step comparison after the
raw benchmark table. AMC Average@32 remains visible in the table, but only AMC
greedy pass@1 enters the composite. A milestone with any missing component is
reported as `pending`; partial benchmark coverage is never promoted to a
complete weighted score.

Expected completed step-20 line:

```text
step 20: TriGLU=64.2849  baseline=63.7890  delta=+0.4959 pp
```

The same logic automatically emits a distinct step-98 comparison once both
step-98 evaluations contain all four required component scores.

## 2026-07-14 MathAvg Reactivation Addendum

The weighted proxy above remains valid under its original name and formula,
but it is a secondary training-distribution-alignment metric. It does not
replace the primary cross-model quality aggregate:

```text
MathAvg = 0.25 * (GSM8K + MATH-500 + OlympiadBench + AMC Average@32)
```

AMC greedy pass@1 is deliberately excluded from `MathAvg`. It remains the AMC
input to the separately named Whole-50K weighted proxy and remains visible as a
standalone modal-path diagnostic. Historical weighted calculations in this
record are therefore preserved exactly; no prior score is silently redefined.

The monitor integration introduced in commit `dc0137f6` now emits both
aggregates and keeps their AMC protocols explicit. Current complete values are:

| Checkpoint | MathAvg | Weighted proxy | AMC greedy |
|---|---:|---:|---:|
| Baseline step 20 | 48.6771 | 63.7890 | 15/40 |
| TriGLU step 20 | 48.7262 | 64.2849 | 15/40 |
| TriGLU step 98 | 51.3401 | 64.0703 | 15/40 |

These values show why both aggregates are retained: TriGLU's primary MathAvg
improves by `+2.6139` points from step 20 to step 98 while the weighted proxy is
nearly flat and AMC greedy remains exactly `15/40`. Every decision report must
show the four raw benchmark scores beside both aggregates.

## 2026-07-14 AMC-Selective-Gain Interpretation

The step-20 to step-98 TriGLU change is concentrated in sampled AMC rather than
distributed uniformly across the four benchmarks:

| Benchmark | Step 20 | Step 98 | Change |
|---|---:|---:|---:|
| AMC Average@32 | 21.8750% | 32.0313% | +10.1563 pp |
| AMC greedy pass@1 | 37.5000% | 37.5000% | 0.0000 pp |
| GSM8K | 83.5481% | 82.7142% | -0.8340 pp |
| MATH-500 | 64.0000% | 63.8000% | -0.2000 pp |
| OlympiadBench | 25.4815% | 26.8148% | +1.3333 pp |

This motivates two registered hypotheses. They are interpretations to test,
not established causal conclusions.

### Hypothesis 1: sampled-policy improvement before greedy-mode movement

GRPO updates are estimated from sampled response groups and increase the
probability of rewarded trajectories relative to weaker trajectories in the
same group. It does not literally require every possible completion to score
well, but it can broaden or redistribute useful probability mass in ways that
a single greedy decode does not reveal. The large AMC Average@32 improvement
with unchanged AMC greedy is therefore consistent with the model assigning
more probability to correct AMC-like solution paths without yet changing its
modal answer. This would make a sampled metric especially sensitive to early
GRPO gains.

This interpretation must be separated from sampling noise or protocol effects.
A confirming test should hold prompts, seeds, temperature, top-p, sample count,
extractor, and verifier fixed, then compare per-item success count, pass@k,
correct-answer probability or sampled reward distribution, and greedy pass@1
across checkpoints.

### Hypothesis 2: the proxy taxonomy underweights K12-to-AMC transfer

The version-1 weighted proxy maps the 51.648% foundation/curriculum-heavy mass
mostly to GSM8K and gives AMC only 9.096% weight. That mapping treats source
provenance as if it were a mutually exclusive capability assignment. In
practice, K12 and curriculum-style training can strengthen arithmetic,
algebra, geometry, counting, and disciplined multi-step reasoning that transfer
directly into AMC problems. AMC is competition-formatted and extends beyond
routine K12 work, but much of its content remains close enough to those
foundations that the base model's existing logic can convert stronger
foundations into sampled AMC gains.

The current proxy can therefore remain a transparent version-1
source-alignment diagnostic, but it should not be interpreted as the true
capability contribution of each source family. A future version-2 proxy should
allow overlapping source-to-capability attribution, especially
`K12/curriculum -> both GSM8K and AMC`, and should preregister those weights
from content labels, difficulty analysis, or source ablations rather than tune
them retrospectively to favor the observed result. Until then, primary
selection remains the four-task MathAvg plus the raw benchmark vector.
