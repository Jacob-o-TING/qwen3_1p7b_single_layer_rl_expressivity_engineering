# 2026-07-17 TriGLU MGSM Failure-Mode Audit Record / 中英混搭版

<!-- bilingual-companion-v1 -->

> **English original / 英文原件：** [2026-07-17_triglu-mgsm-failure-mode-audit-record.md](../2026-07-17_triglu-mgsm-failure-mode-audit-record.md)
> **中文导读 / Bilingual guide：** 这份 companion 的主题是 **2026-07-17 TriGLU MGSM Failure-Mode Audit Record**。下面完整保留 original report 的 metrics、tables、commands、paths、hashes 与 scientific claims，同时加入中文 narrative layer 和 bilingual section handles，方便项目负责人阅读与后续 Agent 检索。
> **Evidence contract / 证据契约：** English original 保持 immutable；如果 companion 与原件出现 wording tension，以原件中的 exact numbers、machine-readable artifacts 和 source paths 为准，companion 不得 retroactively 改写实验事实。


Date: 2026-07-17

Status: **COMPLETE READ-ONLY AUDIT.** No model, prediction, review, report,
checkpoint, evaluator, or active training process was changed.

## 问题与范围 / Question And Scope
This audit investigates why the completed MGSM result for baseline step 196
(`59.017%`) is much higher than TriGLU step 294 (`46.000%`). The paired
population is the 2,750 MGSM rows shared by both evaluations. The detailed
failure-mode population is the 482 rows where baseline step 196 is correct and
TriGLU step 294 is wrong.

The owner's correction is important: this is not a failure isolated to one or
two unusual languages. TriGLU is lower across the major high-resource language
cells. Smaller gaps in some low-resource cells mainly reflect a floor effect,
not evidence that the failure is language-specific.

## Benchmark 难度 / Benchmark Difficulty
MGSM is a relatively simple benchmark by the standards of this project. It is
the same set of 250 grade-school arithmetic word problems from GSM8K translated
into ten languages, plus English. It primarily tests multilingual problem
comprehension, elementary multi-step arithmetic, and stable answer generation;
it is not comparable in mathematical difficulty to MATH-500 or OlympiadBench.

Primary benchmark source:
<https://arxiv.org/abs/2210.03057>.

The simplicity reduces concern about TriGLU's high-end mathematical ceiling,
but it makes the observed failures useful as a generation-stability and
language-conditioned routing diagnostic.

## 配对与完整性 / Pairing And Integrity
- Coverage is exactly 2,750 rows: 11 languages x 250 rows.
- User prompt and target are byte-identical for all 2,750 paired rows after
  removing message IDs and generated assistant content.
- Baseline step 196 is correct on 1,623 review rows; TriGLU step 294 is correct
  on 1,265.
- Baseline-only correct: 482 rows.
- TriGLU-only correct: 124 rows.
- Same outcome: 2,144 rows.
- The six baseline shards independently score between 56.34% and 61.31%, so
  the aggregate is not produced by one anomalous shard.

The official merged report is retained as `59.017%`. Counting binary review
rows gives `1623/2750 = 59.018%`; the 0.001 percentage-point display difference
comes from report precision and is immaterial.

## 逐语言准确率 / Accuracy By Language
| Language | Baseline-196 | TriGLU-294 | Delta | Baseline-only correct | TriGLU-only correct |
|---|---:|---:|---:|---:|---:|
| Bengali (`bn`) | 53.6% | 47.6% | +6.0 pp | 26 | 11 |
| German (`de`) | 73.2% | 46.8% | +26.4 pp | 73 | 7 |
| English (`en`) | 84.4% | 59.6% | +24.8 pp | 66 | 4 |
| Spanish (`es`) | 78.4% | 54.0% | +24.4 pp | 69 | 8 |
| French (`fr`) | 71.2% | 50.8% | +20.4 pp | 63 | 12 |
| Japanese (`ja`) | 52.0% | 48.8% | +3.2 pp | 27 | 19 |
| Russian (`ru`) | 72.8% | 53.6% | +19.2 pp | 56 | 8 |
| Swahili (`sw`) | 6.4% | 9.2% | -2.8 pp | 3 | 10 |
| Telugu (`te`) | 23.6% | 23.6% | 0.0 pp | 16 | 16 |
| Thai (`th`) | 61.6% | 60.4% | +1.2 pp | 24 | 21 |
| Chinese (`zh`) | 72.0% | 51.6% | +20.4 pp | 59 | 8 |

The main gap is broad across German, English, Spanish, French, Russian, and
Chinese (`+19.2` to `+26.4` percentage points). It is therefore inaccurate to
describe the failure as confined to a particular language.

## 四种已观察 Failure Modes / Four Observed Failure Modes
The four modes below are conceptual and can overlap. They must not be forced
into four percentages that sum to 100%.

1. **Repetition / prompt-replay degeneration.** The answer enters a
   high-autocorrelation loop, repeats tokens, equations, or few-shot examples,
   and often fails to terminate cleanly.
2. **Semantic or operation-binding corruption.** Quantities remain visible but
   their roles or entities are rebound incorrectly. Verified examples include
   eggs/muffins becoming chickens/rice and roses/thorns becoming frogs/legs.
3. **Correct intermediate value overwritten.** The correct target occurs as an
   exact numeric token near the end or in boxed content, but later reasoning
   changes the submitted answer.
4. **Direct arithmetic or reasoning error.** A non-degenerate trajectory
   selects the wrong arithmetic base, omits a required operation, or performs
   an otherwise incorrect short/extended solution.

Mode 2 is semantic and cannot be assigned a defensible full-corpus percentage
by regex. The report therefore preserves verified examples and reports the
language-neutral `concise non-degenerate wrong` rate as a **local-error proxy**,
not as a claimed exact semantic-corruption rate. Fabricating a precise semantic
label would be less scientific than preserving this boundary.

## 逐语言 Failure Indicators / Per-Language Failure Indicators
Denominator `n` is the baseline-only-correct population for that language.
Indicators are independently measured and may overlap:

- `Repeat`: repeated 4-gram ratio greater than 0.35, or at least 20 identical
  consecutive tokens.
- `Token loop`: at least 20 identical consecutive tokens.
- `Term/extract`: no extracted prediction, or at least 8,000 generated
  characters.
- `Value lost`: exact numeric target in the final 500 characters or boxed
  content, but the scored final answer is wrong.
- `Short local`: fewer than 1,500 characters and no repetition degeneration;
  this is the local semantic/arithmetic-error proxy.
- `Extended local`: at least 1,500 characters and no repetition degeneration.

| Lang | n | Repeat | Token loop | Term/extract | Value lost | Short local | Extended local |
|---|---:|---:|---:|---:|---:|---:|---:|
| `bn` | 26 | 23.1% | 0.0% | 3.8% | 26.9% | 69.2% | 7.7% |
| `de` | 73 | 64.4% | 20.5% | 15.1% | 5.5% | 26.0% | 9.6% |
| `en` | 66 | 66.7% | 31.8% | 16.7% | 7.6% | 18.2% | 15.2% |
| `es` | 69 | 66.7% | 23.2% | 15.9% | 7.2% | 21.7% | 11.6% |
| `fr` | 63 | 79.4% | 28.6% | 20.6% | 7.9% | 12.7% | 7.9% |
| `ja` | 27 | 22.2% | 7.4% | 11.1% | 25.9% | 55.6% | 22.2% |
| `ru` | 56 | 55.4% | 19.6% | 7.1% | 21.4% | 28.6% | 16.1% |
| `sw` | 3 | 0.0% | 0.0% | 0.0% | 66.7% | 100.0% | 0.0% |
| `te` | 16 | 87.5% | 0.0% | 6.3% | 12.5% | 12.5% | 0.0% |
| `th` | 24 | 58.3% | 20.8% | 0.0% | 16.7% | 37.5% | 4.2% |
| `zh` | 59 | 62.7% | 27.1% | 3.4% | 15.3% | 27.1% | 10.2% |
| **All** | **482** | **61.2%** | **21.6%** | **11.8%** | **12.9%** | **27.6%** | **11.2%** |

`Repeat`, `Short local`, and `Extended local` form an exhaustive mutually
exclusive partition of the 482 rows. The other indicators overlap that
partition. Small denominators for Swahili and Telugu make their percentages
unstable and unsuitable for strong language-level conclusions.

## 代表性 Trace 证据 / Representative Trace Evidence
- Bengali eggs problem, target `18`: TriGLU omits the four eggs used for
  muffins and submits `26`; baseline computes `(16 - 3 - 4) * 2 = 18`.
- Bengali house-flipping problem, target `70000`: TriGLU applies the 150%
  increase to purchase-plus-repair cost and submits `195000`; baseline applies
  the increase to the original house value and subtracts repair cost.
- German running-speed problem, target `10`: TriGLU initially sets up the
  correct six-hour denominator, then replays `Frage/Reasoning/ANSWER` examples
  until truncation; baseline submits `10 mph` directly.
- Bengali distance problem, target `25`: TriGLU begins correctly, then repeats
  numbered steps into the fifties and ends on an unrelated extracted value.

## 解读 / Interpretation
The primary differentiator is generation degeneration, not answer extraction:
61.2% of baseline-only failures meet the language-neutral repetition criterion.
Across all 2,750 rows, empty extraction occurs only 26 times for TriGLU and 10
times for baseline, which cannot explain the 358-row net accuracy gap.

The evidence is consistent with TriGLU preserving useful high-end math features
while making some language-conditioned hidden states more likely to enter a
high-autocorrelation decoding attractor. This is a hypothesis, not a proven
mechanism. A causal test should record Layer-10 main/side activation RMS,
side-to-main ratio, logit entropy, and repetition onset on the paired failure
set, ideally comparing the current FP32 side path with the pending pure-BF16
path.

## 持久证据 / Durable Evidence
Compact metrics:
`compact_metrics/2026-07-17_triglu_mgsm_failure_mode_audit.json`

Compact JSON SHA-256:
`46c49d0e2156a3e203cc419bb46b443c36019b9f225f8f2b14a5ccdcf2dbb4ca`

The compact file records source roots, row-level prompt/target and prediction
digests, exact thresholds, overall counts, and all per-language numerators and
denominators. It contains no model weights or full generations.

## 继承待办 / Pending Obligations Carried Forward
Canonical registry:
`docs/experiment_plans/PENDING_2026-07-12_mandatory-experiment-gates.md`

- **PENDING-01 Eval Parity Matrix:** deferred; this audit uses the completed
  vLLM MGSM cells and does not close HF/vLLM parity.
- **PENDING-02 Pure-BF16 SHS And TriGLU:** directly relevant follow-up, but not
  executed in this read-only audit.
- **PENDING-03 Registered SHS CausalLM Route:** deferred; SHS is out of scope.
