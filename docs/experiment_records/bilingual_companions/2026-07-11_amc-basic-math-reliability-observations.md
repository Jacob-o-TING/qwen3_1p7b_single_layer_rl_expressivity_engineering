# 2026-07-11 AMC Basic-Math Reliability Observations / 中英混搭版

<!-- bilingual-companion-v1 -->

> **English original / 英文原件：** [2026-07-11_amc-basic-math-reliability-observations.md](../2026-07-11_amc-basic-math-reliability-observations.md)
> **中文导读 / Bilingual guide：** 这份 companion 的主题是 **2026-07-11 AMC Basic-Math Reliability Observations**。下面完整保留 original report 的 metrics、tables、commands、paths、hashes 与 scientific claims，同时加入中文 narrative layer 和 bilingual section handles，方便项目负责人阅读与后续 Agent 检索。
> **Evidence contract / 证据契约：** English original 保持 immutable；如果 companion 与原件出现 wording tension，以原件中的 exact numbers、machine-readable artifacts 和 source paths 为准，companion 不得 retroactively 改写实验事实。


## 目的 / Purpose
This record preserves the competing interpretations raised during the SHS AMC
audit. New evidence should update their weight rather than silently deleting an
earlier hypothesis.

## Established Observations

The SHS AMC23 Average@32 primary score is 13.44%. A full semantic audit of all
1,280 responses found 184 definitely correct, 1,094 incorrect, and two
uncertain. Treating the uncertain rows as either all wrong or all correct gives
a semantic accuracy range of 14.38% to 14.53%. Re-reading and extraction repair
therefore recover only about one percentage point.

The extraction and mathematical-correctness axes are distinct:

| Extraction fidelity | Mathematical correctness | Count |
|---|---|---:|
| Correct extraction | Correct answer | 174 |
| Correct extraction | Incorrect answer | 825 |
| Wrong extracted span | Correct answer | 5 |
| Wrong extracted span | Incorrect answer | 196 |
| Missed present answer | Correct answer | 5 |
| Missed present answer | Incorrect answer | 9 |
| No answer existed | Incorrect answer | 64 |
| Uncertain | Uncertain | 2 |

Thus, in 825 rows the parser faithfully extracted an answer that was genuinely
mathematically wrong. Parser failure is real but is not the dominant source of
the AMC deficit.

Terra assigned the following primary failure modes:

| Failure mode | Count |
|---|---:|
| Math error | 826 |
| Correct | 172 |
| Extractor miss | 8 |
| Malformed or missing final answer | 82 |
| Incoherent or other | 114 |
| Contradictory multiple answers | 48 |
| Truncated before answer | 30 |

The single-label boundaries are not perfectly objective. A response containing
mojibake plus a recoverable wrong calculation could reasonably be classified as
either `math_error` or `incoherent_or_other`. Counts should be treated as a
structured audit judgment, not an immutable ground truth.

## Difficulty-Gradient Observation

The earlier difficulty hypothesis remains partially supported. Grouping the
selected AMC questions by published problem number gives:

| Problem-number bucket | Responses | Semantic correct | Math-error label |
|---|---:|---:|---:|
| 1-5 | 320 | 26.56% | 55.62% |
| 6-10 | 288 | 9.03% | 68.40% |
| 11-15 | 256 | 15.62% | 64.84% |
| 16-20 | 192 | 8.33% | 63.54% |
| 21-25 | 224 | 7.59% | 72.77% |

Later problems are generally less reliable, so problem difficulty still
matters. The non-monotonic middle buckets and the selected-question composition
prevent interpreting this as a calibrated psychometric curve.

Difficulty is nevertheless insufficient as the full explanation. The model
also fails elementary questions at an abnormal rate:

- 2023 AMC 12A Problem 1, a basic relative-speed problem, is semantically
  correct in only 20 of 32 samples. One response correctly obtains 1.5 hours
  and then evaluates `18 * 1.5` as 54.
- 2023 AMC 12B Problem 1, a basic equal-volume problem whose required transfer
  is `1/6` and whose requested answer is 7, is correct in 0 of 32 samples. The
  sampled final answers are widely dispersed.

These examples are too elementary to attribute to Olympiad-level problem
difficulty.

## Global-Forgetting Counterargument

The initial counterargument against global arithmetic forgetting should also be
preserved:

| Benchmark | SHS SFT | Paper base row |
|---|---:|---:|
| GSM8K | 76.95 | 74.4 |
| MATH-500 | 59.00 | 57.4 |
| OlympiadBench | 22.96 | 18.7 |
| AMC | 13.44 | 26.1 |

If the model had globally lost all basic arithmetic knowledge, an accompanying
GSM8K collapse would be expected. The three non-AMC aggregates argue against a
simple, universal loss of mathematical capability.

This counterargument is weaker than it first appeared because our local
protocols are not decoding-matched. Our GSM8K, MATH-500, and OlympiadBench runs
use greedy decoding; our AMC run uses temperature-1.0 sampling with 32 repeats.
The paper does not publish whether its three larger math benchmarks used greedy
or sampled decoding, nor does it publish the temperature, top-p, or other
generation settings behind AMC Average@32. Aggregate performance under our
greedy path does not prove that the sampled next-token distribution remains well
calibrated, and comparison to the paper row is not decoding-matched. The paper
base row also is not yet a same-harness, same-checkpoint before/after control.

## Current Competing Hypotheses

The evidence supports retaining, rather than prematurely choosing among, these
hypotheses:

1. **Problem difficulty:** harder AMC questions require more planning and expose
   more mathematical errors. Supported by the broad problem-number gradient,
   but unable to explain severe failures on Problem 1.
2. **Global catastrophic forgetting:** difficult-math SFT may have damaged basic
   skills. The easy-question failures support some form of interference, while
   greedy GSM8K/MATH performance argues against a universal capability loss.
3. **Sampling-distribution flattening:** the highest-probability reasoning path
   may remain correct while SFT or SHS moves substantial probability mass into
   wrong paths. Temperature-1.0 sampling then exposes failures that greedy
   decoding hides.
4. **Long-form overthinking and error accumulation:** Numina-style SFT may make
   the model expand simple problems into unnecessarily long chains, increasing
   the opportunities for arithmetic drift, contradiction, corruption, and
   post-answer revision.
5. **Architecture-specific instability:** SHS dynamic modulation may broaden or
   destabilize the token distribution. The whole-layer baseline AMC control is
   required before assigning this effect to SHS.
6. **SFT-recipe or protocol effect:** if the whole-layer baseline exhibits the
   same collapse, the common SFT data/schedule or AMC sampling protocol is more
   likely than the SHS architecture.
7. **Extraction and cap effects:** these contribute real errors but are already
   bounded by the full audit and cannot explain the majority of the deficit.

These mechanisms are not mutually exclusive. In particular, a model may retain
the correct modal computation while simultaneously becoming less calibrated,
more verbose, and less reliable under sampling.

## Current Interpretation

The strongest statement justified today is that SHS SFT exhibits a severe
basic-math reliability collapse under the current sampled AMC protocol. It is
not yet justified to claim either complete arithmetic forgetting or a purely
decoding-only failure. The baseline AMC result and a decoding-matched greedy
diagnostic are the next discriminating controls.

## Correct Answer Versus Reasoning Faithfulness

The strong local greedy scores on MATH-500, GSM8K, and OlympiadBench establish
that the SHS checkpoint retains meaningful mathematical signal under our
evaluator. These benchmarks generally require open-form numeric or symbolic
answers, so their aggregate accuracy cannot plausibly be explained by blind
random guessing.

Final-answer accuracy nevertheless does not establish that the visible chain of
thought is sound or causally responsible for the answer. A scored-correct trace
may contain:

1. a sound derivation;
2. a local error followed by genuine self-correction;
3. a valid shortcut or implicit pattern recognition with little explicit
   derivation;
4. flawed verbal reasoning attached to a correct answer selected from latent
   model knowledge;
5. contradictory reasoning whose final committed answer happens to be correct;
6. a rare accidental match.

This motivates a separate hypothesis: SHS may retain strong latent answer
intuition or a useful modal answer prior while exhibiting brittle explicit
reasoning, arithmetic instability, or unfaithful verbal rationalization. Greedy
decoding could expose the modal answer path while temperature-1.0 sampling more
frequently leaves that path.

The hypothesis should not be overstated as "no reasoning, only intuition."
MATH-500 accuracy of 59.00% and OlympiadBench accuracy of 22.96% on open-form
answers require substantial structured capability. The unresolved question is
how often the visible derivation is sound and transferable rather than how often
the model can blindly guess.

### Proposed Reasoning-Faithfulness Audit

For correct SHS MATH-500 and OlympiadBench responses, draw a deterministic,
stratified sample by benchmark, response length, problem difficulty proxy, and
cap status. Classify each complete trace as:

```text
sound_derivation
self_corrected_derivation
correct_shortcut_or_implicit_reasoning
flawed_reasoning_correct_answer
contradictory_reasoning_correct_answer
uncertain
```

Preserve the original final-answer score as primary and report these labels only
as secondary semantic diagnostics. A later number-perturbation control should
change quantities while preserving problem structure. Transfer of the same
method to perturbed instances supports algorithmic reasoning; collapse under
small perturbations supports memorized pattern or shallow answer intuition.

Visible chain of thought remains an imperfect causal probe even after semantic
review. The audit measures reasoning coherence and transfer evidence, not direct
access to the model's hidden computation.

## Safe Access To In-Progress EvalScope JSONL

EvalScope writes baseline AMC predictions and reviews as append-only JSONL. A
bounded inspection during the active run observed 828 prediction rows and 828
matching review rows, with both files ending in a complete LF newline. A reader
can safely access or snapshot a fixed prefix without locking, truncating, or
otherwise interfering with the writer.

A safe partial snapshot must:

- record the prediction and review line counts at snapshot start;
- use the smaller matching count as the fixed prefix length;
- copy only complete newline-terminated rows into a separate diagnostics path;
- verify JSON parsing, unique indices, and prediction/review index equality;
- record source paths, byte sizes, timestamps, and hashes;
- never modify or rename the primary files.

Partial rows are not an unbiased estimate of final accuracy. Microbatched
autoregressive evaluation tends to complete short responses before long ones,
and row order reflects completion/scheduling behavior. Partial snapshots are
valid for qualitative trace inspection and pipeline debugging, but final score
comparison must wait for all 1,280 reviews and the completed report.
