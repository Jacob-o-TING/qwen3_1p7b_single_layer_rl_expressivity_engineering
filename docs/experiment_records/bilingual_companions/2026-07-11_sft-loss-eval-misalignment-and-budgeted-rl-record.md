# 2026-07-11 SFT Loss/Eval Misalignment And Budgeted RL Record / 中英混搭版

<!-- bilingual-companion-v1 -->

> **English original / 英文原件：** [2026-07-11_sft-loss-eval-misalignment-and-budgeted-rl-record.md](../2026-07-11_sft-loss-eval-misalignment-and-budgeted-rl-record.md)
> **中文导读 / Bilingual guide：** 这份 companion 的主题是 **2026-07-11 SFT Loss/Eval Misalignment And Budgeted RL Record**。下面完整保留 original report 的 metrics、tables、commands、paths、hashes 与 scientific claims，同时加入中文 narrative layer 和 bilingual section handles，方便项目负责人阅读与后续 Agent 检索。
> **Evidence contract / 证据契约：** English original 保持 immutable；如果 companion 与原件出现 wording tension，以原件中的 exact numbers、machine-readable artifacts 和 source paths 为准，companion 不得 retroactively 改写实验事实。


Date: 2026-07-11

## Observed SFT Loss Versus External Evaluation

The first three completed SFT training runs show:

| Variant | Last-100 train loss | Final validation loss | Train/val gap |
|---|---:|---:|---:|
| Whole-layer baseline | 0.599223 | 0.636781 | 0.037558 |
| SHS | 0.584501 | 0.620852 | 0.036351 |
| TriGLU | 0.586106 | 0.622954 | 0.036848 |

SHS and TriGLU achieve lower token-level training and held-out SFT validation
loss than the naive whole-layer baseline. The train/validation gaps are nearly
identical, and validation loss did not exhibit a late upward reversal. This is
not the usual signature of classical within-distribution overfitting.

However, the completed SHS and baseline external math averages are effectively
tied: 43.0875% for SHS and 43.1750% for the baseline. SHS improves MATH-500 by
1.40 points but loses 1.37 points on GSM8K and is essentially tied on
OlympiadBench and AMC Average@32. Lower SFT validation loss therefore does not
map monotonically to broad exact-answer benchmark accuracy.

## Why The Objectives Diverge

SFT minimizes next-token negative log-likelihood over the complete assistant
trajectory under teacher forcing. External evaluation assigns an item-level
exact-answer score. Several mechanisms separate these objectives:

- long reasoning/style regions contain many more loss-bearing tokens than the
  short final-answer region;
- a model can better imitate wording and local transitions while still making
  one decisive arithmetic or logical error;
- teacher forcing conditions on the correct reference prefix, while free
  generation must recover after its own earlier choices;
- multiple valid solution paths are penalized relative to one reference even
  when they could reach the same correct answer;
- reference trajectories can contain reasoning inconsistency or answer noise;
- token-weighted source composition differs from row-weighted composition,
  especially because olympiad solutions tend to be long.

The most accurate current interpretation is objective misalignment and
task-dependent specialization, not demonstrated classical overfitting. A
stronger overfitting test would evaluate fixed external slices at the saved
392/979/1958/2937/3916 checkpoints. External accuracy peaking and then falling
while SFT validation loss continues to improve would support negative transfer
or benchmark overfitting.

## SFT Versus RLVR

RL with verifiable rewards directly optimizes an outcome signal: any trajectory
that produces a verifier-accepted final answer can receive reward, rather than
being required to imitate one canonical chain of thought. This is better aligned
with exact-answer evaluation, but introduces on-policy sampling cost, reward
variance, KL/length dynamics, and possible reward hacking.

The paper's Qwen3-1.7B Table 2 illustrates the potential gap. Its untuned base,
full-parameter GRPO, and Layer-10 GRPO MATH-500 scores are 57.4%, 64.0%, and
68.6%, respectively. The local SFT baseline and SHS scores are 57.6% and 59.0%.
Cross-paper evaluator differences prevent a controlled causal claim, but the
scale of the paper's RL improvement motivates a matched local RL comparison.

## Paper-Faithful Full Budget

The paper's Qwen3 GRPO recipe uses:

- 50,000 decontaminated NuminaMath problems;
- four epochs;
- train batch size 512 prompts;
- group size four responses per prompt;
- PPO mini-batch 128 and micro-batch eight;
- maximum response length 3,072;
- KL coefficient 0.001 and clip range 0.2.

This corresponds approximately to:

- 200,000 prompt presentations;
- 800,000 generated responses before accounting for failed/retried work;
- 391 global prompt batches (`ceil(200000 / 512)`).

The paper does not publish a Qwen3 learning curve showing the earliest batch at
which benchmark gains become significant. Any early-stop schedule is therefore
a local experimental design, not a paper-reported convergence fact.

## Recommended Budget Ladder

Use the same canonical 50K manifest, deterministic prompt/group schedule, and
matched seeds for the naive Layer-10 control and the selected architecture.
Prefer consuming the shared shuffled stream and stopping early over repeatedly
cycling a very small subset.

| Gate | Global batches | Prompt presentations | Generated responses (G=4) | Purpose |
|---|---:|---:|---:|---|
| Systems smoke | 2 | 1,024 | 4,096 | Plumbing, memory, reward, and checkpoint correctness only |
| Directional pilot | 20 | 10,240 | 40,960 | Detect reward/KL/length trends and gross instability |
| Matched architecture pilot | 50 | 25,600 | 102,400 | First credible curve separation and fixed-panel evaluation |
| One-stream pass | 98 | 50,176 | 200,704 | Approximately one 50K epoch; likely first strong quality gate |
| Half paper budget | 196 | 100,352 | 401,408 | Confirm persistence before full spend |
| Paper-shaped full run | 391 | 200,192 | 800,768 | Final matched comparison |

The counts are rounded to complete 512-prompt batches. The initial expectation,
to be tested rather than assumed, is:

- 1-3 batches can validate systems behavior but say nothing reliable about
  model quality;
- 20-50 batches may reveal a directional reward and benchmark trend;
- around 98 batches/one unique-data pass is a reasonable first gate for a
  visibly meaningful quality change;
- subtle architecture differences may require 196-391 batches and replicated
  seeds.

## Provisional Wall-Time Mapping

The current conservative planning anchor for a four-GPU custom-architecture run
using patched synchronous Hugging Face rollout is approximately 64 hours for
the full 391-batch budget. This is not a measured end-to-end production result;
it is the later planning estimate adopted after observing that the custom model
cannot yet use the optimized vLLM path. Earlier 2.2-7.6 hour four-GPU estimates
assumed much more optimistic continuous-batching behavior and should not budget
the first correctness run.

Linear scaling from the 64-hour anchor gives:

| Gate | Pure proportional time on 4 GPUs | Practical planning range |
|---|---:|---:|
| 20 batches | 3.27 hours | 3.5-4.5 hours |
| 50 batches | 8.18 hours | 8.5-10.5 hours |
| 98 batches | 16.04 hours | 16.5-19 hours |

The practical range adds fixed model initialization, graph warmup/compilation,
checkpoint, reward, and orchestration overhead. It excludes a complete four-
benchmark final evaluation unless explicitly included by the launcher.

On one GPU, ideal replica scaling would multiply the rollout-dominated portion
by approximately four, suggesting roughly 13-18 hours for 20 batches and
32-42 hours for 50 batches. Actual scaling can be worse because actor updates,
weight synchronization, stragglers, and CPU-side reward work do not scale
perfectly.

After vLLM/custom-kernel integration, these estimates must be replaced by a
measured 2-batch and 20-batch end-to-end timing. At that point, extrapolate from
`RUN_SHELL_WALL_SECONDS` and separately report rollout, actor update, reward,
checkpoint, and compile time; do not continue using the 64-hour anchor if the
runtime architecture has changed.

## Data Reduction Guidance

A smaller dataset can reduce cost, but generated-token count and response length
are the dominant budget variables. Repeating 5K examples for four epochs still
creates 20K prompt presentations and can over-specialize to a narrow prompt set.
For a pilot, a deterministic, source-aware 10K-25K prefix/subset with no repeats
is preferable to a tiny subset with many epochs.

The pilot should preserve source proportions or explicitly stratify by the
committed source ledger. Both compared policies must see identical prompt IDs in
identical group order. Because GRPO is on-policy, they normally generate their
own responses; generated trajectories should not be shared across different
policies unless the importance-sampling validity is established.

At batches 20, 50, 98, and optionally 196, record:

- mean reward, zero/one reward fractions, group advantage variance;
- KL to reference, clipping fraction, response length and cap-hit rate;
- rollout and actor tokens per second, GPU-hours, and memory;
- a fixed, preregistered compact evaluation panel;
- checkpoint and RNG/sampler state sufficient for exact resume.

Full four-benchmark evaluation is too expensive to run at every early gate.
Use the compact panel for direction, then run the complete evaluator at the
one-stream gate and final selected budgets. The full 50K x four-epoch run remains
necessary for a paper-shaped final comparison; the cheaper ladder is for
architecture rejection and economic decision-making.

### 2026-07-14 Parallel-Evaluation Supersession

The cost statement above is retained as a historical description of the old
serial HF evaluator. Under the validated six-GPU TP=1 vLLM parallel path, a
complete checkpoint evaluation now takes approximately five to nine minutes
after export. For this topology, full evaluation at steps `30/60/98` is
economically approved and replaces the compact-panel-only policy. New hardware,
backend, or response-length distributions must be remeasured before assuming
the same cost.

The primary aggregate is equal-weight `MathAvg` over GSM8K, MATH-500,
OlympiadBench, and AMC Average@32. The Whole-50K weighted proxy using AMC greedy
is secondary, and AMC greedy alone is diagnostic. With the current 504-prompt
global batch, step 98 is a deterministic matched `near-one-pass` over 49,392
rows, not an exact 50K epoch; the 608 omitted row IDs and source composition
must be recorded.

## TriGLU MATH-500 Evaluator Stall And Recovery

At 2026-07-11 22:12 +08, the TriGLU MATH-500 pass reached 422/500 generated
rows and then stopped making prediction progress. The progress logger continued
to refresh through 22:48, but the prediction and review JSONL files had not
changed since 22:11:47. The GPU remained at 0% utilization while the evaluator
held approximately 12 GiB of memory. Process inspection found all 113 evaluator
threads sleeping in `futex_wait`, including the custom synchronous micro-batcher
path. There was no traceback, OOM, CUDA error, disk pressure, or model-weight
failure before the stall.

The strict conclusion is a generation worker/future synchronization stall: the
custom micro-batcher stopped submitting GPU work while EvalScope kept waiting
for unresolved requests. The available evidence does not isolate the deeper
race to a specific Python or CUDA line, so this incident must not be described
as an architecture failure or a proven framework bug at a particular call.

The stalled evaluator was terminated at 22:49:54 and the same ordered run was
relaunched with the original run ID, seed, batch size, checkpoint roots, and
single-GPU topology. The first recovery launch accidentally omitted the
inherited `NPROC_PER_NODE=1`, immediately failed with `invalid device ordinal`,
and wrote no training or evaluation data. The corrected launch explicitly set
`NPROC_PER_NODE=1`. Completed checkpoints and evaluation receipts caused SHS and
the whole-layer baseline to be skipped, and TriGLU returned to MATH-500 with
active GPU generation by 22:52.

The original 422 prediction/review rows remain intact under the first EvalScope
timestamp directory. EvalScope created a new timestamp directory on relaunch
and reported zero rows cached, so the recovered pass is a clean 500-row rerun,
not an automatic continuation from row 423. Do not splice the two active result
streams. Retain the first partial stream for audit only and use the completed
second pass as the authoritative TriGLU MATH-500 result.

Operational lessons:

- a progress logger that refreshes while both prediction JSONL and GPU work are
  stale is not evidence of forward progress;
- preserve `NPROC_PER_NODE=1` explicitly when reconstructing a one-GPU ordered
  launch rather than relying on inherited shell state;
- EvalScope timestamped output directories do not automatically resume an older
  partial prediction stream in this workflow;
- recovery should preserve the old partial directory, restart cleanly, and use
  only one completed timestamp directory for the final report.

### Cache-Recovery Correction

The initial recovery conclusion above was incomplete: EvalScope 1.8.1 supports
native partial resume through `TaskConfig.use_cache`, but the project wrapper did
not expose it. A clean rerun was started and stopped after 39 rows when this was
recognized. Those 39 rows remain isolated in timestamp directory
`20260711_225142` and are not part of the authoritative result.

The wrapper now exposes `--main-use-cache` and records the selected directory in
`evaluation_manifest.json`. Recovery was relaunched against the original
`20260711_213137` directory. EvalScope reported exactly 422 reused predictions
and 78 remaining MATH-500 samples, confirming an index-matched native resume.
The original cache directory, extended to completion by EvalScope itself, is the
authoritative main-phase result. The earlier instruction to use a separate clean
500-row rerun is superseded by this correction.
