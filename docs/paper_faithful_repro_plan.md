# Qwen3 Single-Layer RL Paper-Faithful Reproduction Plan

This note packages the side-thread discussion into a concrete plan. It is a
planning artifact only: do not treat it as permission to launch GPU jobs, mutate
main-thread configs, or start a new experiment wave without the project owner's explicit
approval.

## Non-Negotiables

- Do not overclaim exact paper faithfulness where the paper does not expose
  exact code, decontamination thresholds, prompt templates, evaluator harness,
  or dataset revisions.
- Do not build a from-scratch evaluation harness unless an existing framework
  cannot cover the needed benchmark after a minimal adapter.
- Do not burn GPU time while the project owner's GPUs are occupied. CPU-only planning,
  data/eval spec work, and dry-run plumbing are fine.
- Do not commit or upload large data/model artifacts by default.
- Before main-thread implementation, propose the exact file/config/run-name
  changes and wait for approval.

## Paper Facts To Anchor On

Target paper: "Is One Layer Enough? Training A Single Transformer Layer Can
Match Full-Parameter RL Training" ([arXiv 2607.01232](https://arxiv.org/abs/2607.01232)).

For Qwen3 experiments, the paper states:

- Models: Qwen3-1.7B-Base, Qwen3-4B-Base, Qwen3-8B-Base.
- Training framework: veRL + GRPO + AdamW.
- Training data: NuminaMath-CoT, decontaminated then random downsampled to 50K
  problems.
- Qwen3 GRPO settings in arXiv v2 Table 8: `train_batch_size=512`,
  `ppo_mini_batch_size=128`, `ppo_micro_batch_size=8`, `group_size=4`,
  `max_response_length=3072`, `KL=0.001`, `clip=0.2`, `epochs=4`.
- Paper reports mean/std over 3 independent evaluation runs.
- AMC is small, so the paper reports `Average@32`.

The key Qwen3 Math Avg results discussed in the side thread:

| Model | Base | Full RL | Best Single Layer | Single vs Full |
|---|---:|---:|---:|---:|
| Qwen3-1.7B | `44.1` | `50.8` | Layer 10 `51.8` | `+1.0`, contribution `1.14` |
| Qwen3-4B | `52.2` | `63.7` | Layer 16 `64.3` | `+0.6`, contribution `1.06` |
| Qwen3-8B | `58.0` | `66.5` | Layer 16 `67.1` | `+0.6`, contribution `1.07` |

## Current Scaffold Interpretation

The current scaffold is paper-shaped, not yet a strict official reproduction.
It already aligns several major choices:

- `Qwen/Qwen3-1.7B-Base`.
- Layer-10 single-layer training as the first anchor.
- GRPO via veRL as the intended training framework.
- NuminaMath-CoT 50K target.
- `group_size=4`, `max_response_length=3072`, `KL=0.001`, `clip=0.2`.
- `train_batch_size=512`, `ppo_mini_batch_size=128`,
  `ppo_micro_batch_size=8`, `epochs=4`, matching the paper v2 Qwen3 table.
- Selected decoder layer trainable; embeddings and LM head frozen.
- Architecture variants are ordered as SHS, single-layer baseline, TriGLU,
  OFT for the first run, so the full layer sanity check lands immediately after
  SHS.

The known gaps are:

- No official paper repo/eval code has been identified.
- Exact decontamination implementation and thresholds are not public in the
  paper.
- The paper does not name an evaluation harness such as OpenCompass, EvalScope,
  or lm-evaluation-harness.
- Prompt templates, decoding params, answer extraction, verifier details, and
  dataset revision hashes need to be reconstructed and recorded.
- veRL release/patch, model revision, rollout backend, and concurrency knobs
  still need to be pinned for a controlled run.

## Data And Decontamination Plan

The strict-ish ordering should be:

```text
NuminaMath-CoT full train
-> collect/pin eval benchmark question texts
-> decontam
-> seeded sample to 50K
-> save manifest and row IDs
```

Recommended decontamination layers:

- Normalized exact match: lowercase/Unicode/whitespace/LaTeX/punctuation
  normalization, then exact compare against benchmark questions.
- n-gram or MinHash overlap: catches near-duplicates with small rewrites.
- Optional semantic similarity: only after an explicit threshold/model decision,
  because this introduces extra degrees of freedom not specified by the paper.

The output manifest should record:

- NuminaMath-CoT source and revision if available.
- Eval benchmark names, versions, splits, and hashes.
- Normalization function version.
- Exact-match removals.
- n-gram/MinHash threshold and removals.
- Optional semantic model/threshold and removals.
- Sampling seed and final 50K row IDs.

Because the paper does not publish exact decontamination code, the honest label
should be "paper-aligned decontamination", not "bitwise paper-identical".

## Evaluation Strategy

Do not build a full evaluator from scratch. Use an existing evaluation
framework first, then add minimal recipe glue if needed.

Recommended evaluation stack:

- Training-time validation: veRL lightweight validation, mostly for monitoring
  reward/accuracy during RL.
- Final paper-style evaluation: an external evaluator, preferably OpenCompass
  or EvalScope first because coverage for math/Chinese benchmark ecosystems is
  likely stronger than generic lm-evaluation-harness. lm-evaluation-harness can
  remain a fallback or cross-check for GSM8K/MATH-style tasks.

The missing paper details to pin explicitly:

- Framework and commit/version.
- Benchmark versions/splits for MATH500, GSM8K, OlympiadBench, and AMC.
- Prompt template.
- Decoding parameters.
- Number of samples per benchmark, especially AMC `Average@32`.
- Answer extraction and math verifier.
- Metric aggregation into Math Avg.

The project reporting formula is explicit:

```text
MathAvg = (GSM8K + MATH-500 + OlympiadBench + AMC Average@32) / 4
```

AMC greedy pass@1 is a separately reported diagnostic and must not replace AMC
Average@32 inside `MathAvg`. A training-mix weighted proxy may also be reported,
but it must retain a distinct name and remain secondary to the paper-aligned
MathAvg.

A good sanity gate before RL is: evaluate the base Qwen3-1.7B checkpoint and
check whether Math Avg is in the neighborhood of the paper's `44.1`. If base
eval is far away, fix eval recipe before spending RL training compute.

## RL And Eval Relationship

RL and final evaluation should not be treated as one monolithic harness.

- veRL is the RL harness: rollout, reward, GRPO update, checkpointing.
- The benchmark evaluator is the final reporting harness: run checkpoints on
  paper benchmarks and produce the table metrics.

They can be connected by scripts, but keeping them conceptually separate makes
the reproduction cleaner. Training-time validation can be cheap and frequent;
final benchmark eval can be slower, more complete, and run on selected
checkpoints.

## Rollout And Hardware Plan

For a plain Qwen3-1.7B baseline, prefer replica/data-parallel vLLM rollout
rather than tensor-parallel rollout:

```text
tensor_parallel_size = 1
one full model replica per GPU
vLLM continuous batching inside each GPU replica
```

For the SHS/TriGLU/OFT architecture-variant wave, the first correctness path is
patched synchronous HF rollout under veRL `v0.6.1`, because current async
vLLM/SGlang rollout would otherwise execute an unmodified Qwen architecture.
This is slower, but keeps rollout generation and actor updates on the same
model graph.

Before GPU launch, make these explicit in config or veRL mapping:

- `tensor_parallel_size: 1`.
- Rollout replicas equal to available GPUs for 1.7B.
- Initial `max_num_seqs: 16` or `32` per GPU.
- `gpu_memory_utilization`.
- Optional `max_num_batched_tokens`.

## Phased Execution Plan

### Phase 0: Approval And Planning

- Review this MD with the project owner.
- Decide whether the next main-thread task is documentation only, config
  tightening, data pipeline tightening, or evaluator selection.
- Do not launch training.

### Phase 1: CPU-Only Tightening

- Choose primary evaluator: OpenCompass or EvalScope first.
- Pin evaluator version/commit and dataset sources.
- Write decontamination spec and manifest schema.
- Prepare benchmark question hashes.
- Add dry-run eval config with fake predictions if useful.
- Keep outputs small and commit only source/config/docs, not data dumps.

### Phase 2: GPU-Available Paper Baseline

- Regenerate/pin decontaminated NuminaMath-CoT 50K if necessary.
- Run base model eval first; compare to paper Base Math Avg `44.1`.
- If base eval is close enough, run Qwen3-1.7B Layer-10 single-layer RL.
- Evaluate the resulting checkpoint on the same final benchmark recipe.
- Compare against paper: Base `44.1`, Full RL `50.8`, Layer 10 `51.8`.

### Phase 3: Variants Under The Same Recipe

Only after the paper baseline is sane, run variants with the same data,
rollout, evaluation, and logging recipe:

1. SHS.
2. Single-layer baseline.
3. TriGLU.
4. OFT.

The comparison goal is not only "does a variant improve", but whether the
improvement survives a paper-aligned recipe and identical eval path.

## Batch Size And Compute Clarification

Smaller batch size does not automatically save compute.

If the run is fixed by epochs over the same 50K prompts, then rough rollout
sample count is:

```text
dataset_size * epochs * group_size
```

In that case, changing `train_batch_size` mostly changes update count,
per-step memory pressure, and scheduling overhead. It does not necessarily
reduce total generated tokens.

If the run is fixed by number of update steps, then larger batch size does
increase total prompts and rollout samples. The compute budget should therefore
be controlled by total prompts, group size, response lengths, epochs/steps,
and eval frequency, not by batch size alone.

## Approval Gates Before Main-Agent Work

Ask the project owner before:

- changing active training configs;
- regenerating the 50K training dataset;
- downloading/pinning benchmark datasets in bulk;
- selecting final evaluator framework;
- launching GPU jobs;
- changing run names or creating a new experiment wave;
- committing generated data, model checkpoints, or large eval outputs.

## Open Questions

- Which evaluator should be primary: OpenCompass or EvalScope?
- Which exact benchmark source/revision should be used for OlympiadBench and
  AMC?
- Should semantic decontamination be used, or should the first pass stay with
  normalized exact + n-gram/MinHash only?
- What tolerance is acceptable for base-model eval mismatch before RL training
  is considered invalid?
- Should final evaluation save full generations, extracted answers only, or
  both?

## Pending Obligations Carried Forward

Canonical registry:
`docs/experiment_plans/PENDING_2026-07-12_mandatory-experiment-gates.md`

- **PENDING-01 Eval Parity Matrix:** still PENDING. It is in scope before final
  strict paper-comparison claims; the current vLLM trend evaluator does not
  complete it.
- **PENDING-02 Pure-BF16 SHS And TriGLU:** still PENDING and deliberately
  deferred from this paper-faithful baseline plan.
- **PENDING-03 Registered SHS CausalLM Route:** still PENDING and deliberately
  deferred because SHS serving is outside the immediate paper baseline.
