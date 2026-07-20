# TriGLU + Baseline 6x5090 GRPO 20-to-98 Serial Plan

## Approved order

1. TriGLU: untuned base to step 20, then six-GPU parallel evaluation.
2. Baseline: the same untuned base to step 20, then six-GPU parallel evaluation.
3. TriGLU: resume the exact step-20 state to step 98, then parallel evaluation.
4. Baseline: resume the exact step-20 state to step 98, then parallel evaluation.

The controller is autonomous across all eight phases. Client disconnects and heartbeat
failures do not pause it. A step-1 save/restart canary is an internal part of each first
training phase.

## Scientific contract

- All owner-approved macro settings are non-negotiable. Hardware and framework
  constraints must be handled by the implementation and may not silently alter
  batch semantics, exposure, order, seeds, caps, cadence, or evaluation protocol.
- Both variants use the same Qwen3-1.7B-Base files, data parquet files, row ledger,
  initialization seed, rollout seed, shuffle seed, group size, and response cap.
- Owner-approved topology deviation: use all six ranks with `504/126/6` train/mini/micro
  batch sizes. This retains four PPO minibatches per global batch and exposes 49,392
  prompts over 98 updates, 1.56% fewer than paper `512 x 98`. TriGLU and baseline remain
  exactly matched. This wave is architecture-comparative, not an exact paper batch-size
  reproduction; the deviation was explicitly approved after considering the four-GPU
  exact alternative.
- The historical `TriGLU` run label refers specifically to the Layer-10
  **compressed side TriGLU with an embedded GeLU-FFN third factor**, not the
  simpler vanilla side-FFN multiplier. Its side graph is
  `down -> {value, SiLU(gate), SiLU(ffn_2(GeLU(ffn_1)))} -> product -> up`,
  with `side_dim=512`, `side_hidden=2048`, exact-no-op initialization, the
  selected backbone, and all 12 side tensors trainable. Existing run IDs and
  checkpoints remain immutable; reports must use this exact architecture
  identity when interpreting the current wave.
- Checkpoints are saved at step 1, every five steps, and each segment endpoint. All are
  retained for this debugging-oriented wave. A disk guard runs before every phase.
- Automatic continuation is preferred. The controller stops only for non-finite training,
  OOM/process failure, missing checkpoint/resume evidence, broken trainable policy, or
  evaluation engine failure. Low accuracy alone is recorded and does not auto-stop.

## Evaluation protocol

Each checkpoint is merged once, then six independent TP=1 vLLM replicas run concurrently:
MATH-500, GSM8K, OlympiadBench, two AMC sampled shards, and one AMC sampled shard followed
by AMC greedy. The three sampled shards contribute 11+11+10 draws per AMC item. This is a
fast trend protocol; it is not strict-compatible evidence and does not complete PENDING-01.

The reporting hierarchy is:

1. Primary model-quality aggregate: equal-weight `MathAvg` over GSM8K,
   MATH-500, OlympiadBench, and AMC Average@32.
2. Secondary training-distribution aggregate: the committed Whole-50K
   training-mix weighted proxy, which uses AMC greedy pass@1.
3. Diagnostic: AMC greedy pass@1 by itself. It must remain visible but must not
   replace AMC Average@32 inside `MathAvg`.

## Approved post-wave learning-curve and continuation plan

The active controller remains unchanged and must first finish baseline step 98
and its final evaluation. Once the wave is complete, evaluate the existing step
30 and step 60 checkpoints for both TriGLU and baseline. Checkpoint cadence is
every five updates, so these are retrospective evaluations of immutable saved
states, not training reruns. The resulting comparison grid is steps
`20/30/60/98` for both variants.

Every milestone report must include the primary `MathAvg`, all four component
benchmark scores, the secondary weighted proxy, mean reward, PPO KL, clip
fraction, response-length statistics, and cap-hit rate. Six-GPU vLLM parallel
evaluation is now the default economic path for this grid; measured complete
phases are approximately five to nine minutes per checkpoint after export.

With the approved six-rank batch size, each update exposes 504 prompts and four
responses per prompt. The optimizer microbatch size `6` and GPU count do not
multiply dataset exposure. Therefore:

```text
step 30: 15,120 prompts,  60,480 responses, 30.240% of 50K
step 60: 30,240 prompts, 120,960 responses, 60.480% of 50K
step 98: 49,392 prompts, 197,568 responses, 98.784% of 50K
```

The current veRL loader uses a deterministic shuffled permutation with
`drop_last=True`. For a 50,000-row dataset, `98 * 504 = 49,392`, so 608 rows
remain unconsumed at the approved step-98 boundary. For this matched
architecture pilot, accept that deterministic omission only if both variants
use the identical permutation. Preserve the omitted row IDs, their hash, and
their source composition, and label the milestone `98-step near-one-pass`, not
an exact full epoch. Do not invent an irregular partial batch. A separately
validated carry-over sampler is the future exact-50K option; it is not part of
this active wave.

After the `20/30/60/98` curve is complete:

- If `MathAvg` continues to rise and the architecture advantage remains stable,
  resume the same checkpoints for one additional 98-update block. The owner's
  approved relative `+30/+60/+98` labels are recorded authoritatively as
  cumulative global steps `128/158/196`.
- If the curve plateaus, regresses, or the advantage is unstable, do not spend
  the second block on the current compressed TriGLU. Prioritize the pure-BF16
  path, native vLLM trunk, or the separately named vanilla side-FFN variant.
- Every new architecture starts from the same untuned base and uses prospective
  `30/60/98` evaluations under the same data ledger and metric hierarchy.

The second-block execution must be interleaved by matched exposure, not run one
variant all the way to step 196 before the other:

1. TriGLU `98 -> 128`, then six-GPU parallel evaluation.
2. Baseline `98 -> 128`, then six-GPU parallel evaluation and the paired
   step-128 comparison.
3. TriGLU `128 -> 158`, then six-GPU parallel evaluation.
4. Baseline `128 -> 158`, then six-GPU parallel evaluation and the paired
   step-158 comparison.
5. TriGLU `158 -> 196`, then six-GPU parallel evaluation.
6. Baseline `158 -> 196`, then six-GPU parallel evaluation and the paired
   step-196 comparison.

This order limits the exposure lead of either architecture to 30 updates and
lets the owner observe divergence while the wave is running. The human-readable
monitor must show each completed first-member evaluation immediately, including
its change from the same variant's preceding milestone, while labeling the
paired cell `pending`. Once the second member completes, it must emit the
matched same-step delta for MathAvg, all raw benchmarks, the weighted proxy,
reward, KL, response length, and cap-hit rate. Run IDs, checkpoint directories,
and reports use only cumulative labels `128/158/196`; relative labels are
explanatory aliases and are never resume identifiers.

Both step-98 checkpoints must retain their exact sampler and RNG state. At each
paired milestone, fail fast unless consumed-row ledger hashes match between
variants, including the epoch-boundary tail transition after step 98.

## Pending Obligations Carried Forward

- PENDING-01 Eval Parity Matrix: deliberately deferred; the production trend evaluator is
  labeled separately.
- PENDING-02 pure-BF16 SHS and TriGLU: deliberately deferred; this run uses the validated
  historical TriGLU path.
- PENDING-03 registered SHS CausalLM: deliberately deferred; SHS is outside this wave.

## Acceptance and monitoring

The human monitor reports current phase, update progress, latest reward/KL/clip/length,
checkpoint state, partial/final benchmark accuracy, GPU load, elapsed time, and ETA. It
must not dump artifact inventories. The first scheduled heartbeat is chosen from measured
step timing after production launch and performs one bounded check only.

## Follow-up: TriGLU regularization ablation

Owner hypothesis to preserve: a single policy-level KL coefficient may over-constrain the
new TriGLU third branch, whose purpose is to add function space unavailable to the frozen
reference policy. The current GRPO KL is not parameter-wise or module-wise; it measures the
current actor versus frozen reference output distributions on sampled tokens, and its
gradient consequently reaches every trainable component that changes logits, including the
third branch.

Do not alter the active approved wave. After its matched comparison, design a separately
named ablation that distinguishes:

1. the current policy-level KL (`0.001`) applied normally;
2. reduced or zero policy-level KL for the whole actor;
3. branch-specific freedom implemented explicitly, for example a detached auxiliary
   base-path distillation term or different optimizer/regularization treatment for side
   parameters, rather than inaccurately claiming that ordinary KL can be switched off for
   one module;
4. branch learning-rate and direct parameter regularization controls, so KL effects are not
   confounded with optimization scale.

For comparison, conventional LoRA SFT normally optimizes supervised token cross-entropy and
does not add actor-versus-reference policy KL. LoRA-based RLHF/GRPO commonly retains an
output-distribution KL even though only adapter parameters are trainable.

## Follow-up: topology correction and runtime optimization

The owner originally had the distinct FineWeb ablation
`SwiGLU + vanilla GeLU-FFN multiplier side branch` in mind. That topology is
`down 2048->512 -> ffn_1 512->2048 -> ffn_2 2048->512 -> up 512->6144` and
must be implemented, named, trained, and evaluated as a separate future
variant. It must never overwrite or be conflated with this wave's compressed
side TriGLU checkpoints.

The measured production timing and the complete shared optimization ladder are
recorded in the `2026-07-13 TriGLU Topology Correction And Shared Runtime
Optimization Plan` addendum of
`2026-07-11_four-variant-selection-naive-rl-and-kernel-plan.md`. None of those
future changes alters the active serial controller or its approved macro
contract.
