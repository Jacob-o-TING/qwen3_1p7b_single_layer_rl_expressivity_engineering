# Baseline Then FP32 SwiGLU-Only OFT After TriGLU-196 Plan

Date: 2026-07-15

Status: **APPROVED - AUTONOMOUS SUCCESSOR ARMED**

## Objective

Stop the approved TriGLU trajectory after the durable global-step-196
checkpoint and six-GPU evaluation. Defer the proposed TriGLU `197 -> 294`+stage, finish the matched whole-layer baseline to global step 196, and then
stage, finish the matched whole-layer baseline to global step 196, and then
train an OFT precision/expression control from the untuned Qwen3-1.7B-Base to
global step 196.

The experiment tests two distinct claims:

1. compare TriGLU with baseline under the same late cosine schedule before a
   third TriGLU stage can confound the comparison;
2. determine whether another added-expressivity FFN method with an explicitly
   FP32 custom transform can reproduce gains, which would argue against
   attributing TriGLU behavior only to its FP32 custom branch.

## Approved Identity

- Run/wave: `baseline_then_oft_fp32_6x5090_grpo_after_triglu196_20260715_v1`
- Boundary watcher screen: `qwen_reorder_after_triglu196_20260715_v1`
- Successor screen: `qwen_baseline_then_oft_fp32_to196_20260715_v1`
- OFT runtime config:
  `oft_fp32_swigluonly_6x5090_grpo_untunedbase_to196_seed20260707_v1.yaml`
- Controller:
  `run_baseline_then_oft_fp32_after_triglu196_20260715_v1.sh`
- Monitor:
  `monitor_baseline_then_oft_fp32_after_triglu196_20260715_v1.sh`
- Report label: `TriGLU-196 then baseline-196 then FP32 SwiGLU-only OFT-196`

## Autonomous Order

1. Wait for TriGLU step-196 checkpoint and six-GPU evaluation to become
   durable.
2. Stop the old priority controller before any TriGLU step-197 update.
3. Resume baseline from its complete step-128 actor, optimizer, RNG, and data
   state; train and evaluate at steps 158 and 196.
4. Build a separately registered exact-identity OFT model from the untuned
   Qwen3-1.7B-Base.
5. Pass a bounded HF/vLLM greedy-parity, dtype, architecture-selection, and
   semantic-dispatch preflight on one GPU.
6. Train OFT with checkpoints and six-GPU evaluations at steps
   `1/20/98/128/158/196`.
7. Compare normalized dataloader state and deterministic prompt-index ledger
   against baseline at every OFT milestone.

The controller is receipt-driven and idempotent. A durable completed segment
or evaluation is skipped on relaunch; a failed segment leaves its latest
checkpoint intact and writes `WAVE_FAILED`. It never shuts down the host.

## Immutable Macro Contract

Both baseline and OFT use seed `20260707`, shuffle enabled, group size 4,
six-rank train/mini/micro batch contract `504/126/6`, response cap 3072, the
same decontaminated veRL parquet hashes, KL coefficient 0.001, clip range 0.2,
and the same six-GPU parallel vLLM evaluator. The six-rank adjustment is the
owner-approved topology-specific replacement for the earlier paper-shaped
`512/128/8` contract.

Checkpoints use `save_freq=10` plus exact milestone endpoints. Steps
`100/130/160` are redundant regular checkpoints and may be removed only after
the protected `128/158/196` checkpoint and evaluation are durable. Step 98 and
step 196 exports are retained; other merged evaluation exports are disposable
only after their eval receipts are durable. A 100 GiB free-space guard applies
before each expensive phase.

## Exact OFT Architecture Contract

OFT applies only to the Layer-10 SwiGLU `gate_proj`, `up_proj`, and
`down_proj` input rotations. It does not wrap or modify Attention. The
rotation parameter, Cayley solve, and rotation-times-activation operation use
FP32; the transformed activation is cast back to BF16 before the frozen base
projection. The rest of the model remains BF16.

The original Layer-10 SwiGLU gate/up/down weights are frozen. Layer-10
Attention remains architecturally vanilla and fully trainable within the
single-layer policy: q/k/v/o projections plus q_norm and k_norm. Layer-10
input and post-attention RMSNorm are trainable. Every layer outside Layer 10,
the embeddings, and the LM head are frozen.

The production fail-fast audit therefore requires exactly 11 trainable
tensors: six Attention tensors, two RMSNorm tensors, and three FP32 OFT raw
rotation tensors. It rejects any trainable base-SwiGLU tensor, any Attention
OFT tensor, a missing OFT rotation, or a partial custom-model construction.

## Initialization, Reference, And LR

The three zero raw OFT matrices make the initial custom model exactly equal to
the untuned base model. This exact-identity export is the actor initialization
and frozen reference through global step 98. After the complete step-98
checkpoint is merged and hashed, OFT uses its own frozen step-98 reference for
steps 99-196.

Learning rate is constant at `5e-6` through step 128. Steps 129-196 use the
same fixed-horizon cosine schedule as baseline and TriGLU, from `5e-6` toward
the `5e-7` floor. Temporary segment endpoints do not reset the scheduler
horizon.

## Launch Gates

Before the successor may run unattended, all of the following must pass on the
pinned host:

- Python compilation and shell syntax for every new source/controller file;
- focused OFT export, freeze-policy, command-builder, model-hook, and plugin
  tests;
- resolved config audit for `504/126/6`, all immutable seeds/hashes, save/eval
  endpoints, FP32 OFT, and no Attention OFT;
- validation of the TriGLU-196 and baseline-128 six-rank checkpoints;
- proof that no TriGLU checkpoint/update beyond 196 occurred;
- editable package registration of the OFT vLLM plugin;
- exact-identity exporter and bounded live HF/vLLM OFT preflight before the
  first OFT GRPO update.

The last two model-sized gates run autonomously after baseline frees the GPUs.
If either fails, OFT training does not start and the controller preserves the
failure evidence.

## Pending Obligations Carried Forward

Canonical registry:
`docs/experiment_plans/PENDING_2026-07-12_mandatory-experiment-gates.md`

- **PENDING-01 Eval Parity Matrix:** deliberately deferred. This wave uses the
  established all-model vLLM trend evaluator and does not close strict HF/vLLM
  evaluator parity.
- **PENDING-02 Pure-BF16 SHS And TriGLU:** deliberately deferred. The OFT FP32
  control does not implement either required pure-BF16 architecture path.
- **PENDING-03 Registered SHS CausalLM Route:** deliberately deferred because
  SHS is outside this wave.

## Owner Superseding Addendum - 2026-07-16

The original baseline-then-OFT plan above is retained as designed evidence,
but its live execution order is superseded before any OFT update. The immediate
OFT comparison is not sufficiently matched to answer the architecture question:
the OFT control freezes the original Layer-10 SwiGLU projections, whereas the
TriGLU policy jointly trains the original SwiGLU and the added branch. A result
would therefore mix architecture expressivity with a materially different
backbone-SwiGLU adaptation policy.

OFT remains a worthwhile later control, especially with a matched policy such
as adding OFT to the TriGLU backbone SwiGLU or otherwise equalizing which base
SwiGLU degrees of freedom can move. It is deferred, not rejected or deleted.
The OFT architecture, preflight, and orchestration work above remain available
for that separately approved comparison.

The live order is restored to the previously approved TriGLU-priority plan:

1. preserve the validated baseline global-step-190 full-state checkpoint;
2. train TriGLU from global step 196 through a third complete 98-update stage
   to step 294;
3. evaluate at steps 226, 256, and 294 with the existing six-GPU evaluator;
4. resume baseline from step 190, finish to step 196, and run its step-196
   evaluation;
5. do not start OFT until the owner separately reactivates a matched control.

TriGLU retains the fixed-horizon cosine schedule from `5e-7` at step 196 to
`5e-8` at step 294. Its checkpoint cadence remains every 10 updates with exact
milestone endpoints retained; its evaluation cadence remains `226/256/294`.
The per-variant frozen step-98 reference, data order, seed, batch, KL, clip,
response-cap, and evaluator contracts do not change.

### Execution Sequencing Correction

Because baseline was already durably checkpointed at step 190 when this
addendum was activated, the owner immediately refined the order to avoid
leaving six nearly complete updates suspended: finish baseline `190 -> 196`
and its evaluation first, then run TriGLU `196 -> 226 -> 256 -> 294`. This is
only a sequencing change. TriGLU's third-stage LR, checkpoint, evaluation,
reference, seed, data-order, batch, KL, clip, and response-cap contracts remain
exactly as stated above. OFT remains deferred.

### Matched Third-Stage Correction

The phrase "TriGLU third 98, then baseline" means both variants must complete
the matched third 98-update stage. Baseline does not terminate scientifically
at step 196. The corrected order is baseline completion and evaluation at 196,
TriGLU to `226/256/294` with evaluation at each endpoint, and then baseline to
the same `226/256/294` endpoints. Both third stages rebase from their own exact
step-196 full state, use the same frozen per-variant step-98 reference, and
cosine-decay from `5e-7` to `5e-8` over the complete 98 updates. Save cadence,
evaluation cadence, data order, seeds, batch contract, KL, clip, and response
cap are matched.

### Step-196 Reference Reset Amendment - 2026-07-16

The owner subsequently required a fresh per-variant KL anchor for the third
98-update stage. The preceding statement that global steps `197 -> 294` retain
the step-98 reference is superseded: TriGLU and baseline freeze their own
step-196 exports as references for that stage. The historical `99 -> 196`
results still used step 98 and are not relabeled. LR decay, AdamW state, data
order, batch contract, reward, clipping, response cap, checkpoint cadence, and
evaluation cadence are unchanged.
