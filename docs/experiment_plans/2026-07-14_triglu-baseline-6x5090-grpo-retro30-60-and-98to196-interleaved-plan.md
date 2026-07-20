# TriGLU + Baseline 6x5090 GRPO Retro-30/60 And 98-to-196 Interleaved Plan

Date: 2026-07-14

Owner approval: explicit.

Wave identity:
`triglu_baseline_6x5090_grpo_retro30_60_then_98to196_interleaved_20260714_v1`

## Objective

After the active 20-to-98 wave finishes successfully, autonomously recover the
missing step-30 and step-60 evaluation cells and then continue both exact
step-98 trainer states through a second 98-update block. Keep architecture
exposure paired closely enough that the owner can observe matched progress in
real time.

The watcher and continuation controller run remotely under `screen`; client
disconnects, local sleep, and absent Codex heartbeats do not block progress.

## Immutable Scientific Contract

- Variants: historical compressed TriGLU and Layer-10 whole-layer baseline.
- Resume source: each variant's own `global_step_98` full veRL checkpoint from
  `triglu_baseline_6x5090_grpo_20to98_serial_20260712_v1`.
- Reference reset: each variant's own merged step-98 policy is frozen as its
  reference model for the continuation. TriGLU is never regularized toward the
  baseline policy, and neither variant remains anchored to the untuned base for
  its already-completed `0 -> 98` drift.
- Resume payload: actor, optimizer, RNG/extra state, and StatefulDataLoader
  `data.pt`; weights-only continuation is forbidden.
- Data: identical decontaminated veRL parquet hashes and committed selection
  ledger.
- Seeds: experiment/init/rollout/dataloader seed `20260707`.
- Shuffle: enabled through veRL's explicitly seeded `RandomSampler`.
- Batch contract: `504/126/6`, group size `4`, six ranks, response cap `3072`.
- Optimization amendment approved on 2026-07-14: keep the paper-aligned
  constant `5e-6` learning rate through cumulative step 128, then apply the
  same cosine decay to both variants over steps 129-196, ending at `5e-7`
  (`min_lr_ratio=0.1`). KL, clipping, reward, and trainable-parameter policies
  remain unchanged.
- Checkpoint cadence: every ten updates plus mandatory segment endpoints
  `128/158/196`.
- Evaluation: six independent TP=1 vLLM replicas using the unchanged trend
  protocol.

No active first-wave source, evaluation, or controller file may be modified or
deleted. The only checkpoint exception is the explicit owner-approved
five-step-to-ten-step retention pruning recorded under Storage And Retention;
step 98 and all retained recovery points remain protected.

## Automatic Start Gate

The watcher waits for the old wave's durable `WAVE_COMPLETE` marker. Before
launching the continuation controller it requires:

1. old state `PHASE=COMPLETE`, `VARIANT=all`, `TARGET=98`;
2. both step-98 checkpoint trees with six model, optimizer, and extra-state
   shards, `fsdp_config.json`, and `data.pt`;
3. complete step-98 parallel-evaluation markers for both variants;
4. no live old controller screen and no GPU compute process;
5. at least 100 GiB free after the old wave closes;
6. identical normalized dataloader cursor state between variants, excluding
   only worker `_base_seed`, which does not control the explicitly seeded
   RandomSampler order for this deterministic dataset.

Any failed gate writes a durable failure receipt and does not launch training.
Low benchmark accuracy by itself is not an automatic stop condition.

## Exact Autonomous Order

1. Evaluate old TriGLU step 30.
2. Evaluate old baseline step 30 and emit the paired step-30 comparison.
3. Evaluate old TriGLU step 60.
4. Evaluate old baseline step 60 and emit the paired step-60 comparison.
5. Resume TriGLU `98 -> 128`, then evaluate step 128.
6. Resume baseline `98 -> 128`, evaluate, and emit the paired step-128 result.
7. Resume TriGLU `128 -> 158`, then evaluate step 158.
8. Resume baseline `128 -> 158`, evaluate, and emit the paired step-158 result.
9. Resume TriGLU `158 -> 196`, then evaluate step 196.
10. Resume baseline `158 -> 196`, evaluate, and emit the paired step-196 result.

Durable checkpoint, directory, and report labels use cumulative global steps
`128/158/196`. Relative `+30/+60/+98` labels are explanatory aliases only.

The optimizer schedule has a fixed global horizon of 196 even when the trainer
temporarily stops at interleaved milestone 158. The project-owned veRL patch
adds this schedule-horizon override so resuming at 158 cannot restart, compress,
or rebound the cosine curve. The decay activates only when the run's durable
latest-checkpoint tracker is at least 128; therefore both `98 -> 128` segments
remain constant-LR runs.

Step-128 evaluation is the first decision point for this amendment. If it is
materially poor, preserve the current trajectory and investigate a separately
named rewind branch from the durable checkpoint nearest the second observed
reward divergence (approximately global step 90), with an earlier/lower LR
schedule. This is an analysis option, not an automatic rollback: it requires
owner approval of the exact source checkpoint, schedule, and run name.

## Approved Owner-Priority Amendment (2026-07-15; Remote Preflight Passed)

The owner prioritizes measuring the TriGLU upper-bound trajectory before
spending additional compute on baseline catch-up. PPO KL and clipping are
optimization constraints or diagnostics; they do not identify the custom
architecture's inductive bias. Consequently, low KL or clip fraction alone is
not evidence against an architecture-specific lower-LR continuation.

The requested logical order is:

1. finish the already-running baseline `99 -> 128` segment and its step-128
   evaluation;
2. pause baseline continuation at global step 128;
3. run TriGLU `129 -> 158`, evaluate, then `159 -> 196`, and evaluate, using
   the already approved cosine schedule from `5e-6` to `5e-7`;
4. continue TriGLU through a third 98-update stage, global steps `197 -> 294`,
   with a new cosine schedule from `5e-7` to `5e-8`;
5. evaluate the third stage at proposed global milestones `226/256/294`, which
   correspond to relative `+30/+60/+98` within that stage;
6. only after TriGLU step 294 is durable, return to baseline at step 128 and
   catch it up through the existing `158/196` milestones and evaluations.

The existing frozen per-variant step-98 reference remains the conservative
default through TriGLU step 294 because no later reference reset has been
approved. A step-196 reference reset would change the RL objective and remains
a separate owner decision. Baseline's delayed `129 -> 196` continuation must
resume its own full step-128 optimizer, RNG, and dataloader state so its prompt
order remains paired with TriGLU for shared global steps `129 -> 196`.

The owner explicitly approved the following successor naming and the default
step-98 reference plus `226/256/294` evaluation contract:

- run ID: `triglu_priority_6x5090_grpo_step128_to294_then_baseline128_to196_20260715_v1`;
- screen: `qwen_triglu_priority_to294_then_baseline196_20260715_v1`;
- third-stage config: `triglu_6x5090_grpo_resume196_to294_cosine5e7_to5e8_seed20260707_v1.yaml`;
- controller: `run_triglu_priority_to294_then_baseline196_20260715_v1.sh`;
- monitor: `monitor_triglu_priority_to294_then_baseline196_20260715_v1.sh`;
- report label: `TriGLU-priority 128-to-294 then baseline catch-up to 196`.

The active controller process is not hot-modified. The approved implementation
uses a successor handoff at a durable boundary;
it must never terminate the active baseline `99 -> 128` worker or mutate its
checkpoint. The current controller would otherwise retain its old interleaved
order, so the successor must be approved and armed before that order reaches
the undesired baseline `129 -> 158` segment. The handoff watcher may be armed
only after focused local and pinned-remote preflight pass.

At the step-196 stage boundary, the third-stage checkpoint is created as a
hardlink tree so model, optimizer, RNG, and dataloader payloads remain the exact
full-state continuation without doubling storage. The six small scheduler
extra-state files are privately copied and rebased from `base_lrs=5e-6` to
`base_lrs=5e-7`, with `last_epoch=196`; the immutable source checkpoint is not
modified. This is required so the third cosine actually reaches `5e-8` rather
than loading the prior scheduler base and ending at `5e-7` again.

The third TriGLU stage mirrors the second 98-update stage's signal cadence:
`save_freq=10` plus mandatory checkpoint-and-evaluation endpoints at
`226/256/294`. Its exact retained steps are
`210,220,226,240,250,256,270,280,290,294`. Regular checkpoints `200/230/260`
are transient because they are adjacent to source or endpoint checkpoints;
each may be removed only after the corresponding later endpoint and evaluation
are validated and durable.

Implementation preflight passed on the pinned six-GPU host before watcher
arming: scoped archive SHA256
`f02395a8eb027a80b799b2cbae2273155af7dc52b4e6dc70e0d47c1ae4020711`,
three shell syntax checks, Python compilation, `22/22` focused tests, and a
synthetic six-rank scheduler-transition checkpoint smoke all passed. The
synthetic smoke proved hardlinked model payloads, private scheduler extra-state
copies, unchanged source scheduler state, rebased target `base_lrs=5e-7`, and
an exact step-196 tracker. This satisfies the preregistered watcher launch gate.

The bounded handoff watcher was then armed at `2026-07-15T05:11:01+08:00`
under screen `qwen_triglu_priority_handoff_20260715_v1`. Its first durable
state is `HANDOFF_PHASE=WAITING_FOR_TRIGLU_158`. The original screen
`qwen_grpo_98to196_interleaved_20260714_v1` and its baseline worker remained
alive and unchanged; no successor training screen exists before the validated
TriGLU step-158 handoff boundary. The owner-facing monitor entry point is:

```bash
cd /root/autodl-tmp/qwen3_1p7b_single_layer_rl
bash scripts/monitor_triglu_priority_to294_then_baseline196_20260715_v1.sh
```

## Metrics And Live Reporting

The primary aggregate is equal-weight `MathAvg` over GSM8K, MATH-500,
OlympiadBench, and AMC Average@32. The Whole-50K weighted proxy using AMC
greedy is secondary; AMC greedy pass@1 is a separate diagnostic.

The human-readable monitor reports watcher/controller status, current variant
and milestone, latest reward/KL/clip/length/cap metrics, recent speed, segment
ETA, every partial or complete benchmark result, MathAvg, weighted proxy, and
paired same-step deltas. When the first variant at a milestone finishes, its
self-trend is visible immediately and the paired cell is labeled pending.

## Resume And Data-Order Evidence

The first continuation segment uses veRL `resume_mode=resume_path` pointing to
the immutable old step-98 checkpoint. Later segments use `resume_mode=auto`
inside the new run identity.

Raw `data.pt` SHA equality is not a valid cross-process gate because worker
`_base_seed` differs even when the explicit sampler seed and row order are
identical. The controller instead records raw hashes, compares normalized
dataloader state with `_base_seed` removed, checks step and samples-yielded
cursors, pins the explicit sampler seed, and emits deterministic expected
prompt-index ledger hashes at each paired milestone.

Before any continuation update, the controller exports each variant's own
step-98 actor into a durable HF reference directory, records per-file SHA256,
and passes it only through `actor_rollout_ref.ref.model.path`. Actor and rollout
initialization paths remain unchanged, while the full actor/optimizer/data
state still resumes from the corresponding veRL checkpoint. Consequently the
initial continuation KL is zero per variant and `beta=0.001` penalizes only
additional `98 -> 196` behavioral drift.

## Storage And Retention

The old wave currently occupies about 320 GiB, and one checkpoint is about
8.2 GiB. The owner explicitly changed the continuation cadence from five to
ten updates after this storage audit. The controller uses veRL
`save_freq=10`, requires durable endpoint checkpoints at `128/158/196`, and
deduplicates the regular checkpoints nearest the existing source or a durable
endpoint.

The exact retained continuation steps per variant are
`110,120,128,140,150,158,170,180,190,196`. The transient regular checkpoints
`100,130,160` are removed only after the corresponding protected endpoint
`128,158,196` has both a validated checkpoint and durable evaluation receipt.
The immutable old step-98 source is never touched. This retains ten new
checkpoints per variant, approximately 164 GiB at the observed size, while the
100 GiB free-space guard remains active. Disposable merged HF exports for
non-final evaluations are likewise removed only after durable evaluation;
final step-196 exports are retained.

The owner subsequently applied the same ten-update principle retroactively to
the completed first `0 -> 98` stage. For both TriGLU and baseline, the retained
old-wave steps are exactly `1,10,20,30,40,50,60,70,80,90,98`; steps
`5,15,25,35,45,55,65,75,85,95` were approved for deletion. Step 1 remains the
canary, step 90 preserves the requested rewind neighborhood, and step 98
remains the immutable continuation source.

## Failure And Recovery

The controller is idempotent. Completed training, exports, and evaluations are
skipped by durable receipts. A process failure writes `WAVE_FAILED` and leaves
the latest retained ten-step or endpoint checkpoint available. Relaunch resumes
from the latest new-wave checkpoint, or from immutable old step 98 if no new
checkpoint exists.

Training stops for non-finite metrics, OOM/process failure, checkpoint or
resume failure, normalized data-order mismatch, missing eval rows, evaluator
failure, or disk guard failure. It does not shut down the instance.

## Pending Obligations Carried Forward

Canonical registry:
`docs/experiment_plans/PENDING_2026-07-12_mandatory-experiment-gates.md`

- **PENDING-01 Eval Parity Matrix:** still PENDING and deliberately deferred.
  This wave uses the labeled vLLM trend evaluator and does not close strict
  backend parity.
- **PENDING-02 Pure-BF16 SHS And TriGLU:** still PENDING and deliberately
  deferred. The historical mixed-precision TriGLU implementation is preserved.
- **PENDING-03 Registered SHS CausalLM Route:** still PENDING and deliberately
  deferred because SHS is outside this wave.

## 2026-07-15 Superseding Order Amendment

The owner superseded the proposed TriGLU `197 -> 294` execution order before
that stage began. The durable TriGLU step-196 result is retained, but the third
TriGLU stage is deferred. Baseline now finishes to 196 first, followed by the
separately named FP32 SwiGLU-only OFT control. The authoritative successor
plan is
`docs/experiment_plans/2026-07-15_baseline-then-oft-fp32-after-triglu196-plan.md`.

## Owner-Approved Step-196 KL Reference Reset - 2026-07-16

The owner explicitly superseded only the third-stage reference-policy text.
The completed `99 -> 196` stage remains historically unchanged and used each
variant's frozen step-98 reference. Global updates `197 -> 294` instead use
each variant's own frozen step-196 policy as the KL reference, so that stage
begins with zero per-variant policy/reference drift while retaining
`beta=0.001`.

The routing is target-aware: targets at or below 196 use the immutable step-98
exports, while targets above 196 require SHA256-audited step-196 exports and a
`FROZEN_OWN_STEP196_REFERENCE_READY` receipt. The recovery watcher may promote
the corrected controller only after both step-196 checkpoints/evaluations are
durable and it proves that no checkpoint above 196 was produced under the old
reference contract.
