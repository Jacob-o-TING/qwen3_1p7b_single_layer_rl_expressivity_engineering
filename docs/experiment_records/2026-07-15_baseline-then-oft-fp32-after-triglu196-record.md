# Baseline Then FP32 SwiGLU-Only OFT After TriGLU-196 Record

Date: 2026-07-15

Status: **AUTONOMOUS SUCCESSOR ARMED - TRAINING IN PROGRESS**

## Owner Decision

The owner changed the approved priority order before the TriGLU third
98-update stage began. TriGLU must stop after its global-step-196 evaluation;
baseline must finish to 196 next; then a new OFT control must run from the
untuned base to 196. This ordering tests the architecture comparison before a
third TriGLU schedule and tests whether added FFN expressivity under an FP32
custom transform, rather than TriGLU alone, explains the observed behavior.

The OFT contract was explicitly confirmed as follows: Attention receives no
OFT and remains normal full training within Layer 10; Layer-10 RMSNorm remains
trainable; the original SwiGLU matrices are frozen; only three OFT rotations
train on the SwiGLU side. The expected total is exactly 11 tensors.

## Boundary Protection

The watcher `qwen_reorder_after_triglu196_20260715_v1` is active remotely. It
waits for the durable TriGLU-196 parallel-eval receipt, validates the six-rank
checkpoint, stops the old priority controller, rejects any observed update or
checkpoint beyond 196, and then starts the successor controller. If the
successor is not yet executable, it waits safely rather than allowing the old
TriGLU third stage to begin.

Initial watcher state:

```text
HANDOFF_PHASE=WAITING_FOR_TRIGLU_196_EVAL
```

This is boundary protection, not yet evidence that the entire successor wave
is production-ready.

## Implementation In Progress

The scoped implementation adds:

- a separately registered `Qwen3OFTForCausalLM` HF/vLLM route;
- FP32 OFT parameter, Cayley solve, and transform application with BF16 output
  restoration before frozen base projections;
- an exact-identity untuned-base export utility;
- a bounded live HF/vLLM parity and dispatch preflight;
- an idempotent baseline-then-OFT controller with six-GPU eval, data-order
  receipts, reference freezing, retention, disk guard, and fail-fast trainable
  audit;
- an owner-facing human-readable monitor.

## Remote Verification And Arming

The scoped initial deployment archive SHA256 was
`f306f80ac3ba124f37fbbc0571a9a252242c9b71d3552e8c90334f7cea9bdc90`.
Two focused lifecycle patches were deployed under SHA256
`1645c7f7d0894a14e59b587c33cc742fb89a3ceaaeff656b4182d9ca761ac2ff`
and
`a3efe302b9ef9f3566c27a6c96b7ad0d8ffa3f0ce6a725371ab7c1af52c6a18c`.
The second patch replaced an invalid initialization-time
`_keep_in_fp32_modules` strategy with post-`from_pretrained` precision
finalization. No live training process used the failed source-only attempt.

Pinned-host verification passed:

- all three new shell entry points passed `bash -n`;
- all new Python files passed `py_compile`;
- the editable package installation exposed the
  `qwen_single_layer_rl_oft` vLLM plugin entry point;
- focused tests passed `27/27`, including tiny-model export/reload key parity,
  BF16 model load with three OFT rotations restored to FP32, exact 11-tensor
  partition, command ordering, and prior freeze/model-hook coverage;
- the resolved production config and generated veRL command preserved
  `504/126/6`, group 4, six GPUs, all three seed fields at `20260707`, the
  exact target modules, and the approved milestone/LR contract;
- both the original TriGLU priority screen and the boundary watcher remained
  alive after deployment. The bounded live check observed TriGLU at step 164,
  six GPUs at 100% utilization, and no interruption.

The successor controller is now executable at the path watched by the already
active boundary guard. The durable handoff state remains:

```text
HANDOFF_PHASE=WAITING_FOR_TRIGLU_196_EVAL
```

At the boundary, the guard will launch
`qwen_baseline_then_oft_fp32_to196_20260715_v1` automatically. The model-sized
exact-identity export and live HF/vLLM preflight intentionally remain inside
the autonomous controller after baseline releases the GPUs; OFT update 1 is
gated on both passing.

Owner-facing status command:

```bash
cd /root/autodl-tmp/qwen3_1p7b_single_layer_rl
bash scripts/monitor_baseline_then_oft_fp32_after_triglu196_20260715_v1.sh
```

## Handoff Root Repair

At `2026-07-15T22:12:04+08:00`, the boundary watcher correctly observed the
durable TriGLU-196 evaluation but looked for the corresponding checkpoint
under the priority wave root. The evaluation belongs to that root, while the
completed `<=196` training checkpoint intentionally remains under the earlier
interleaved wave root. The watcher therefore failed closed with
`triglu_step196_checkpoint_invalid`; it did not start baseline or mutate a
checkpoint. The old priority controller subsequently exited during its
superseded third-stage attempt with no durable step beyond 196.

The repair separates `SOURCE_ROOT` for the evaluation/controller from
`CHECKPOINT_ROOT` for the six-rank TriGLU-196 state. A focused regression test
pins both paths. Recovery validates the existing TriGLU-196 and baseline-128
checkpoints, confirms the third-stage tracker did not exceed 196, and launches
the same approved successor without rerunning any completed training or eval.

## Pending Obligations Carried Forward

Canonical registry:
`docs/experiment_plans/PENDING_2026-07-12_mandatory-experiment-gates.md`

- **PENDING-01 Eval Parity Matrix:** PENDING, deferred.
- **PENDING-02 Pure-BF16 SHS And TriGLU:** PENDING, deferred.
- **PENDING-03 Registered SHS CausalLM Route:** PENDING, deferred.

## Live Order Superseded Before OFT - 2026-07-16

The owner identified a comparison-policy confound before OFT training began.
The planned OFT control freezes the original Layer-10 SwiGLU projections while
TriGLU trains its original SwiGLU jointly with the side branch. Treating the
result as a direct architecture comparison would therefore conflate added
expressivity with different backbone-SwiGLU adaptation constraints.

At the decision boundary, the baseline had a validated global-step-190
checkpoint with all six actor, optimizer, and extra-state shards. The baseline
trainer was interrupted after that durable boundary; no completed state was
lost. The OFT tracker remained at zero and no OFT model update had run.

The controller was replaced by the already verified TriGLU-priority
continuation. The new durable order is TriGLU `196 -> 226 -> 256 -> 294`, with
six-GPU evaluation at every listed endpoint, followed by baseline `190 -> 196`
and its step-196 evaluation. TriGLU keeps the owner-approved cosine decay from
`5e-7` to `5e-8` across the full 98-update stage and the existing save/eval
cadence. OFT is explicitly deferred for a later matched-policy comparison; the
original plan and implementation record above remain intact rather than being
rewritten as if they had never existed.

The first restored-controller launch then failed before rollout or update
because the LR-stage transition helper wrote `global_step_196` into veRL's
`latest_checkpointed_iteration.txt`. Pinned veRL 0.6.1 parses that file with
`int(...)` and requires the canonical payload `196`. The failure left the
source TriGLU-196 checkpoint and baseline-190 checkpoint unchanged and produced
no step beyond 196. The helper now writes the canonical integer form, a focused
regression test pins that contract, and relaunch recreates the disposable
transition copy from the validated source checkpoint.

Before the repaired TriGLU continuation produced step 197, the owner refined
the order once more because baseline was only six updates from completion.
TriGLU was interrupted with its canonical tracker still at `196`; baseline is
therefore completed and evaluated at 196 first, followed by the full TriGLU
third stage. Baseline resumes from its validated step-190 checkpoint. TriGLU
later resumes from its validated step-196 transition checkpoint.

Both resumes preserve all six AdamW optimizer shards byte-for-byte, including
per-parameter `exp_avg`, `exp_avg_sq`, and step counters. Each checkpoint also
retains all six model and extra-state shards plus `data.pt` for RNG/dataloader
continuity. The TriGLU stage transition rewrites only scheduler metadata in
private copies of the six extra-state shards; model, AdamW optimizer, and data
state content are not rewritten. No partially executed update is accepted as a
checkpoint.

The first baseline-first relaunch also exposed a cross-root idempotency gap:
the priority controller looked only in its own evaluation root and began to
stage a redundant baseline-158 evaluation even though the authoritative
predecessor reordered-wave receipt was already complete. It was stopped before becoming
the active continuation. The evaluator now checks the archived source receipt
first and skips completed historical milestones, so recovery proceeds directly
from baseline checkpoint 190 toward the still-missing step-196 cell.

Final recovery verification passed after the cross-root repair: focused tests
passed `18/18`; the detached controller reported `TRAIN baseline target=196`;
veRL logged an exact resume from baseline `global_step_190`; the durable tracker
remained 190 before the next completed update; and all six RTX 5090s were active
at approximately 81-83% utilization with about 16.9 GiB allocated per GPU. The
controller will evaluate baseline-196 before creating the TriGLU third-stage
transition copy.

The owner then corrected an inherited old-controller assumption: baseline was
displayed and scheduled with a final step of 196 while TriGLU continued to 294.
That asymmetry belonged to the superseded "TriGLU priority, baseline catch-up"
plan and is not the requested matched third-stage comparison. The durable
contract is now both variants to 294, in the order baseline-196 completion,
TriGLU third 98, then baseline third 98. A handoff watcher preserves the active
baseline computation and promotes the extended controller only after the
currently loaded controller exits cleanly.

The human-readable monitor is restored as a union of its useful historical
features and later metrics: current phase, per-variant progress to 294, latest
checkpoint, archived-metric fallback, reward/KL/clip, LR schedule, response
length/cap rate, rollout/update/step timing, ETA, partial and final benchmark
cells, AMC greedy and Average@32 separately, whole-50K weighted score, MathAvg,
data-order receipts, alerts, GPU utilization, and free disk.

Remote deployment preserved the active controller inode by staging the extended
controller as `.next`. The existing approved handoff screen now reports
`WAITING_FOR_BOTH_TO_294_EXTENSION`; it will promote that staged controller only
after the current screen completes cleanly. Focused remote verification passed
`9/9` extension tests and `11/11` monitor/OFT compatibility tests. A live monitor
smoke showed baseline step 191, checkpoint 190, current reward/KL/clip, LR,
response-length/cap-hit, step/rollout/actor timing, throughput, and ETA, followed
by the complete milestone comparison table. The active training process was not
interrupted by this deployment.

## Live Scientific-Contract Correction - 2026-07-16

While baseline was live at step 194 of 196, an audit found that the staged
third-stage configs and already loaded controller still routed KL to each
variant's step-98 export. The owner clarified that the third 98 updates must
reset the reference to each variant's own step-196 policy.

Before any third-stage update, the TriGLU step-98 export completion marker was
renamed to `EXPORT_COMPLETE.blocked_before_third_stage_step196_reference`.
This left its weights intact and did not interrupt baseline training, but makes
the old in-memory controller fail closed before update 197. The durable receipt
is `receipts/third_stage_reference_fail_closed.txt`. The corrected controller
uses step 98 only through update 196, creates SHA256-audited frozen step-196
exports for both variants, and routes updates 197-294 exclusively to those
references. Promotion requires both step-196 checkpoints/evaluations and proof
that no checkpoint above 196 exists.

The corrected controller, watcher, and monitor passed remote `bash -n`; the
focused step-294 continuation suite passed `10/10` in the pinned environment.
The corrected controller was staged as `.next` and atomically installed at the
on-disk controller path without interrupting the already parsed live Bash
process. The replacement handoff watcher is detached and reports
`WAITING_FOR_BOTH_TO_294_EXTENSION`. At arming time, baseline remained live at
target 196 and all six GPUs were active; no third-stage update had started.
