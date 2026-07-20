# TriGLU + Baseline 6x5090 GRPO Retro-30/60 And 98-to-196 Interleaved Record

Date: 2026-07-14

Status: WATCHER ARMED; WAITING FOR THE ACTIVE 20-TO-98 WAVE

Approved wave:
`triglu_baseline_6x5090_grpo_retro30_60_then_98to196_interleaved_20260714_v1`

The owner approved an autonomous remote watcher that starts only after the
active 20-to-98 controller and both step-98 evaluations complete successfully.
The continuation order is paired at cumulative steps `128/158/196`, with
retrospective step-30 and step-60 evaluations first.

Initial audit findings:

- active old wave was healthy at baseline step 81/98;
- old wave occupied approximately 320 GiB with 379 GiB free at that snapshot;
- a TriGLU checkpoint occupied approximately 8.2 GiB;
- veRL supports explicit full-state `resume_path` and restores `data.pt`;
- old TriGLU and baseline step-20 `data.pt` files have different raw SHA values
  because worker `_base_seed` differs, while normalized cursor state matches;
- veRL constructs the actual row sampler with an explicitly seeded
  `torch.Generator` and `RandomSampler`, so worker base seed is not the row-order
  authority for this deterministic dataset.

Owner storage decision: continuation checkpoint cadence is `save_freq=10`,
with mandatory endpoint saves at `128/158/196`. To avoid redundant nearby
saves, the exact retained new steps are
`110,120,128,140,150,158,170,180,190,196`; transient steps `100,130,160` are
removed only after their protected endpoint checkpoint and evaluation are
durable. At this point old-wave checkpoints remained untouched; the later
owner-approved first-wave pruning receipt below supersedes that retention
decision while preserving step 98.
Disposable non-final merged HF exports are pruned after durable evaluation;
the final step-196 exports are retained. Final launch, watcher receipt, remote
syntax tests, and initial waiting state will be appended after verification.

## Local Verification

- focused continuation/controller tests: `7/7 PASS`;
- existing veRL command compatibility tests: `3/3 PASS`;
- changed Python modules compile successfully;
- `git diff --check`: clean;
- exact retention-policy coverage protects the old root and permits only
  new-wave steps `100/130/160` to be removed after durable endpoint evidence.

## Remote Preflight And Arming

The scoped deployment archive SHA256 was
`54420081f96002e6d184c3963e95c90139b21edc25397274a5f6d1492ba18029`.
It contained exactly nine files plus tar directory entries and was overlaid on
the deploy snapshot without modifying the active old controller or its config.

- all three new shell scripts passed remote `bash -n`;
- focused continuation/controller tests: `7/7 PASS` remotely;
- the relevant existing registry/veRL command tests passed remotely;
- the full registry file had one unrelated environment failure in its parquet
  materializer test because the image's installed pandas cannot import against
  its current NumPy build; no continuation or active-data code uses that test
  path;
- watcher screen `qwen_grpo_98to196_autostart_20260714_v1` started at
  `2026-07-14T16:39:16+08:00`;
- its first durable state was `AUTOSTART_PHASE=WAITING_OLD_WAVE` with the old
  baseline's latest checkpoint at step 80;
- the old screen remained independently healthy and in `TRAIN baseline -> 98`.

The watcher never shuts down the host. It waits for old `WAVE_COMPLETE`, checks
both step-98 checkpoints and evaluations, waits for the old screen and GPU
processes to drain, enforces the 100 GiB disk guard, and only then launches
controller screen `qwen_grpo_98to196_interleaved_20260714_v1`.

## Owner-Approved Step-98 Reference Reset

Before the watcher launched the continuation, the owner changed the reference
contract: TriGLU and baseline must each freeze their own step-98 policy as the
reference model. The actor still resumes its own full step-98 veRL state,
including optimizer and data cursor. The separate reference path resets each
variant's continuation KL to zero without cross-regularizing TriGLU toward the
baseline or changing `beta=0.001`. This is intentionally a two-stage RL
objective rather than an uninterrupted fixed-base-reference run.

Implementation commit `bf9a194` added an explicit, separately frozen
`actor_rollout_ref.ref.model.path` while preserving the actor/rollout model
path and full-state resume. Local focused tests passed `8/8`, existing veRL
command compatibility tests passed `3/3`, and the remote focused suite passed
`8/8`. Remote `bash -n` passed, and pinned veRL Hydra successfully resolved a
distinct reference-model override. After deployment the watcher remained in
`WAITING_OLD_WAVE` at old baseline checkpoint 85; no continuation process had
started and the active old controller was not restarted.

## Owner-Approved Learning-Rate Amendment

On 2026-07-14, while TriGLU was still in the constant-LR `98 -> 128` segment,
the owner approved a matched continuation schedule for both variants. The
paper-aligned `5e-6` rate remains unchanged through global step 128. Starting
after that checkpoint, global steps 129-196 use cosine decay with a `5e-7`
floor (`min_lr_ratio=0.1`). The intent is to protect a potentially delicate
custom architecture while preserving a paired baseline control.

This amendment intentionally departs from strict paper reproduction. It does
not retroactively change completed updates, the frozen step-98 references,
data order, KL, clipping, or checkpoint/evaluation milestones. Activation is
checkpoint-gated, and the optimizer schedule horizon is fixed at global step
196 independently of the trainer's temporary interleaved stop at step 158.

### Remote activation evidence

The fixed-horizon veRL patch and continuation command path were deployed while
the live TriGLU `98 -> 128` segment remained untouched. Remote verification on
2026-07-14 produced:

- focused tests: `14/14 PASS`, plus monitor shell syntax PASS;
- fixed optimizer horizon: global step `196`, independent of temporary trainer
  stops at `128` and `158`;
- CPU state-handoff simulation: step 129 `5.000000e-6`, step 158
  `3.265139e-6`, step 196 `5.024008e-7`, post-step-196 `5.000000e-7`;
- monotonic decay PASS with no LR discontinuity at the constant-to-cosine
  boundary;
- patched veRL trainer SHA256
  `940fbfbdea06602e87f8226eb678c910b9509210b9f8c0bb404fbd1a70ad1360`.

The live bounded health check observed TriGLU metrics through global step 114,
all six GPUs at 100% utilization, and 311 GiB free. The old human-readable
monitor entry point was also repaired to forward to the active continuation
wave whenever its `state.env` exists; this fixes a display-only mismatch where
the old-wave header said COMPLETE while current-wave metrics were still read.

Owner interpretation: a delicate custom architecture may need a lower LR than
the paper baseline. Step-128 evaluation will decide whether the new decay is
promising. A poor result may justify a separately named, explicitly approved
rewind experiment from the durable checkpoint nearest the second reward
divergence around global step 90, with decay beginning earlier. The current
trajectory and checkpoints must remain intact; no automatic rewind is armed.

## Owner-Requested TriGLU-Priority Reordering

On 2026-07-15 the owner requested that, after the currently active baseline
`99 -> 128` segment and its evaluation, baseline continuation be deferred while
TriGLU is prioritized to global step 196 and then through a third 98-update
stage to global step 294. TriGLU retains the approved `5e-6 -> 5e-7` cosine
through step 196; the proposed third stage uses cosine `5e-7 -> 5e-8` over
steps 197-294. Baseline later resumes from its full step-128 state and catches
up from step 129 onward.

The scientific motivation is an upper-bound probe of TriGLU. PPO KL measures
policy difference and clipping is an imposed optimization constraint; neither
contains the architecture's inductive bias, so their small observed values do
not settle whether TriGLU benefits from a lower late-stage LR.

An attempted controller-only `SIGSTOP` at `2026-07-15T03:10:19+08:00` did not
arm the handoff because `screen` immediately continued its foreground shell.
The receipt was corrected at `03:11:15+08:00` to
`OWNER_PRIORITY_HANDOFF_NOT_ARMED`. The baseline worker was never stopped or
restarted, all six GPUs remained active, and controller control flow remained
unchanged. No successor wave has been created or launched pending explicit
approval of its full naming and reference/evaluation contract.

The owner subsequently approved the proposed successor run, screen, config,
controller, monitor, frozen step-98 reference, and `226/256/294` third-stage
evaluation contract. Implementation and remote preflight may proceed, but the
handoff watcher must not be armed until focused tests and pinned-environment
syntax/config checks pass. The active baseline worker remains untouched.

## First-Wave Checkpoint Pruning And Third-Stage Cadence

On 2026-07-15 the owner required the proposed third TriGLU 98-update stage to
mirror the second stage's evaluation and saving intervals. The exact proposed
contract is `save_freq=10`, with endpoint checkpoint-and-evaluation milestones
at global steps `226/256/294`. The retained third-stage steps are
`210,220,226,240,250,256,270,280,290,294`; transient regular checkpoints
`200/230/260` may be deleted only after their corresponding protected endpoint
and evaluation are durable.

The owner also approved retroactive conversion of the completed first-wave
checkpoint cadence from every five updates to every ten. A scoped remote audit
found no config, script, or active-continuation reference to odd-five
checkpoints. For each variant, steps `5,15,25,35,45,55,65,75,85,95` were then
deleted while `1,10,20,30,40,50,60,70,80,90,98` were preserved. The operation
reclaimed `173,464,035,328` bytes (about 161.55 GiB), leaving approximately
448.3 GiB free. TriGLU's checkpoint root changed from 173 GiB to 91 GiB and
baseline's from 168 GiB to 88 GiB. The active baseline `99 -> 128` process was
not stopped or modified. A before/delete/after manifest is stored remotely as
`runs/grpo_serial/triglu_baseline_6x5090_grpo_20to98_serial_20260712_v1/checkpoint_prune_5_to_10_20260715.txt`.

## Approved Successor Remote Preflight

The owner explicitly approved the successor naming and default frozen step-98
reference/evaluation contract. The scoped deployment archive SHA256 was
`f02395a8eb027a80b799b2cbae2273155af7dc52b4e6dc70e0d47c1ae4020711` and
matched remotely. On the pinned host:

- all three successor shell scripts passed `bash -n`;
- the transition utility and focused test module compiled;
- successor plus existing focused suites passed `22/22`;
- a synthetic six-rank hardlink transition passed;
- source scheduler state remained `base_lrs=[5e-6]`;
- target private extra-state copies used `base_lrs=[5e-7]`, `_last_lr=[5e-7]`,
  `last_epoch=196`, and an exact `global_step_196` tracker;
- model payloads shared inodes while scheduler extra-state files did not;
- no active training, checkpoint, dependency, or current controller process
  was modified during preflight.

The implementation is therefore eligible for watcher arming. The watcher waits
for the old controller to finish baseline step 128 plus evaluation/data-order
receipt and then finish TriGLU training to a validated step-158 checkpoint. It
terminates the old controller only after that trainer exits, reconstructs any
disposable interrupted export/eval under the successor root, and never shuts
down the host.

## Successor Handoff Watcher Armed

At `2026-07-15T05:11:01+08:00`, the bounded watcher was launched in screen
`qwen_triglu_priority_handoff_20260715_v1`. Its initial durable receipt is
`HANDOFF_PHASE=WAITING_FOR_TRIGLU_158`. The existing controller screen
`qwen_grpo_98to196_interleaved_20260714_v1` remained active, and its baseline
trainer was alive at global step 114 during the arming health check. The low
instantaneous GPU-utilization snapshot occurred at a step boundary; recent
metrics continued through step 114 and no process or error evidence indicated
a stall.

The watcher has not launched the successor screen yet, by design. It will wait
for baseline step 128 plus evaluation/data-order evidence, then for TriGLU to
finish training a complete six-rank step-158 checkpoint. Only after the
TriGLU trainer exits and the checkpoint validates will it stop the old
controller, clear disposable staging processes, and launch
`qwen_triglu_priority_to294_then_baseline196_20260715_v1`. Any interrupted
step-158 export/evaluation is regenerated under the successor run identity.
The host is never shut down by either watcher or controller.

Owner-facing status command:

```bash
cd /root/autodl-tmp/qwen3_1p7b_single_layer_rl
bash scripts/monitor_triglu_priority_to294_then_baseline196_20260715_v1.sh
```

## 2026-07-15 TriGLU-196 Boundary Reorder

Before the proposed TriGLU third stage began, the owner changed the order to
finish baseline through step 196 and then run an FP32 SwiGLU-only OFT control
through step 196. A dedicated watcher protects the TriGLU-196 boundary and
prevents the superseded third stage from silently starting. The authoritative
implementation record is
`docs/experiment_records/2026-07-15_baseline-then-oft-fp32-after-triglu196-record.md`.
