# Step-294 OOD Interleaving Amendment Record

Date: 2026-07-16

Status: **IMPLEMENTED, VERIFIED, AND REMOTE RE-ARMED**

## Owner-Approved Order

The prior queue waited for both GRPO variants to finish and then evaluated
TriGLU, baseline, and the untuned base. The owner replaced it with this strict
order:

1. finish TriGLU's third 98 updates;
2. finish TriGLU step-294 Math evaluation;
3. run TriGLU step-294 OOD evaluation;
4. run untuned Qwen3-1.7B-Base OOD evaluation;
5. run baseline's third 98 updates;
6. finish baseline step-294 Math evaluation; and
7. run baseline step-294 OOD evaluation.

## Live-Run Adaptation

The GRPO controller was already running when this amendment was approved, so
its process is not restarted. The replacement OOD watcher waits for the
TriGLU Math completion marker, pauses only the direct controller-shell child of
the named GRPO screen, waits for the Math evaluator and GPU allocations to
exit, executes the first two OOD models, and resumes the same controller. The
training child is never signalled. A trap resumes the controller after any
watcher failure, interrupt, termination, or screen hangup.

The OOD runner now accepts explicit model subsets under a file lock. TriGLU
plus untuned-base completion writes `PRE_BASELINE_OOD_COMPLETE`; baseline
completion writes the original aggregate comparison and `OOD_COMPLETE`.
Existing per-rank resume markers remain authoritative.

## Verification And Arming Receipt

- Remote `bash -n` passed for both the split OOD runner and replacement
  watcher.
- Remote focused tests passed: `test_ood_step294_queue.py` 3/3 and
  `test_triglu_priority_to294_then_baseline196.py` 12/12.
- The controller selector resolved screen PID `190322` to its direct child
  controller PID `190324`; the live GRPO actor PID `191684` was not selected.
- `/usr/bin/flock` is present on the remote image.
- The sleeping legacy OOD screen was replaced, then re-armed with explicit
  `SIGHUP` fail-safe coverage as final screen PID `258396` at
  `2026-07-16T14:02:20+08:00`.
- The new watcher emitted the exact order receipt
  `triglu_math,triglu_ood,untuned_base_ood,baseline_third98,baseline_math,baseline_ood`.
- After re-arming, GRPO remained `PHASE=TRAIN`, `VARIANT=triglu`,
  `TARGET=226`; the live actor and controller PIDs were unchanged.

No controller signal has been sent yet. The pause/resume barrier remains armed
for the future durable TriGLU step-294 Math completion marker.

## Pending Obligations Carried Forward

Canonical registry:
`docs/experiment_plans/PENDING_2026-07-12_mandatory-experiment-gates.md`

- **PENDING-01 Eval Parity Matrix:** deferred; this remains the named vLLM OOD
  protocol and does not close evaluator parity.
- **PENDING-02 Pure-BF16 SHS And TriGLU:** deferred; the evaluated TriGLU path
  retains its historical dtype contract.
- **PENDING-03 Registered SHS CausalLM Route:** deferred; SHS is outside this
  OOD wave.
