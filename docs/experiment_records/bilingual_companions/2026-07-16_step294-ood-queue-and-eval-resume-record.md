# Step-294 OOD Queue And Eval Resume Record / 中英混搭版

<!-- bilingual-companion-v1 -->

> **English original / 英文原件：** [2026-07-16_step294-ood-queue-and-eval-resume-record.md](../2026-07-16_step294-ood-queue-and-eval-resume-record.md)
> **中文导读 / Bilingual guide：** 这份 companion 的主题是 **Step-294 OOD Queue And Eval Resume Record**。下面完整保留 original report 的 metrics、tables、commands、paths、hashes 与 scientific claims，同时加入中文 narrative layer 和 bilingual section handles，方便项目负责人阅读与后续 Agent 检索。
> **Evidence contract / 证据契约：** English original 保持 immutable；如果 companion 与原件出现 wording tension，以原件中的 exact numbers、machine-readable artifacts 和 source paths 为准，companion 不得 retroactively 改写实验事实。


Date: 2026-07-16

Status: **GRPO RECOVERED; OOD QUEUE VERIFIED AND ARMED**

## Baseline-196 Eval Stall

The baseline step-196 six-GPU math evaluation reached five complete ranks, but
the `amc_sample_00_10` rank stopped writing progress at `128/440` at
12:22:26 +08:00. At 13:03 it had consumed approximately 44 minutes of CPU time
with no file or progress update. This met the economically-abnormal gate.

The process was terminated without deleting any evaluation directory. The
initial handoff watcher correctly failed closed because the baseline-196
overall completion marker was absent. Two restart-only defects were then
identified and repaired:

1. the controller preflight required old boundary checkpoints already removed
   by the approved retention policy;
2. it required a step-98 reference marker deliberately blocked during the
   step-196 KL-reference reset before checking whether the old training segment
   was already complete.

The repaired controller now validates retained step-196 checkpoints for a late
resume, checks completed training before requiring an old reference, preserves
partial evaluation directories, skips ranks with `model_summary.json`, and
passes the latest EvalScope timestamp to the incomplete rank as cache.

Recovery evidence was explicit: five ranks logged
`EVAL_RANK_ALREADY_COMPLETE`; the incomplete rank logged
`128 already fully cached`, `312 items to process`, and advanced beyond the
old stall to `257/440`. No completed shard or checkpoint was rerun.

Implementation commit: `dff76df6` (`Resume partial parallel eval shards
safely`).

## OOD Queue

The approved OOD identity and order are implemented in
`docs/experiment_plans/2026-07-16_qwen3-step294-ood-triglu-baseline-base-plan.md`.
The queue waits for both variants to finish step 294 and then evaluates
TriGLU, baseline, and untuned base in that order. The eight paper-named OOD
benchmarks are partitioned across all six GPUs, with per-rank cache resume and
a privilege-dropped/seccomp code-review backend.

Remote pinned-environment verification passed:

- OOD queue tests: `3/3`;
- EvalScope configuration tests: `4/4`;
- all new shell scripts passed `bash -n` and Python files passed `py_compile`;
- the privilege-dropped backend executed a harmless payload with output `42`;
- separate socket and `os.execv` payloads both failed under seccomp as
  required, with isolation receipt
  `uid65534+no_new_privs+rlimits+seccomp_no_network_no_exec`.

The approved `qwen_ood_step294_20260716_v1` screen was then armed to wait for
the durable GRPO completion boundary. This record must be updated again after
the OOD run completes. A source implementation, armed watcher, or partial
benchmark result is not itself a completed scientific result.

## Storage Receipt

- Filesystem total: 833,223,655,424 bytes.
- Used at audit: 502,952,808,448 bytes.
- Available at audit: 330,270,846,976 bytes.
- Typical GRPO checkpoint: approximately 8.799 GB decimal.
- Conservative added peak through both step-294 stages plus OOD: 212-222 GB.
- Conservative residual free space: 108-118 GB decimal.
- Decision: capacity pass under the existing 100 GB runtime guard.

## 继承待办 / Pending Obligations Carried Forward
Canonical registry:
`docs/experiment_plans/PENDING_2026-07-12_mandatory-experiment-gates.md`

- **PENDING-01 Eval Parity Matrix:** deferred and still pending.
- **PENDING-02 Pure-BF16 SHS And TriGLU:** deferred and still pending.
- **PENDING-03 Registered SHS CausalLM Route:** deferred and still pending.
