# Step-294 OOD Handoff And Preflight Recovery Record / 中英混搭版

<!-- bilingual-companion-v1 -->

> **English original / 英文原件：** [2026-07-17_step294-ood-handoff-and-preflight-recovery-record.md](../2026-07-17_step294-ood-handoff-and-preflight-recovery-record.md)
> **中文导读 / Bilingual guide：** 这份 companion 的主题是 **Step-294 OOD Handoff And Preflight Recovery Record**。下面完整保留 original report 的 metrics、tables、commands、paths、hashes 与 scientific claims，同时加入中文 narrative layer 和 bilingual section handles，方便项目负责人阅读与后续 Agent 检索。
> **Evidence contract / 证据契约：** English original 保持 immutable；如果 companion 与原件出现 wording tension，以原件中的 exact numbers、machine-readable artifacts 和 source paths 为准，companion 不得 retroactively 改写实验事实。


Date: 2026-07-17

Status: **RECOVERED AND ACTIVE - DURABLE BARRIER AND SIX-GPU OOD VERIFIED LIVE**

## Observed State

TriGLU completed global step 294 and its six-GPU Math evaluation. Baseline
remained at global step 196 with no update, optimizer, RNG, or data-cursor
advance into the third stage. All six GPUs were idle when the incident was
inspected.

The first pre-baseline OOD handoff failed for two independent reasons:

1. the signal-based watcher paused one controller-shell PID, but a descendant
   raced into baseline startup while OOD acquired the GPUs. Baseline vLLM saw
   only 4.06 GiB free against its 15.68 GiB request and failed before update
   197;
2. OOD preflight found missing `langdetect`, EvalScope's unused
   `ms_enclave` import gate despite the project-local sandbox backend, and
   missing explicit sample metadata for generic OOD datasets under seeded
   generation.

No checkpoint or completed evaluation was deleted. Existing EvalScope cache
directories remain the resume source.

## Implemented Repair

- The GRPO controller now writes
  `TRIGLU_294_PRE_BASELINE_OOD_READY` and blocks on
  `PRE_BASELINE_OOD_COMPLETE` before preparing or launching baseline's
  step-196-to-294 transition.
- The OOD watcher no longer sends `SIGSTOP` or `SIGCONT`; it starts the first
  OOD pair only after the durable controller barrier and idle-GPU checks.
- Generic OOD requests use a deterministic canonical identity containing the
  dataset namespace and rendered-prompt SHA-256. Existing explicit paper
  benchmark identities are preserved exactly.
- The project-local privilege-dropped sandbox adapter bypasses only the pinned
  EvalScope validator's unused `ms_enclave` import requirement. The actual
  sandbox backend remains UID/GID 65534 plus `no_new_privs`, resource limits,
  and seccomp restrictions.
- `langdetect==1.0.9` is pinned and imported before any six-GPU OOD launch.

Implementation commit: `0c29a10` (`Harden OOD handoff and evaluator
preflight`).

## Verification So Far

- Local focused tests: 8/8 passed.
- Remote pinned-environment focused tests: 4/4 evaluator batching and identity,
  4/4 OOD queue, and 12/12 GRPO continuation tests passed.
- Remote shell syntax and Python compilation passed.
- A real pinned EvalScope `TaskConfig` accepted the project-local sandbox with
  `sandbox.enabled=true` while `ms_enclave` remained absent.
- A bounded six-GPU, one-item-per-shard interface preflight crossed the prior
  identity and dependency failures. Five ranks completed end to end;
  LiveCodeBench crossed TaskConfig, sandbox, canonical identity, and engine
  initialization, then remained in first-run multi-file dataset
  materialization. That temporary process was terminated without deleting the
  global partial cache so the real OOD run could resume the download while the
  other five production shards made progress.
- Replaying the GRPO controller recognized every completed training and Math
  boundary, wrote `TRIGLU_294_PRE_BASELINE_OOD_READY`, and reached
  `PHASE=WAITING_PRE_BASELINE_OOD`, `VARIANT=triglu`, `TARGET=294`. It launched
  no evaluator or trainer while waiting.
- The first source-bundle restart failed before GPU launch with exit 126
  because three shell entrypoints were archived as mode 0644. Their executable
  bits were restored remotely and recorded in Git as mode 0755.
- The final OOD restart emitted all three prerequisite receipts and
  `OOD_PHASE_START ... models=triglu untuned_base`. Six independent TP=1 vLLM
  evaluator processes were live, all six GPUs held the intended model engines,
  and observed utilization ranged from 48% to 85% on active compute ranks.
  The OOD screen and waiting GRPO controller screen were both detached and
  healthy at closeout.

## 继承待办 / Pending Obligations Carried Forward
Canonical registry:
`docs/experiment_plans/PENDING_2026-07-12_mandatory-experiment-gates.md`

- **PENDING-01 Eval Parity Matrix:** deferred; this OOD recovery does not close
  evaluator parity.
- **PENDING-02 Pure-BF16 SHS And TriGLU:** deferred; the active TriGLU
  checkpoint retains its historical dtype contract.
- **PENDING-03 Registered SHS CausalLM Route:** deferred; SHS is outside this
  OOD wave.
