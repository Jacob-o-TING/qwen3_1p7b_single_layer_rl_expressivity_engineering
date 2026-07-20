# 2026-07-11 SFT Production Orchestration Gate Record / 中英混搭版

<!-- bilingual-companion-v1 -->

> **English original / 英文原件：** [2026-07-11_sft-production-orchestration-gate-record.md](../2026-07-11_sft-production-orchestration-gate-record.md)
> **中文导读 / Bilingual guide：** 这份 companion 的主题是 **2026-07-11 SFT Production Orchestration Gate Record**。下面完整保留 original report 的 metrics、tables、commands、paths、hashes 与 scientific claims，同时加入中文 narrative layer 和 bilingual section handles，方便项目负责人阅读与后续 Agent 检索。
> **Evidence contract / 证据契约：** English original 保持 immutable；如果 companion 与原件出现 wording tension，以原件中的 exact numbers、machine-readable artifacts 和 source paths 为准，companion 不得 retroactively 改写实验事实。


## 范围 / Scope
This gate completes the resumable ordered SFT control path. It does not report
model quality and did not launch the 50K production wave by itself.

The production order is fixed to:

1. SHS.
2. Layer-10 whole-layer baseline.
3. TriGLU side FFN.
4. OFT.

Each variant now performs training, exact final-checkpoint verification, all
four paper-pinned evaluations, and evaluation receipt hashing before the next
variant starts.

## Distributed Batch Contract

The launcher auto-detects visible GPUs and derives gradient accumulation from a
target effective packed batch size of 8:

| Visible GPUs | Micro-batch per GPU | Accumulation | Effective packed batch |
|---:|---:|---:|---:|
| 1 | 1 | 8 | 8 |
| 4 | 1 | 2 | 8 |

The calculation rejects non-divisible topologies instead of silently changing
the experiment. Together with the pinned initialization, dataloader, and model
surgery seeds, this preserves the controlled global sample order across the
one-GPU and four-GPU launch modes.

## Handoff Contract

Training is accepted as complete only when all of the following agree on the
exact final step:

- `train_result.json` is not a benchmark result.
- `train_result.json.global_step` equals `run_manifest.json.total_steps`.
- `checkpoints/latest.json.global_step` equals the same total.
- The final checkpoint directory name encodes that step.
- `trainable_state.pt`, `trainer_state.pt`, and checkpoint `manifest.json` are
  present and non-empty.
- The checkpoint manifest records the same total step count.

Evaluation completion is represented by `evaluation_complete.json`. The
receipt binds the final checkpoint path, config SHA-256, evaluation-manifest
SHA-256, and one report SHA-256 for each of `paper_math500`, `paper_gsm8k`,
`paper_olympiadbench`, and `paper_amc23`. A bounded evaluation cannot receive a
production receipt. Modified or missing reports invalidate the receipt and
force evaluation to run again.

## 验证 / Verification
- Local targeted tests: 8 passed.
- Remote full suite: 45 passed.
- Remote shell syntax checks passed for the ordered launcher, starter, and
  monitor.
- Real one-GPU accumulation result: 8.
- Simulated four-GPU accumulation result: 2.
- The completed two-step smoke run resolved exactly to
  `runs/sft_milestone_smoke_20260711/checkpoints/step_00000002`.
- The four-adapter bounded preflight produced a hash-valid test receipt and
  subsequently returned `complete`.

The limited preflight receipt is isolated under ignored evaluation artifacts
and is not eligible as a production-wave receipt.

## Observability And Resume

`scripts/monitor_sft.sh` now reports ordered phase markers, latest optimizer
step, validation/checkpoint milestones, EvalScope benchmark progress, report
writes, GPU state, disk state, and recent errors. The ordered job remains in a
remote `screen` session, reducing dependence on repeated gateway handshakes.

Reusing an explicit `RUN_STAMP` reuses the same run and evaluation roots. The
trainer resumes its optimizer/RNG/sampler cursor from the latest checkpoint;
the orchestration skips evaluation only when the final checkpoint and hashed
receipt still match.
