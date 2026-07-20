# 2026-07-11 SFT Durable Pause Record / 中英混搭版

<!-- bilingual-companion-v1 -->

> **English original / 英文原件：** [2026-07-11_sft-durable-pause-record.md](../2026-07-11_sft-durable-pause-record.md)
> **中文导读 / Bilingual guide：** 这份 companion 的主题是 **2026-07-11 SFT Durable Pause Record**。下面完整保留 original report 的 metrics、tables、commands、paths、hashes 与 scientific claims，同时加入中文 narrative layer 和 bilingual section handles，方便项目负责人阅读与后续 Agent 检索。
> **Evidence contract / 证据契约：** English original 保持 immutable；如果 companion 与原件出现 wording tension，以原件中的 exact numbers、machine-readable artifacts 和 source paths 为准，companion 不得 retroactively 改写实验事实。


## 范围 / Scope
The active `sft_ordered_20260711_sft50k_v1` wave was deliberately paused so the
original RTX PRO 6000 instance could be shut down and cloned for isolated kernel
and continuous-batching development. The production run was not declared
complete.

Pause request time: `2026-07-11T23:31:35+08:00`.

## Durable State

- SHS training and all primary evaluations: complete with verified receipt.
- Whole-layer baseline training and all primary evaluations: complete with
  verified receipt.
- TriGLU training: complete at step 3916 with final checkpoint.
- TriGLU MATH-500: complete, 500 cached rows.
- TriGLU GSM8K: partial, 982 of 1,319 cached rows; 337 remain.
- TriGLU OlympiadBench and AMC Average@32: not started.
- OFT: queued, training not started.
- Main production `screen`: stopped.
- GPU after synchronization: 0% utilization and 0 MiB memory used.
- Disk at the preceding monitor: approximately 215 GB free.

Authoritative TriGLU main cache:

```text
/root/autodl-tmp/qwen3_1p7b_single_layer_rl/runs/sft_ordered_20260711_sft50k_v1/evaluations/layer10_whole_layer_triglu_side_ffn/main/20260711_213137
```

## Cache Integrity

| Artifact | Rows | Unique indices | SHA-256 |
|---|---:|---:|---|
| MATH predictions | 500 | 500 | `5864aacce0882ecf41df60d828b1e31649cdf11a2bc3ed5a189becbae6ca9804` |
| MATH reviews | 500 | 500 | `41ff4145ec17962ab56eecaa39a3aa5c09ea687b5b9b2c36327575ed25c0721f` |
| GSM8K predictions | 982 | 982 | `35fb7015e7abaee59143ee576b6bbc189edf7920d2d09706a27233d7665f3b57` |
| GSM8K reviews | 982 | 982 | `a67d4be46eabe2b4c832d49275bffdd85bf1b4b12ea403bfb679f122926f0290` |

The lower-coverage aborted timestamp directory remains non-authoritative and is
retained only for audit.

## Resume Contract

Run:

```bash
cd /root/autodl-tmp/qwen3_1p7b_single_layer_rl
bash scripts/launch_sft_ordered_resume.sh
```

Expected initial behavior:

1. SHS and baseline training/evaluation are skipped by verified checkpoints and
   receipts.
2. TriGLU training is skipped by its exact final checkpoint.
3. Cache discovery selects `20260711_213137` by unique-row coverage.
4. EvalScope reports 500 cached MATH rows and 982 cached GSM8K rows, then
   generates only the remaining 337 GSM8K rows before continuing the remaining
   benchmarks.
5. The ordered wave subsequently runs TriGLU greedy control and OFT
   training/evaluation under the original run root.

Do not use plain `launch_sft_ordered_variants.sh` from an unverified older source
bundle. The committed resume launcher and cache-aware ordered launcher are the
approved recovery entrypoint.
