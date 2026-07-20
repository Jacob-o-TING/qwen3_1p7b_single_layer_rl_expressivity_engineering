# 2026-07-11 SHS SFT 2048 Production Gate Record / 中英混搭版

<!-- bilingual-companion-v1 -->

> **English original / 英文原件：** [2026-07-11_sft-shs-2048-production-gate-record.md](../2026-07-11_sft-shs-2048-production-gate-record.md)
> **中文导读 / Bilingual guide：** 这份 companion 的主题是 **2026-07-11 SHS SFT 2048 Production Gate Record**。下面完整保留 original report 的 metrics、tables、commands、paths、hashes 与 scientific claims，同时加入中文 narrative layer 和 bilingual section handles，方便项目负责人阅读与后续 Agent 检索。
> **Evidence contract / 证据契约：** English original 保持 immutable；如果 companion 与原件出现 wording tension，以原件中的 exact numbers、machine-readable artifacts 和 source paths 为准，companion 不得 retroactively 改写实验事实。


## 范围 / Scope
This gate selects eager versus `torch.compile` for the first production SHS
SFT run using the actual single-GPU training shape:

- sequence length: 2,048;
- micro-batch size: 1;
- gradient accumulation: 8;
- two warmup optimizer steps and five timed optimizer steps;
- identical canonical data, initialization seed, data seed, and sample order.

Hardware: one NVIDIA RTX PRO 6000 Blackwell Server Edition with 97,887 MiB.

## Canonical Packed Data

- Source JSONL SHA-256:
  `723d9cacd4fed74efa871ef0d1ff59535f5882f23cf798ff09bb9b2b3a5dc4fc`.
- Source examples: 50,000.
- Packed sequences: 15,664.
- Packing order SHA-256:
  `5c9a4acf627b42cf173b51fb950d02c86a53e23c3c2b6388cf97451cac7ffd39`.
- Overlong prompts dropped: 0.
- Solution tokens truncated after preserving full prompts: 13,391.
- Steps per epoch: 1,958.
- Production two-epoch optimizer steps: 3,916.
- Checkpoint/validation steps: 392, 979, 1,958, 2,937, and 3,916.

## 结果 / Results
| Case | Initial loss | Cold step | Median step | Assistant tok/s | Peak allocated |
| --- | ---: | ---: | ---: | ---: | ---: |
| SHS eager | 0.578135 | 2.104 s | 1.487 s | 7,681.8 | 18.39 GB |
| SHS compile | 0.578302 | 135.139 s | 1.891 s | 6,045.4 | 11.34 GB |

Derived results:

- Compile speedup: 0.787x; compiled steady state was 27.1% slower.
- Compile initial-loss relative delta: 0.000289.
- Compile break-even: none.
- Compile captured one graph with 3,915 calls and no graph-break evidence.
- Compile reduced peak allocation by 7.05 GB, but memory is not the limiting
  resource on the 98 GB device.

## Decision And Estimate

Production SHS remains in eager mode. The base production config already uses
`compile_mode: eager`, so no config override is required.

Using the measured eager mean of 1.489 seconds per optimizer step, 3,916 steps
project to approximately 1.62 hours of pure optimizer-loop time. Allowing for
model startup, validation at five milestones, and five compact checkpoints,
the single-GPU SHS training budget is 1.7-1.9 hours. This estimate excludes the
subsequent full benchmark evaluation.

No result met the economically abnormal shutdown condition. Disk free space
after the gate was 222 GB.

## 产物 / Artifacts
- Remote run root:
  `runs/sft_shs_2048_production_gate_20260711_v1`.
- Remote log:
  `logs/sft_shs_2048_production_gate_20260711_v1.log`.
- Reusable launchers:
  `scripts/start_sft_shs_2048_production_gate.sh` and
  `scripts/launch_sft_shs_2048_production_gate.sh`.
