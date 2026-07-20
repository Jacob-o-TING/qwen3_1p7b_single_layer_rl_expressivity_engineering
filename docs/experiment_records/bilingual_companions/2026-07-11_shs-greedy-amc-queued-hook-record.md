# 2026-07-11 SHS Greedy AMC Queued Hook Record / 中英混搭版

<!-- bilingual-companion-v1 -->

> **English original / 英文原件：** [2026-07-11_shs-greedy-amc-queued-hook-record.md](../2026-07-11_shs-greedy-amc-queued-hook-record.md)
> **中文导读 / Bilingual guide：** 这份 companion 的主题是 **2026-07-11 SHS Greedy AMC Queued Hook Record**。下面完整保留 original report 的 metrics、tables、commands、paths、hashes 与 scientific claims，同时加入中文 narrative layer 和 bilingual section handles，方便项目负责人阅读与后续 Agent 检索。
> **Evidence contract / 证据契约：** English original 保持 immutable；如果 companion 与原件出现 wording tension，以原件中的 exact numbers、machine-readable artifacts 和 source paths 为准，companion 不得 retroactively 改写实验事实。


## Approved Placement

SHS, whole-layer baseline, and untuned base greedy AMC modal-path diagnostics,
followed by untuned-base AMC Average@32, are queued
synchronously after the whole-layer baseline completes all four final-evaluation
benchmarks and before TriGLU training begins. They are not inserted between
baseline benchmarks because the three baseline main benchmarks execute inside
one already-running EvalScope `run_task` call with no safe shell boundary.

## Configuration

```text
diagnostic IDs:
  amc_greedy_modal_path_shs_sft50k_v1
  amc_greedy_modal_path_baseline_sft50k_v1
  amc_greedy_modal_path_untuned_qwen3_1p7b_base_v1
  amc_average_at_32_untuned_qwen3_1p7b_base_v1
control order: SHS greedy, baseline greedy, base greedy, base Average@32
checkpoints: completed SHS and baseline step_00003916
base source: /root/autodl-tmp/qwen3_single_layer_rl/models/Qwen3-1.7B-Base
base isolation: base_model_only=true, checkpoint_dir=null, no variant application
dataset: paper_amc23, 40 prompts
decode: do_sample=false, one repeat, max_tokens=3072
concurrency: eval_batch_size=16
preflight: 16 prompts, max_tokens=64, eval_batch_size=16
outputs: <ordered-run-root>/diagnostics/<diagnostic-id>
trigger: immediately before layer10_whole_layer_triglu_side_ffn_sft.yaml
```

All four diagnostics run within the existing serial controller. They do not create a
second screen or overlap baseline evaluation or training. Each completion
receipt requires exactly 40 review rows and stores SHA-256 hashes for the report,
predictions, and reviews before the hook allows TriGLU training to start.

## Deployment Verification

Commit `576b6d2` (`Queue SHS greedy AMC diagnostic before TriGLU`) was synced to
the active AutoDL project without changing the already-loaded baseline evaluator.
Remote verification confirmed:

- `bash -n` passed for the single-node launcher, diagnostic hook, and monitor;
- two focused remote hook tests passed;
- the trigger matches only the exact TriGLU config basename;
- the default diagnostic batch size is 16;
- production uses `paper_amc23` as the sole main dataset with AMC repeats set to
  zero, yielding one greedy response per question;
- no diagnostic receipt existed, so the hook is queued and not already run;
- the original ordered screen remained detached and healthy;
- baseline AMC continued to approximately 1,133 of 1,280 responses after sync;
- GPU memory remained approximately 37.7 GB with no traceback or OOM.

The preflight adds one short model load and 16 responses capped at 64 tokens. It
is retained because batch size 16 has not previously been production-tested for
the SHS evaluator; a fail-fast gate is preferable to aborting the serial wave on
the full diagnostic.

## Untuned Base Extension Deployment

The approved untuned Qwen3-1.7B-Base greedy control was added after the paired
SHS and trained whole-layer baseline controls. Remote verification confirmed:

- the base weights file exists at the recorded path and is 3,441,185,608 bytes;
- base mode applies no architecture variant and loads no SFT checkpoint;
- its evaluation manifest and completion receipt record
  `base_model_only=true` and `checkpoint_dir=null`;
- three model-source validation tests and two hook/order tests passed;
- Python bytecode compilation and both relevant shell syntax checks passed;
- the active baseline evaluation remained uninterrupted at approximately
  242/500 MATH-500 examples, with no traceback or OOM.

The subsequently approved base Average@32 control uses an AMC-only phase with
32 repeats and exactly 1,280 expected reviews. Post-primary-evaluation greedy
controls are also queued for TriGLU and OFT under these IDs:

```text
amc_greedy_modal_path_triglu_sft50k_v1
amc_greedy_modal_path_oft_sft50k_v1
```

Each runs after that variant's primary evaluation receipt, which for both
variants follows their AMC Average@32 phase.

Because the active parent shell parsed its ordered loop before these lines were
deployed, the current wave also uses two receipt-gated bridges: the freshly
loaded OFT single-node launcher runs the TriGLU control before OFT training, and
`wait_and_run_oft_greedy_amc.sh` waits for the OFT primary receipt before the
terminal OFT control. Future waves use the ordered-controller hooks directly;
duplicate execution is prevented by each diagnostic completion receipt.

## Extended Control Deployment Verification

The base Average@32 and post-variant greedy extension was atomically deployed
while the active parent shell retained its original parsed script inode. Remote
verification confirmed:

- four relevant shell scripts passed `bash -n`;
- nine focused evaluator, source-mode, and hook-order tests passed;
- the AMC-only invalid-repeat case failed fast before importing EvalScope;
- the pre-TriGLU order assertion passed for SHS greedy, baseline greedy, base
  greedy, and base Average@32;
- the detached OFT receipt waiter started under
  `qwen_sft_oft_greedy_tail_20260711_v1` and entered its 60-second sleep loop;
- the primary ordered screen remained detached and healthy, with baseline
  MATH-500 at approximately 402/500 and no traceback or OOM.
