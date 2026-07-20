# 2026-07-12 SFT50K Four-Variant Completion Record / 中英混搭版

<!-- bilingual-companion-v1 -->

> **English original / 英文原件：** [2026-07-12_sft50k-four-variant-completion-record.md](../2026-07-12_sft50k-four-variant-completion-record.md)
> **中文导读 / Bilingual guide：** 这份 companion 的主题是 **2026-07-12 SFT50K Four-Variant Completion Record**。下面完整保留 original report 的 metrics、tables、commands、paths、hashes 与 scientific claims，同时加入中文 narrative layer 和 bilingual section handles，方便项目负责人阅读与后续 Agent 检索。
> **Evidence contract / 证据契约：** English original 保持 immutable；如果 companion 与原件出现 wording tension，以原件中的 exact numbers、machine-readable artifacts 和 source paths 为准，companion 不得 retroactively 改写实验事实。


Status: production wave complete; lightweight records and all four final
trainable checkpoint states archived and SHA-256 verified locally; full
exact-resume checkpoint archive staged on the OLD GPU for external upload.

## Run Identity And Completion

- Wave: `sft_ordered_20260711_sft50k_v1`.
- Model: Qwen3-1.7B-Base.
- Updated layer: Layer 10.
- Data: the shared decontaminated NuminaMath 50K materialization.
- Schedule: two SFT epochs, 3,916 optimizer steps per variant.
- Shared seed: `20260707`.
- Order: SHS, whole-layer baseline, TriGLU side FFN, OFT.
- Final state: all training and all scheduled evaluations complete; no active
  screen, evaluator, trainer, or GPU allocation remained at closeout.

The historical `ChildFailedError` / `SIGTERM` text still shown by the dashboard
belongs to the deliberate durable pause. It is not the terminal state of this
wave. Every variant reached step 3,916 and every reported benchmark has its
full expected row count.

## 最终训练指标 / Final Training Metrics
| Variant | Last-100 train loss | Validation loss | Train wall | Mean step |
|---|---:|---:|---:|---:|
| SHS | 0.5845 | 0.6209 | 5,822.787 s | 1.483 s |
| Whole-layer baseline | 0.5992 | 0.6368 | 3,714.761 s | 0.948 s |
| TriGLU side FFN | 0.5861 | 0.6230 | 3,876.889 s | 0.987 s |
| OFT | 0.5524 | 0.5893 | approximately 59 min | 0.914 s |

OFT obtains the lowest teacher-forced losses while producing by far the worst
autoregressive benchmark result. This is the strongest observation in the wave
that same-distribution SFT loss is not a sufficient model-selection proxy for
the intended mathematical reasoning behavior.

## 最终评估 / Final Evaluation
All values below are exact-match percentages. The four-task average uses
MATH-500, GSM8K, OlympiadBench, and AMC Average@32; greedy AMC is retained as a
separate modal-path diagnostic.

| Variant | MATH-500 (500) | GSM8K (1,319) | OlympiadBench (675) | AMC Avg@32 (1,280) | Four-task avg | AMC greedy (40) |
|---|---:|---:|---:|---:|---:|---:|
| SHS | 59.00 | 76.95 | 22.96 | 13.44 | 43.09 | 27.50 |
| Whole-layer baseline | 57.60 | 78.32 | 23.11 | 13.67 | 43.17 | 32.50 |
| TriGLU side FFN | 57.80 | 78.70 | 21.48 | 13.44 | 42.85 | 37.50 |
| OFT | 14.40 | 28.58 | 3.26 | 5.94 | 13.04 | 12.50 |
| Untuned base | - | - | - | 18.83 | - | 30.00 |

Relative to the whole-layer baseline:

- SHS gains 1.40 points on MATH-500, but loses 0.08 point on the four-task
  average and 5 points on the 40-item greedy AMC diagnostic.
- TriGLU gains 0.20 point on MATH-500, 0.38 point on GSM8K, and 5 points on
  greedy AMC, but loses 0.32 point on the four-task average because of its
  weaker OlympiadBench result.
- OFT collapses across every external benchmark despite its lower SFT loss.
  This result applies to the implemented mixed update contract, not to the OFT
  family in general.

The SFT wave therefore does not establish a broad quality winner among SHS,
baseline, and TriGLU. Their four-task averages differ by only 0.32 point, below
the evidential strength of a single stochastic run. The benchmark profiles do
differ: SHS leads MATH-500, TriGLU leads GSM8K and greedy AMC, and baseline
leads the aggregate by a negligible margin.

## 选择结论 / Selection Decision
For the first controlled single-layer RL comparison:

1. retain the whole-layer baseline as the mandatory control;
2. use SHS as the primary custom-architecture candidate because it has the
   strongest MATH-500 result and the most mature measured vLLM/rollout path;
3. retain TriGLU as the secondary candidate, especially if the greedy AMC
   signal survives a larger deterministic panel;
4. do not advance the current OFT configuration directly into the main RL wave
   until trainer/evaluator parity, checkpoint-curve, and pure-OFT controls
   explain the collapse.

This is a systems-aware experimental ordering, not a claim that SFT proved SHS
superior to TriGLU. Any final architecture claim requires matched RL seeds and
the same evaluation backend/protocol.

## Local Archive

The lightweight scientific record contains all non-checkpoint run files,
including predictions, reviews, metrics, manifests, diagnostics, and both
production logs. It was archived as:

`runs/sft_ordered_20260711_sft50k_v1/archive/sft_ordered_20260711_sft50k_v1_records.tar.gz`

SHA-256:
`a739606994acd7eabce738aa9482e56e197795489ac12557a49adabe442d3fa6`.

The four final trainable states and checkpoint manifests were separately
archived as:

`runs/sft_ordered_20260711_sft50k_v1/archive/sft_ordered_20260711_sft50k_v1_final_trainable.tar`

SHA-256:
`690092464aa49fe99ba4994bb1e204e06fa4e7c7307808a0a6646ead0208782e`.

Individual final `trainable_state.pt` hashes are:

| Variant | SHA-256 |
|---|---|
| SHS | `ebb4cd92f6e890c17cf0e14a883557358dff927e901b69588f1867e6dd016712` |
| Whole-layer baseline | `d69424814ef849d9c6120714de93c2f2c2b436f5224ee7e30b12d090b42853df` |
| TriGLU side FFN | `8a463168b4dce0f698357a821dfaca2d7b7fa90032841adb717251e323c48ab8` |
| OFT | `8856f008c87b1c18c39fc43f6486d0bcced0252e2267b69f42f02cffc770d8f9` |

The larger `trainer_state.pt` files are deliberately treated as an optional
exact-SFT-resume archive. They are not needed for inference, evaluation,
deployment export, or initializing downstream RL from the completed SFT
weights. Their hashes remain in the locally archived
`checkpoint_inventory.txt` so a later copy can be verified.

The full exact-resume archive, including every final `trainer_state.pt`, is
staged on the OLD GPU at:

`/root/autodl-tmp/qwen3_1p7b_single_layer_rl/shutdown_ready/sft_ordered_20260711_sft50k_v1/sft_ordered_20260711_sft50k_v1_final_checkpoints_full.tar`

It is 1,512,120,320 bytes and has SHA-256
`8f518f5e5f21e754d9e5c7c0376c5aa698bd51a7a358c59857cabd1d5db93852`.
The tar and its adjacent `.sha256` file are intended for external artifact
storage rather than Git. Standard GitHub rejects individual blobs over 100 MB;
Git contains only source, compact evidence, documentation, and checksums.
