# GPQA-Diamond-Freeform-126 Major-Checkpoint Eval Preparation Record

Date: 2026-07-18

Status: **SOURCE READY; REMOTE ASSETS / GPU EXECUTION PENDING**

## Owner Decision / 项目负责人决策

项目负责人批准为 completed GRPO trajectory 增加一个真正 question-only 的
`GPQA-Diamond-Freeform-126` protocol，覆盖 TriGLU 与 whole-layer baseline 的
global steps `158/196/226/256/294`。本轮 preparation 明确不得 interrupt 当前正在运行的 Other/OOD wave，
所以只完成 local source、docs 与 deterministic tests；没有 SSH、没有 remote process mutation，也没有下载
gated dataset 或 Qwen3-4B matcher weights。

Approved identity 保持不变：

- Run ID: `qwen3_1p7b_gpqa_diamond_freeform126_greedy3072_qwen3_4b_match_6x5090_steps158_196_226_256_294_20260718_v1`;
- screen: `qwen_gpqa_free126_majorsteps6_20260718_v1`;
- result root: `runs/freeform_eval/<RUN_ID>/`.

## Implemented Surface / 已实现

| Component | File | Contract |
|---|---|---|
| experiment config | `configs/eval/qwen3_1p7b_gpqa_diamond_freeform126_majorsteps_6x5090_20260718_v1.yaml` | exact 10-cell ledger, greedy3072, 6 x TP=1, Qwen3-4B matcher |
| protocol core | `src/qwen_single_layer_rl/eval/gpqa_freeform.py` | normalize, hash, shard, exact merge, tag parsing, matcher ledger, denominator accounting |
| asset builder | `scripts/prepare_gpqa_diamond_freeform126_assets.py` | resolve immutable HF revisions, 126-row ledger, matcher receipt; no credential serialization |
| checkpoint exporter | `scripts/export_qwen3_major_checkpoint.py` | six-shard checkpoint validation, existing-export reuse, scoped TriGLU staging repair |
| GPU worker | `scripts/run_gpqa_diamond_freeform126_worker.py` | vLLM generation, TriGLU registration/dispatch receipt, matcher load-once shards, atomic outputs |
| controller | `scripts/run_qwen3_1p7b_gpqa_diamond_freeform126_majorsteps_6x5090_20260718_v1.sh` | autonomous screen, fail-closed contention gates, resume markers, serial cell export/generation then parallel match |
| monitor | `scripts/monitor_qwen3_1p7b_gpqa_diamond_freeform126_majorsteps_6x5090_20260718_v1.sh` | existing dashboard grammar, paired score table, progress/speed/ETA, diagnostics/GPU/disk/errors |
| tests | `tests/test_gpqa_diamond_freeform126_majorsteps_eval.py` | schema, balance, parsing, exact cover, denominator, immutable order, monitor, credential boundary |

## Scientific Contract / 科学边界

Generation prompt 只包含 `question`，不包含 choices；每个 cell 使用同一个按 canonical `question_id`
排序的 126-row ledger，`ordered_index % 6` 后每张 GPU 恰好 21 条。Generation 是
`temperature=0`, pass@1, cap `3072`，这是 approved longitudinal protocol，不冒充原 answer-matching paper
的 `temperature=0.6` / cap `16384` generation setting。

十个 generation cells 全部完成后才建立 1,260-row matcher ledger。六个独立 Qwen3-4B no-thinking
replicas 每张处理 210 条，只 load 一次 matcher。Primary decision 必须是 exact
`<answer>0</answer>` 或 `<answer>1</answer>`；任何多 tag、missing tag 或附加文本都进入
`matcher_failure`，仍计入 126 denominator，绝不 silent-cast 成 wrong/correct。

Monitor layout 刻意与 existing major-checkpoint dashboard 保持一致：先显示 controller / phase / current
cell / elapsed，再给 progress bar、recent speed、ETA；PRIMARY table 按 global step 同组展示
TriGLU、baseline 与 delta，随后才显示 cap hits、missing tags、matcher failures、audit queue、GPU 和 disk。
Official MCQ `GPQA-Diamond` 继续作为 separate metric，不生成 cross-protocol hard average。

Protocol provenance: Chandak et al., *Answer Matching Outperforms Multiple Choice for Language Model
Evaluation*, Appendix F.1, `https://arxiv.org/pdf/2507.02856`. Implementation preserves its core
reference-aware matching semantics, the published `<1%` numeric relative-error rule, no incorrect options,
and `0/1` answer tags while using the owner-approved no-thinking matcher mode.

## Local Verification

- `python -m py_compile`: PASS for four new Python modules and test;
- Python AST parse: PASS;
- focused unittest: **8/8 PASS**;
- `bash -n`: PASS for controller and monitor using local Git Bash;
- `git diff --check`: PASS.

Synthetic evidence verifies:

- dataset ledger `126/126`, unique IDs, six ranks exactly `[21,21,21,21,21,21]`;
- ten generation cells produce matcher ledger `1260/1260`;
- matcher ranks exactly `[210,210,210,210,210,210]`;
- matcher failures remain explicit and each cell denominator closes to 126;
- launcher order is exactly TriGLU then baseline at `158/196/226/256/294`.

## Deferred Remote Gates / 待远端执行

在 current Other/OOD wave complete 且 six GPUs idle 后，依次执行：

1. use `prepare_gpqa_diamond_freeform126_assets.py ... all` to authenticate, pin the gated dataset SHA,
   materialize the 126-row ledger, download/pin Qwen3-4B, and write both receipts;
2. launch the approved controller once; it self-detaches into the approved `screen`;
3. inspect the approved monitor with one bounded invocation;
4. require all 10 generation cells, six matcher shards, final paired summary, dispatch receipts, audit queue,
   compact artifact pullback, and remote `bash -n` / smoke evidence before changing status to COMPLETE.

The controller deliberately refuses to start if `qwen_other_majorsteps6_20260718_v1` is active, if any GPU
compute process exists, if disk free is below 60 GiB, or if pinned asset receipts are absent/mismatched.

## Pending Obligations Carried Forward

Canonical registry:
`docs/experiment_plans/PENDING_2026-07-12_mandatory-experiment-gates.md`.

- **PENDING-01 Eval Parity Matrix:** deliberately deferred; this all-vLLM free-form run does not close HF-vLLM parity.
- **PENDING-02 Pure-BF16 SHS And TriGLU:** deliberately deferred; historical checkpoint dtypes remain unchanged.
- **PENDING-03 Registered SHS CausalLM Route:** deliberately deferred; SHS is outside this TriGLU-baseline grid.
