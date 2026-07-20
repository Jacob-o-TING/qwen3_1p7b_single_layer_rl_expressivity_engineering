# GPQA-Diamond-Freeform-126 Manual Agent Audit Record

Date: 2026-07-18

Status: **MANUAL AUDIT COMPLETE / VALIDATED**

## Protocol Change / 协议变更

项目负责人批准将 primary evaluator 从 `Qwen3-4B no-thinking matcher` 改为一只 Sub Agent 的 complete row-by-row
manual audit。Answer matching paper 确实验证过 Qwen3-4B matcher，但具体 matcher 并非原始 GPQA-Diamond
强制项；因此新 protocol 使用独立 identity，不重写旧 route 的 provenance。

本记录将在 generation launch、audit package closeout、Sub Agent progress 和 final verdict gates 完成后持续
追加 evidence。Large responses、audit rows、verdicts 和 checkpoints 位于 gitignored `runs/` / `audit_inputs/`；
Git 只保存 source、tests、config、plan、record 和 compact final metrics。

## Pre-launch Verification / 启动前验证

Local focused verification passed：manual-audit suite `5/5`、unified-monitor suite `6/6`、Python
`py_compile` 与 `git diff --check` 全部通过。随后只包含本 protocol source/docs/config/tests 的 bundle
被传至 6x RTX 5090 instance；local/remote SHA-256 均为
`cd4cac955eca4ee5a43b52d21c79e71072dc8ae5988381607c28d789d038207a`。

Remote Linux verification 同样通过：controller、dedicated monitor 与 unified monitor 均通过 `bash -n`，
两个 package helpers 通过 `py_compile`，focused suites 再次为 `5/5 + 6/6`。Remote preflight observation
为 six GPUs idle、Other/OOD `WAVE_COMPLETE` present、data disk `124G` free。Config 中 `60G` 是 refusal
floor，不是 allocation；本 wave 的 persistent generation/audit outputs 预计远低于该数字。

## Launch Repair / 启动修复

First launch 在 `triglu_step158` 的 vLLM compiled dummy forward fail-fast。Trace 明确落在
`write_dispatch_receipt_once()` 内的 `Path.mkdir()`：TorchDynamo 不允许 compiled graph 执行 filesystem
side effect；这不是 model math、checkpoint 或 dataset failure。Repair 将 registered TriGLU vLLM wrapper
的真实 dispatch receipt 提前到 `load_weights()` 结束后、dummy compilation 之前写入，并标记 wrapper 已记录；
compiled `forward()` 因此只走 receipt helper 的 early return。该修复不改变 logits、weights、sampling、
`enforce_eager` 或 six-replica protocol，actor/training 的原有 forward-time receipt path 仍保留。

First repair 后 receipt 已可在 pre-compile phase 写入，但 call site 仍无条件进入 helper；Dynamo 因此继续
inspect helper 所在的 frozen `os` module，并在 vLLM compiler source lookup 报
`FileNotFoundError: <frozen os>`。Second repair 在 TriGLU wrapper call site 先检查已记录 flag：registered vLLM
route 完全不把 helper 纳入 graph，未预写 receipt 的 ordinary actor/HF route 仍按原语义在首次 forward
记录。两次 failed attempts 均未产生 generation shard 或 checkpoint mutation。

Second repair 后 architecture compile 与 generation 已真实通过：ranks `0/2/3/5` 各自完成 21 rows 并写出
`SHARD_COMPLETE`。Ranks `1/4` 的 independent engines 则在 TorchInductor cache atomic write/rename 处失败；
六个 TP=1 processes 都将自己标为 vLLM `rank_0_0`，因此 concurrent fresh compilation 争用了同一个
`~/.cache/vllm/torch_compile_cache/...` namespace。Controller repair 为每个 physical GPU 配置独立
`VLLM_CACHE_ROOT` 与 `TORCHINDUCTOR_CACHE_DIR`。Resume 保留四个已闭合 shards，只重跑 missing ranks；
这同样不改变 model/sampling protocol。

项目负责人指出并经 repo-wide source audit 确认：established SHS production-length throughput、TriGLU onboarding、
SHS rollout/GRPO smoke、6-GPU parallel eval、HumanEval+ matrix 与 2x5090 prelaunch gate 均使用
`enforce_eager=true`。此前把 non-eager 归因于 throughput experiments 是错误 interpretation；只有本轮未成功
的 GPQA-Freeform config 及其 clone 来源的 Qwen3-4B matcher draft 写成 `false`。

Official protocol 因此 correction 为 `enforce_eager=true` across all ten cells：仍保留 vLLM continuous
batching、six independent TP=1 replicas 与 exact 21-row shards，但不把 compile/CUDA-graph behavior 纳入
architecture comparison。Compiled attempt 已完成的 84 rows 被 archive 为 failure evidence，不与 eager
official cell 混合；`triglu_step158` 从 clean eager shards 重新开始。Compile receipt/cache repairs 保留供未来
独立 compile benchmark 使用。

Clean eager restart 的 first paired gate 已通过：`triglu_step158` 与 `baseline_step158` 均为 exact
`126/126`，question IDs 和 per-request seeds 逐行一致，ledger SHA-256 均为
`849d6bcbaad0619c8ade3eb392a59c9203466615da2675bfac02dcaf1c937901`。Observed wall speed 约
`1.12 min/cell`；controller 已自主进入 step 196，未中断 queue。

## Generation Closeout / 生成闭环

Clean eager wave completed all ten cells and stopped at `AWAITING_MANUAL_AUDIT`。Every cell closed exact
`126/126`; aggregate payload is 1,260 responses。Final generation diagnostics：

| Step | Variant | Cap hits | Missing answer tags | Generated tokens |
| ---: | --- | ---: | ---: | ---: |
| 158 | TriGLU | 4 | 82 | 81,333 |
| 158 | baseline | 14 | 89 | 100,006 |
| 196 | TriGLU | 8 | 85 | 96,259 |
| 196 | baseline | 10 | 82 | 89,210 |
| 226 | TriGLU | 7 | 83 | 88,494 |
| 226 | baseline | 13 | 87 | 100,563 |
| 256 | TriGLU | 6 | 83 | 88,251 |
| 256 | baseline | 15 | 83 | 104,589 |
| 294 | TriGLU | 7 | 87 | 89,049 |
| 294 | baseline | 12 | 88 | 96,952 |

Agent-readable archive is `683,306` bytes with SHA-256
`be389f259b5c8c830062dc77182f5aa9232cfc756b244771996738d03e9f9241`。It was pulled to local
gitignored `audit_inputs/gpqa_diamond_freeform126_manual_agent_audit_20260718_v1/`; local SHA matches remote。
Local package validation passed：1,260 unique audit IDs、126 questions、ten responses/question、all ten source-cell
hashes present。Generation heartbeat was paused after package closeout。

One and only one Sub Agent (`019f7649-141c-7a12-a7b4-65098e567297`, nickname `Mendel`) was assigned the
complete row-by-row audit。It did not spawn/delegate any additional agent and has now closed all `1,260`
verdict gates。

## Manual Audit Closeout / 人工审计闭环

audit sub-agent 已完整阅读 `126 questions x 10 checkpoint cells = 1,260 responses`。Merger validation clean pass：

- `verdict_count=1260`，`unique_verdict_ids=1260`，canonical/verdict ID sets 完全一致；
- 每题恰好 `10` rows，全部 `response_read_completely=true`；
- `AUDIT_COMPLETE`、`final_summary.json`、`review_uncertain.jsonl` 与
  `review_disagreements.jsonl` 均存在；
- schema、enums、required fields 全部合法；
- semantic totals 为 `69 correct / 1178 incorrect / 13 uncertain`，另有 `80` disagreement rows。

Strict accuracy 固定使用全部 `126` questions 作为 denominator；`uncertain` 不作为 correct。Decided-only
accuracy 仅作为 auxiliary diagnostic，不替代 strict score。

| Cell | Correct | Incorrect | Uncertain | Strict accuracy |
| --- | ---: | ---: | ---: | ---: |
| baseline step158 | 8 | 117 | 1 | 6.349% |
| baseline step196 | 6 | 119 | 1 | 4.762% |
| baseline step226 | 4 | 120 | 2 | 3.175% |
| baseline step256 | 5 | 120 | 1 | 3.968% |
| baseline step294 | 6 | 118 | 2 | 4.762% |
| TriGLU step158 | 7 | 118 | 1 | 5.556% |
| TriGLU step196 | 6 | 119 | 1 | 4.762% |
| TriGLU step226 | 7 | 117 | 2 | 5.556% |
| TriGLU step256 | 10 | 115 | 1 | 7.937% |
| TriGLU step294 | 10 | 115 | 1 | 7.937% |

Across checkpoints，TriGLU strict mean 为 `6.349%`、population std 为 `1.328 pp`；baseline strict mean
为 `4.603%`、population std 为 `1.053 pp`。这些是五个 checkpoint cells 的 descriptive statistics，
不是 independent-seed uncertainty estimate。

## Variant Aggregate / 架构聚合

`Unique correct questions` 会把同一 architecture 在多个 checkpoints 对同一道题的成功合并；
`correct checkpoint runs` 则逐 cell 计数，因此两个 quantity 回答不同问题。

| Model | Unique correct questions | Correct checkpoint runs |
| --- | ---: | ---: |
| TriGLU | 18 | 40 |
| Baseline | 12 | 29 |

TriGLU correct question indices：`0, 7, 9, 27, 29, 38, 47, 50, 54, 55, 71, 78, 79, 94, 99, 104, 109, 118`。
Baseline correct question indices：`27, 29, 35, 38, 50, 55, 62, 78, 94, 104, 107, 109`。

## Failure Modes / 失败模式

| Failure category | Rows |
| --- | ---: |
| reasoning_error | 702 |
| factual_error | 208 |
| insufficient_or_vague | 151 |
| truncated | 92 |
| malformed_or_no_answer | 22 |
| reference_or_question_ambiguity | 11 |
| contradictory_answer | 5 |
| correct | 69 |

Dominant observed mode 是 `reasoning_error`，而非纯粹 `factual_error`。不过该 taxonomy 是 manual audit
classification，不应把 row counts 直接解释为 mutually independent latent causes；`80` disagreement rows
与 `13` uncertain rows 继续保留 review provenance。

## Cap-Hit Interaction / 截断交互

Canonical package 中共有 `96` cap-hit rows；全部已经完整阅读，并在 original-response protocol 下判为
`incorrect`。这说明当前 strict score 没有把未审计截断行静默丢弃。Future cap-repair 若生成 continuation
replacement，只对这 `96` 条 replacement outputs 做 targeted re-audit；original verdicts 不覆盖，
original/repaired provenance 与 score columns 必须并存。

## Durable Evidence / 持久证据

Gitignored canonical package：
`audit_inputs/gpqa_diamond_freeform126_manual_agent_audit_20260718_v1/`。

Tracked compact summary：
`docs/experiment_records/compact_metrics/2026-07-18_qwen3_1p7b_gpqa_diamond_freeform126_manual_agent_audit.json`。

Source hashes：

- `final_summary.json`: `a5c1c9d0484fc84f1a8d152bfd14c67e8dc150e42ed35ea4a1f96c69c0ae2154`;
- `verdicts.jsonl`: `2b71fc2da07cd8b1c699c15cb56b4ef17b3375a65b3811d11007ccda710772ea`;
- `audit_rows.jsonl`: `83a80b6ff45d5a0ca52b3afc8e9a7ee85354c305cffc054fddffa00a7dfbd544`;
- `review_uncertain.jsonl`: `bef49249e2ecd3d716995f740d4cd4c832ae95726ab5ea0c638bd62ad9f9acfa`;
- `review_disagreements.jsonl`: `6c3ca3e337aa6afb96eea8e3ac05aadb571dbfa4bc9071867f10975783313cee`。

## Pending Obligations Carried Forward

Canonical registry: `docs/experiment_plans/PENDING_2026-07-12_mandatory-experiment-gates.md`

- PENDING-01 Eval Parity Matrix: deferred。
- PENDING-02 Pure-BF16 SHS And TriGLU: deferred。
- PENDING-03 Registered SHS CausalLM Route: deferred。
