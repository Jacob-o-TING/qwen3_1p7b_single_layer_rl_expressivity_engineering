# Qwen3-1.7B GPQA-Diamond-Freeform-126 Manual Agent Audit Plan

Date: 2026-07-18

Status: **COMPLETED / VALIDATED**

## Owner Decision / 项目负责人决定

项目负责人明确选择由一只 audit sub-agent 逐条 manual audit，而不是把 `Qwen3-4B` 的 semantic matcher decision 当作
primary ground truth。原 `qwen3_4b_match` route 作为 immutable alternate protocol 保留；它的 failed asset-download
attempt 不会被覆盖，也不会与本计划共享 run identity。

Approved identity:

- run ID: `qwen3_1p7b_gpqa_diamond_freeform126_greedy3072_manual_agent_audit_6x5090_steps158_196_226_256_294_20260718_v1`;
- screen: `qwen_gpqa_free126_manualaudit_gen6_20260718_v1`;
- config: `configs/eval/qwen3_1p7b_gpqa_diamond_freeform126_manual_agent_audit_majorsteps_6x5090_20260718_v1.yaml`;
- remote result root: `runs/freeform_eval/<RUN_ID>/`;
- local audit package: `audit_inputs/gpqa_diamond_freeform126_manual_agent_audit_20260718_v1/`.

## Scientific Contract

Generation uses the same pinned 126-row ledger for every cell、seed `20260707`、greedy temperature `0`、
`max_tokens=3072`、six independent TP=1 vLLM replicas and exact `[21,21,21,21,21,21]` sharding。Cell order
is TriGLU then baseline at global steps `158,196,226,256,294`; every pair sees identical ordered questions and
request seeds。

Owner protocol correction：all ten official cells use vLLM `enforce_eager=true`。This matches the established
SHS/TriGLU rollout and parallel-eval workflow、preserves continuous batching and six-way GPU parallelism, and
keeps TorchDynamo/CUDA-graph compilation outside this scientific comparison。No compiled-attempt shard may be
mixed into the eager primary results。

No LLM matcher is loaded。After all ten cells close exact `126/126`, the controller creates canonical
`audit_rows.jsonl` with 1,260 unique IDs and 126 question-centric Markdown chunks。Each chunk presents one
question、one reference answer and all ten architecture/checkpoint responses, allowing consistent across-checkpoint
judgment。

## Manual Audit Procedure

One Sub Agent owns all 126 chunks and must read every response completely。For every row it appends exactly one
verdict with `correct | incorrect | uncertain`、answer-tag status、failure category、confidence and concise
row-specific evidence。The agent checkpoints after every row and updates `progress.json` after every question。

Heuristic parsing、normalized exact match and `<answer>` extraction are auxiliary metadata only；they may never
substitute for reading。Ambiguous reference equivalence、domain uncertainty or underspecified questions remain
`uncertain` for a second pass rather than being forced into correct/incorrect。

Completion requires exactly 1,260 unique verdicts、10 verdicts per question、126 completed question IDs、valid
enums、no null required fields、separate uncertain/disagreement review files and a deterministic per-cell summary。
Primary official MCQ GPQA-Diamond remains a separate protocol and is never hard-averaged with this score。

## Autonomous Execution

The remote controller performs dataset-only asset preparation, validates all ten source checkpoints, exports one
cell at a time, runs six generation shards concurrently, exact-merges and prunes only disposable exports。After
generation it builds the audit package、archives it and stops at `AWAITING_MANUAL_AUDIT`; it never fabricates a
score or shuts down the instance。

## Completion / 完成状态

本计划现已完整 closeout。Canonical package 包含 `126` questions、`1,260` responses 与 `1,260`
unique verdict IDs；每题恰好 `10` rows。`AUDIT_COMPLETE`、`final_summary.json`、
`review_uncertain.jsonl` 与 `review_disagreements.jsonl` 均已生成，canonical/verdict ID sets 完全一致，
schema、enums、required fields 与 `response_read_completely=true` gates 全部通过。

Primary result 使用 strict denominator `126` per cell；`uncertain` 不计入 correct。TriGLU across-checkpoint
strict mean 为 `6.349%`、population std 为 `1.328 pp`；baseline mean 为 `4.603%`、population std 为
`1.053 pp`。完整 per-cell 与 failure-mode evidence 记录于
`docs/experiment_records/2026-07-18_qwen3-1p7b-gpqa-diamond-freeform126-manual-agent-audit-record.md`。

原始 package 中 `96` 条 cap-hit responses 均已完整审计且 verdict 为 `incorrect`。这不授权丢弃原 verdict；
future cap continuation 若生成 replacement outputs，只对这 `96` 条 replacement rows 做 targeted manual
re-audit，并保留 original/repaired 双份 provenance。

## Pending Obligations Carried Forward

Canonical registry: `docs/experiment_plans/PENDING_2026-07-12_mandatory-experiment-gates.md`

- PENDING-01 Eval Parity Matrix: deferred; this named freeform audit does not close backend parity。
- PENDING-02 Pure-BF16 SHS And TriGLU: deferred; historical mixed-precision checkpoints remain inputs。
- PENDING-03 Registered SHS CausalLM Route: deferred; SHS is outside this TriGLU/baseline audit。
