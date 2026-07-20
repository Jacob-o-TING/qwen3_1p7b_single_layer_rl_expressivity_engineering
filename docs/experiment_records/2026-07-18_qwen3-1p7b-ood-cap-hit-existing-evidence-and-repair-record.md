# Qwen3-1.7B OOD Cap-Hit Existing Evidence And Repair Record

Date: 2026-07-18

Status: **EXISTING EVIDENCE PULLED / REPAIR NOT LAUNCHED**

Companion plan:
`docs/experiment_plans/2026-07-18_qwen3-1p7b-ood-cap-hit-prefill-continuation-repair-plan.md`.

## Closeout Boundary / 本轮边界

本轮只做 remote existing-result inspection、compact pullback、SHA-256 verification 与 documentation。
没有启动 cap continuation、没有 rerun 任何 benchmark、没有修改 model/evaluator outputs、
没有使用 GPU。一个越界创建的 executable inventory draft 在进入 generation 前即被撤回；
local/remote script 与 temporary output directory 已删除，remote `GPU_PROCESSES=0`。

## Pullback Receipt / 拉回回执

Remote existing compact evidence 已拉到：

`docs/experiment_records/compact_metrics/2026-07-18_qwen3_1p7b_existing_ood_cap_evidence/`

Bundle 包含：

- ten OOD cell `summary.json` files;
- ten corrected raw no-chat HumanEval+ protocol summaries;
- ten GPQA-Diamond-Freeform generation summaries;
- OOD final dashboard, timings, import/export receipts;
- GPQA-Freeform export receipts;
- per-file `SHA256SUMS`.

Remote transfer archive 为 `28K`，SHA-256：
`9738835f4ff95ebcd68a9617f902f9aa1f46cf378da486275ce90d30447e933f`。Local unpacked payload
为 70 evidence files，约 `240 KB`；除 checksum file 本身外的所有 payload hashes 已逐文件验证通过。
GPQA-Freeform full 1,260-response agent-readable archive 此前已拉至 gitignored
`audit_inputs/gpqa_diamond_freeform126_manual_agent_audit_20260718_v1/`，本次不重复提交 raw traces。

## Existing OOD Scores / 已有分数

下表使用拉回的 current cell summaries；HumanEval+ 列专门使用 corrected raw no-chat protocol，
不是 heritage chat score。其余列保留已有 evaluator protocol，未做 cap repair。

| Cell | HE+ no-chat | MBPP | LCB | GPQA-Diamond | MMLU-Pro | C-Eval | IFEval | MGSM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TriGLU-158 | 59.15 | 30.40 | 10.99 | 26.77 | 33.99 | 46.58 | 28.51 | 59.42 |
| baseline-158 | 59.76 | 29.60 | 9.76 | 28.28 | 34.43 | 45.39 | 26.42 | 58.91 |
| TriGLU-196 | 61.59 | 28.60 | 10.81 | 26.77 | 34.26 | 47.40 | 26.83 | 57.56 |
| baseline-196 | 60.98 | 28.20 | 9.76 | 22.73 | 34.53 | 45.69 | 27.05 | 59.02 |
| TriGLU-226 | 60.98 | 30.60 | 10.90 | 25.25 | 34.10 | 47.55 | 26.62 | 60.15 |
| baseline-226 | 60.37 | 29.60 | 10.05 | 25.25 | 34.03 | 46.28 | 26.00 | 58.73 |
| TriGLU-256 | 61.59 | 27.60 | 11.38 | 24.24 | 34.35 | 48.22 | 27.67 | 59.57 |
| baseline-256 | 57.93 | 28.80 | 9.67 | 27.78 | 34.47 | 46.73 | 27.88 | 58.84 |
| TriGLU-294 | 60.98 | 29.20 | 10.14 | **29.80** | 32.77 | 47.99 | 27.46 | 46.00 |
| baseline-294 | 58.54 | 27.40 | 9.67 | 28.79 | 34.14 | 46.81 | 26.42 | 59.31 |

这些分数可以用来追踪 architecture/checkpoint trend，但不能将 MBPP、C-Eval 的 heritage
protocol 与 future non-chat repair 直接当成同一 protocol 的 longitudinal delta。Historical OOD/category
hard averages 也不作为本记录的 primary decision metric。

## Exact Cap-Hit Scope / 精确范围

Current ten-checkpoint OOD evidence 中，所有下表 cap hits 都是同一个 `3072` ceiling；
`4,696` 与 `9,195` 是 row counts，不是另外的 token caps。

| Benchmark | Denominator | 3072 cap hits | Rate | Evidence interpretation |
| --- | ---: | ---: | ---: | --- |
| HumanEval+ corrected no-chat | 1,640 | 41 | 2.50% | current repaired prompt protocol |
| MBPP heritage | 5,000 | 4,696 | 93.92% | protocol-level contamination likely |
| LiveCodeBench corrected | 10,550 | 1,027 | 9.73% | material long-tail |
| GPQA-Diamond MCQ | 1,980 | 195 | 9.85% | material long-tail |
| MMLU-Pro | 120,320 | 5,686 | 4.73% | smaller rate, large absolute count |
| C-Eval | 13,460 | 9,195 | 68.31% | protocol-level contamination likely |
| IFEval | 5,410 | 2,115 | 39.09% | length plus possible loop behavior |
| MGSM | 27,500 | 2,369 | 8.61% | material long-tail |
| **OOD subtotal** | **185,860** | **25,324** | **13.63%** | excludes paper-locked Math |
| GPQA-Diamond-Freeform | 1,260 | 96 | 7.62% | replacement rows need re-audit |
| **Included total** | **187,120** | **25,420** | **13.59%** | planning selection set |

Corrected HumanEval+ per-cell cap hits 为 `6,2,3,3,3,4,6,3,6,5` （按表格 cell 顺序），合计
41。GPQA-Freeform per-cell cap hits 为 `4,14,8,10,7,13,6,15,7,12`，合计 96。

## Interpretation / 解读

`8192` 不表示所有 selected rows 都会到 8192。它是 first-pass total ceiling；一条 row
只要在 3072 之后生成 EOS，就在实际长度结束。因此 runtime 主要由 additional-token
distribution 决定，而不是 `25,420 x (8192-3072)` 的必然 full allocation。

MBPP 和 C-Eval 的 cap rates 过高，不应直接归因于题目真需要超长 reasoning。更可能的
explanation 是 heritage chat prompt 导致 continued thinking/repetition/未及时输出最终格式。所以
它们需要 protocol matrix 先行；若改 protocol，必须 full rerun 全部 `5,000 + 13,460`
rows，不做 mixed-protocol score。

## Planned Workload, Not Execution / 计划量而非已执行量

若简单对所有 cap hits 做 targeted continuation，selection 为 `25,420`。将 MBPP 与 C-Eval
替换成 full new-protocol reruns 后，protocol-aware workload 为：

```text
25,420 - 4,696 - 9,195 + 5,000 + 13,460 = 29,989 generations
```

该 workload 仍是 future plan，本轮未 launch。基于 six RTX 5090 TP=1 replicas、vLLM continuous
batching、prefill existing prefix 与 full OOD `~7.14 h` anchor，当前 realistic estimate 为
`2.5-4.5 h`，conservative `5-7 h`，若大量 rows 持续 loop 到 8192/16384，upper envelope
为 `8-10 h`。

## GPQA Manual Audit Interaction

The 1,260-row manual audit 已按项目负责人决定完整 closeout：`69 correct / 1178 incorrect / 13 uncertain`。
其中 96 条 cap-hit original responses 全部完成阅读并判为 `incorrect`，没有因 truncation 丢弃 verdict。
Future continuation 完成后，只对 96 条 replacement outputs 做 targeted re-audit，然后保留
original/repaired 两份 verdict provenance。Canonical audit details 见
`docs/experiment_records/2026-07-18_qwen3-1p7b-gpqa-diamond-freeform126-manual-agent-audit-record.md`。

## Pending Obligations Carried Forward

Canonical registry:
`docs/experiment_plans/PENDING_2026-07-12_mandatory-experiment-gates.md`.

- **PENDING-01 Eval Parity Matrix:** remains pending and is not closed by this evidence pullback.
- **PENDING-02 Pure-BF16 SHS And TriGLU:** remains pending; no dtype path changed.
- **PENDING-03 Registered SHS CausalLM Route:** remains pending; SHS is out of scope.
