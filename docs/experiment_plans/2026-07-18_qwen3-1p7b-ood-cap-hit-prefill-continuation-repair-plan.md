# Qwen3-1.7B OOD Cap-Hit Prefill-Continuation Repair Plan

Date: 2026-07-18

Status: **PLANNED / NOT LAUNCHED**

Methodological ancestor:
`docs/experiment_plans/2026-07-11_cap-hit-continuation-and-extraction-diagnostics-plan.md`.

## Owner Decision / 项目负责人决定

本 plan 只定义未来如何 repair 已有 OOD `finish_reason=length` rows，不授权当前启动
generation、rerun、grading 或 GPU wave。本次已完成的事情只是 existing evidence pullback
与 documentation closeout。

`8192` 是 adaptive continuation 的 **per-response total ceiling**，不是要求每条 row 都生成
8192 tokens。Model 在 3400、4800 或任何长度生成 EOS 时立即结束；只有仍然没有 EOS
的 long-tail rows 才会真正撞上 8192。本 protocol 不设 `min_tokens=8192`。

## Scientific Scope / 科学边界

只包含 TriGLU 与 whole-layer baseline 在 global steps `158/196/226/256/294` 的 OOD
results，以及同样 ten-cell 的 GPQA-Diamond-Freeform generation。

下列 Math benchmarks 属于 paper-comparable `3072` protocol，必须保留原样，**不进入
repair queue**：

- `AMC Avg@32`;
- `AMC greedy pass@1`;
- `GSM8K`;
- `MATH-500`;
- `OlympiadBench`.

Heritage results 不丢弃，但不与 repaired results 静默混合。HumanEval+ 已有 raw no-chat
replacement，因此后续只使用 corrected protocol 中的 41 条 cap hits；旧 chat result
仍作 provenance control 保留。

## Existing Evidence / 已有证据

| OOD benchmark | Rows at the 3072 cap | Planned treatment |
| --- | ---: | --- |
| HumanEval+ corrected raw no-chat | 41 | targeted prefill continuation |
| MBPP heritage | 4,696 / 5,000 | protocol matrix, then full 5,000-row rerun |
| LiveCodeBench corrected | 1,027 | targeted prefill continuation |
| GPQA-Diamond MCQ | 195 | targeted prefill continuation |
| MMLU-Pro | 5,686 | targeted prefill continuation |
| C-Eval | 9,195 / 13,460 | protocol matrix, then full 13,460-row rerun |
| IFEval | 2,115 | targeted continuation plus passive loop classification |
| MGSM | 2,369 | targeted prefill continuation |
| GPQA-Diamond-Freeform | 96 / 1,260 | original audit complete: 96/96 incorrect; targeted continuation, then replacement-only manual re-audit |

OOD subtotal excluding GPQA-Freeform 为 `25,324` cap-hit rows；加上 Freeform 后 targeted set
为 `25,420`。由于 MBPP 与 C-Eval 的 cap-hit rates 高到不适合仅修复 truncated subset，
将它们替换为 full new-protocol rerun 后，recommended production workload 为 `29,989`
generations。

GPQA-Diamond-Freeform original manual audit 已完成全部 `1,260` rows；其中 `96` cap-hit rows 全部在
original-response protocol 下判为 `incorrect`。Future repair 不回写这些 verdicts，而是生成独立
replacement rows 并做 targeted re-audit。

## Planned Continuation Contract / 预计协议

1. Eligibility 只由已有 receipt 决定：`finish_reason == length` 或经校验的
   `generated_tokens >= 3072`；不读 correctness 或 reference answer。
2. Prefill exact existing prompt plus assistant prefix，然后只 decode 增量 tokens。Original 3072-token
   output 不改写。
3. First ceiling 为 total length `8192`；EOS 随时 early-stop。
4. 只有仍然 coherent 且没有 EOS 的 residual rows 才进入 total `16384`。
5. `32768` 仅用于极少数经人工确认仍在有效推理的 exception rows；不是 default。
6. 只做 passive exact-cycle / repetition classification；不加 `repetition_penalty`，因为 penalty
   会改变被测 policy。
7. 保留 pre-repair 与 post-repair 两份 score/report，禁止 in-place overwrite。

## Protocol Repair Before Length Repair

MBPP `4,696/5,000` 与 C-Eval `9,195/13,460` 的 cap rates 更像 prompt/protocol-induced
runaway reasoning，不像少数难题需要更长 reasoning。因此必须先做 small protocol matrix，比较
heritage chat 与候选 raw/concise non-chat protocol。一旦选择新 protocol，就要对该 benchmark
的 ten-cell full denominator 重跑，不能将 new-protocol cap rows 与 old-protocol non-cap rows
拼成一个 score。

## Runtime Envelope / 时间边界

ETA 假设 six independent RTX 5090 `TP=1` vLLM replicas、continuous batching、existing-prefix
prefill，并以已观察 full OOD wave 约 `7.14 h` 作 sanity anchor。

- 若 targeted rows 平均只需 `+512` tokens：continuation 约 `0.5-1 h`;
- 平均 `+1024`：约 `1-1.5 h`;
- 平均 `+2048`：约 `2-3 h`;
- 若所有 targeted rows 都再撞上 8192：约 `6-8 h`，这是 pessimistic tail；
- 加上 MBPP/C-Eval protocol matrix、full rerun、merge/grading：realistic total `2.5-4.5 h`,
  conservative `5-7 h`, pathological loop-heavy upper envelope `8-10 h`.

以上是 planning estimate，不是 launch receipt。

## Acceptance Gates / 验收门

- exact ten-cell identity and checkpoint ledger;
- no Math benchmark enters the repair manifest;
- no non-cap row enters targeted continuation;
- original prefix hash and source receipt provenance preserved;
- EOS/length/loop classifications reported separately;
- old score reproduced before applying any replacement;
- replacement-only rescoring and explicit pre/post delta;
- GPQA-Freeform 96 replacement rows receive targeted manual re-audit;
- compact outputs pulled locally with SHA-256 verification before closeout.

## Pending Obligations Carried Forward

Canonical registry:
`docs/experiment_plans/PENDING_2026-07-12_mandatory-experiment-gates.md`.

- **PENDING-01 Eval Parity Matrix:** deferred; this plan does not close backend/evaluator parity.
- **PENDING-02 Pure-BF16 SHS And TriGLU:** deferred; no architecture dtype change is in scope.
- **PENDING-03 Registered SHS CausalLM Route:** deferred; SHS is outside this ten-cell grid.
