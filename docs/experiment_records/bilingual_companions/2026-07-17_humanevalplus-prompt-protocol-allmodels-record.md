# 2026-07-17 HumanEvalPlus Prompt-Protocol All-Models Record / 中英混搭版

<!-- bilingual-companion-v1 -->

> **English original / 英文原件：** [2026-07-17_humanevalplus-prompt-protocol-allmodels-record.md](../2026-07-17_humanevalplus-prompt-protocol-allmodels-record.md)
> **中文导读 / Bilingual guide：** 这份 companion 的主题是 **2026-07-17 HumanEvalPlus Prompt-Protocol All-Models Record**。下面完整保留 original report 的 metrics、tables、commands、paths、hashes 与 scientific claims，同时加入中文 narrative layer 和 bilingual section handles，方便项目负责人阅读与后续 Agent 检索。
> **Evidence contract / 证据契约：** English original 保持 immutable；如果 companion 与原件出现 wording tension，以原件中的 exact numbers、machine-readable artifacts 和 source paths 为准，companion 不得 retroactively 改写实验事实。


Date: 2026-07-17

Status: **COMPLETE.** Untuned base, TriGLU step-294, and baseline step-196 have
exact full-164 coverage under one corrected prompt protocol and task ledger.

## Contract

- Prompt: byte-identical EvalScope HumanEval+ instruction and task payload,
  without a chat template.
- Decoding: greedy, seed `20260707`, maximum 3,072 generated tokens.
- Execution: parser-v2 plus privilege-dropped no-network/no-exec sandbox, with
  a 300-second CPU limit per generated program.
- Parallelism: six independent TP=1 vLLM replicas and deterministic
  `ledger_index % 6` ownership.
- Ledger SHA-256:
  `7bb4ed06a3a4c725a9893f38e76087c9d6bf2c3caa8d0c880e061ffecc0a1baa`.
- No checkpoint, tokenizer, parser, tests, or model weights changed between
  model cells.

TriGLU used the registered `Qwen3TriGLUForCausalLM` route. Every rank produced
one valid `qwen_swiglu_triglu_side / reference_pytorch_cublas` dispatch receipt;
there was no fallback.

## 结果 / Results
| Model | Passed | Collapse loops | Syntax-valid | Cap hits | Timeouts |
|---|---:|---:|---:|---:|---:|
| Untuned base | 97/164 (59.146%) | 0 | 161 | 6 | 4 |
| TriGLU step-294 | **100/164 (60.976%)** | 0 | 159 | 6 | 3 |
| Baseline step-196 | **100/164 (60.976%)** | 0 | 164 | 3 | 3 |

Sandbox timeouts count as failures. The restricted runner reports these as
Linux `SIGXCPU` (`exit_code=-24`) inside the broader `error` status, which is
now classified explicitly by the evaluator and monitor.

## Task-Level Agreement

| Pair | Both pass | Left only | Right only | Both fail |
|---|---:|---:|---:|---:|
| TriGLU vs baseline | 91 | 9 | 9 | 55 |
| TriGLU vs untuned base | 84 | 16 | 13 | 51 |
| Baseline vs untuned base | 87 | 13 | 10 | 54 |

The TriGLU/baseline scalar tie is not behavioral equivalence. The compact
metrics preserve all exclusive-pass task IDs for subsequent row-level audit.

## Code Aggregate

The prompt-corrected CodeAvg is the equal-weight mean of corrected HumanEval+,
unchanged MBPP, and corrected LiveCodeBench.

| Model | HumanEval+ corrected | MBPP | LiveCodeBench corrected | CodeAvg corrected |
|---|---:|---:|---:|---:|
| Untuned base | 59.146% | 22.600% | 7.396% | 29.714% |
| TriGLU step-294 | 60.976% | 29.200% | 10.142% | **33.439%** |
| Baseline step-196 | 60.976% | 28.202% | 9.764% | 32.980% |

This aggregate is a separately named corrected-protocol view. The original
chat/parser HumanEval+, CodeAvg, and OOD means remain immutable so the record
does not silently mix protocols.

## Runtime Evidence

The six-way generation stage produced 83,114 tokens for untuned base, 78,179
for TriGLU, and 69,125 for baseline. Approximate aggregate generation throughput
from total tokens divided by the slowest concurrent worker's generation wall
was 3,180.7, 1,317.1, and 2,586.0 tokens/s respectively. This is an operational
six-replica estimate, not per-request latency or pure hardware parity: model
outputs differ in total length and content.

Slow sandbox tails, not model decoding, dominated end-to-end wall time. The
maximum worker review durations were 605.1 seconds for untuned base, 321.2 for
TriGLU, and 310.2 for baseline. Peak GPU memory was not instrumented in this
protocol and is not claimed.

## 解读 / Interpretation
The old chat-wrapped HumanEval+ scores primarily measured interface mismatch:
0.610% untuned, 21.951% TriGLU, and 31.097% baseline became 59.146%, 60.976%,
and 60.976% when only chat wrapping was removed. Training still changes which
tasks pass, but the enormous old between-model gap is not robust to the prompt
interface.

The corrected untuned-base score exceeds the paper's 44.5% anchor, so this run
does not establish exact paper parity. It establishes a local, paired,
base-model-compatible protocol for architecture comparison. Paper prompt,
stop, harness, and scorer details remain underdetermined.

## Evidence

- Run: `qwen3_1p7b_heplus_prompt_matrix_full164_20260717_v1`
- Compact JSON:
  `docs/experiment_records/compact_metrics/2026-07-17_humanevalplus_prompt_protocol_allmodels.json`
- Compact JSON SHA-256:
  `31894c6fcc22d4afbde211ea8e4bfada73fb9a74569d4ff82d4a33e7966c6666`
- Remote compact archive SHA-256:
  `e2cc340ab1f8a587efc83c8e323fba29df75a2f3beb603e93fe903b299a9aba9`
- Paper anchor: `https://arxiv.org/pdf/2607.01232`, Table 13.

After compact evidence was pulled and both monitors passed their presentation
audit, this experiment supplied the final pre-baseline barrier evidence. The
existing GRPO controller resumed baseline from step 196 toward step 226 through
its original `resume_mode=auto` path; no model or evaluator process was restarted
to produce the HumanEval+ results above.

## Unified Human-Readable Monitor

The recommended single entry point is now:

```bash
bash scripts/monitor_triglu_priority_to294_then_baseline196_20260715_v1.sh
```

It includes current GRPO progress and speed, a current-first consolidated
evaluation dashboard, corrected HumanEval+ diagnostics, the preserved heritage
chat-protocol table, OOD stage progress, same-step Math comparisons, data-order
receipts, GPU utilization, and disk capacity. The older OOD-specific monitor is
retained for focused diagnostics and supports an embedded mode, but it is no
longer required to obtain the complete user-facing status.

The dashboard deliberately puts corrected raw-no-chat HumanEval+ and corrected
CodeAvg first. Heritage chat/parser HumanEval+, CodeAvg, Reasoning, Language,
OOD-8, and OOD-category means remain visible in a separate labeled table. No
historical score is deleted or silently replaced.

## 继承待办 / Pending Obligations Carried Forward
Canonical registry:
`docs/experiment_plans/PENDING_2026-07-12_mandatory-experiment-gates.md`

- **PENDING-01 Eval Parity Matrix:** still pending. This experiment fixes one
  prompt-interface mismatch but does not complete the broader HF/vLLM matrix.
- **PENDING-02 Pure-BF16 SHS And TriGLU:** deferred and unchanged.
- **PENDING-03 Registered SHS CausalLM Route:** deferred and unchanged.
