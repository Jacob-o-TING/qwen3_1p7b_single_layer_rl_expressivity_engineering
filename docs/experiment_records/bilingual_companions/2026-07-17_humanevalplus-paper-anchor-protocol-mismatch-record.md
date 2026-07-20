# 2026-07-17 HumanEvalPlus Paper-Anchor Protocol-Mismatch Record / 中英混搭版

<!-- bilingual-companion-v1 -->

> **English original / 英文原件：** [2026-07-17_humanevalplus-paper-anchor-protocol-mismatch-record.md](../2026-07-17_humanevalplus-paper-anchor-protocol-mismatch-record.md)
> **中文导读 / Bilingual guide：** 这份 companion 的主题是 **2026-07-17 HumanEvalPlus Paper-Anchor Protocol-Mismatch Record**。下面完整保留 original report 的 metrics、tables、commands、paths、hashes 与 scientific claims，同时加入中文 narrative layer 和 bilingual section handles，方便项目负责人阅读与后续 Agent 检索。
> **Evidence contract / 证据契约：** English original 保持 immutable；如果 companion 与原件出现 wording tension，以原件中的 exact numbers、machine-readable artifacts 和 source paths 为准，companion 不得 retroactively 改写实验事实。


Date: 2026-07-17

Status: **PROMPT-MISMATCH HYPOTHESIS CONFIRMED AT FULL COVERAGE.** Full-164
untuned-base, TriGLU step-294, and baseline step-196 corrected-protocol runs are
complete. Exact paper parity is not claimed.

## 发现 / Finding
The untuned Qwen3-1.7B-Base HumanEval+ result is execution-correct for its
frozen generated outputs but is not a credible paper-comparable capability
estimate. The dominant failure is generation collapse under the current prompt
interface, not a parser miss. The available cross-benchmark evidence points to
prompt/harness protocol mismatch more strongly than checkpoint corruption.

## Paper And Local Anchors

The paper's Table 13 reports the following Qwen3-1.7B-Base code results. The
paper does not disclose enough evaluator detail to reconstruct its exact
HumanEval+ prompt and decoding interface.

| Untuned-base code benchmark | Paper | Local current protocol | Delta |
|---|---:|---:|---:|
| HumanEval+ | 44.5% | 0.610% (1/164) | -43.890 pp |
| MBPP | 52.9% | 22.600% (113/500) | -30.300 pp |
| LiveCodeBench | 7.4% | 7.396% (78/1055) | -0.004 pp |
| Equal-weight Code mean | 34.9% | 10.202% | -24.698 pp |

Paper source: `https://arxiv.org/pdf/2607.01232`, Table 13.

LiveCodeBench matching the paper anchor to rounding precision is strong
evidence that the model files and tokenizer are not globally broken. It does
not prove that HumanEval+ uses the same prompt or scorer as the paper.

## Current Local Protocol

The local evaluator's common generation path renders every conversation with:

```python
tokenizer.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=True,
)
```

The Hugging Face path applies the same chat template during tokenization. For
HumanEval+, EvalScope supplies a zero-shot user instruction requesting only the
function code. Consequently the untuned base model is being evaluated through
an instruction-style chat interface, even though it is not an instruction-
tuned checkpoint.

This path is internally coherent and shared by the compared models. It remains
a valid local all-model protocol, but it is not known to match the paper's
base-model evaluation interface.

## Frozen-Output Evidence

The parser-v2 plus repaired sandbox review preserved all 164 untuned-base
predictions and still scored exactly `1/164 = 0.6098%`.

- 153/164 outputs contained Hebrew and no recoverable Python declaration.
- 139/164 repeated one Hebrew answer-label token for 1,024 lines.
- 10/164 repeated the English fragment `umably` for 1,536 lines.
- Only one output contained a fenced implementation and passed.
- Parser-v2 changed only two rows and recovered no additional pass.
- The prediction JSONL omitted `finish_reason`, so cap-versus-EOS attribution
  is not claimed for every row.

These outputs cannot be repaired honestly by adding extraction heuristics. The
model must be regenerated under controlled prompt variants.

## Same-Protocol Model Results

These values remain useful for internal architecture comparison under the
current vLLM greedy chat protocol:

| Model | HumanEval+ | MBPP | LiveCodeBench | Code mean |
|---|---:|---:|---:|---:|
| Untuned base | 0.610% | 22.600% | 7.396% | 10.202% |
| Baseline step-196 | 31.097% | 28.202% | 9.764% | 23.021% |
| TriGLU step-294 | 21.951% | 29.200% | 10.142% | 20.431% |

The baseline and TriGLU rows must not be compared directly with the paper until
the prompt-protocol diagnostic establishes a plausible base-model interface.

## 解读 / Interpretation
The evidence supports the following ordered hypotheses:

1. **Most likely:** the zero-shot instruction plus Qwen chat template is a poor
   interface for the untuned base model on HumanEval+.
2. **Possible:** stop-token, completion-boundary, or harness differences amplify
   the prompt mismatch.
3. **Less likely as the sole cause:** corrupted or wrong model/tokenizer files,
   because LiveCodeBench reproduces the paper's untuned-base anchor closely.
4. **Rejected for the frozen outputs:** a parser-only explanation; almost all
   failed outputs contain no implementation to extract.

The proposed controlled follow-up is documented in
`docs/experiment_plans/2026-07-17_qwen3-1p7b-humanevalplus-prompt-protocol-matrix-plan.md`.
It compares the current chat-wrapped instruction, the same instruction without
a chat template, and canonical raw function completion.

## Controlled 32-Task Prompt Matrix

The canary used one frozen SHA-selected task ledger, the exact untuned-base
checkpoint, greedy decoding, seed `20260707`, a 3,072-token cap, parser-v2, and
the repaired privilege-dropped sandbox in all cells. Only prompt rendering
changed.

| Prompt cell | Passed | Collapse loops | Cap hits |
|---|---:|---:|---:|
| Current EvalScope instruction plus Qwen chat template | 0/32 | 32/32 | 32/32 |
| Byte-identical EvalScope instruction without chat template | **21/32** | **0/32** | 1/32 |
| Canonical function completion without chat template | 6/32 | 0/32 | 16/32 |

Removing only the chat template while retaining the EvalScope instruction
changed the result by `+65.625` percentage points and removed all 32 observed
collapse loops. Canonical raw completion also repaired generation collapse but
was materially weaker than the raw instruction. This isolates the dominant
failure to the chat-wrapped untuned-base interface rather than to the model
weights, HumanEval+ tests, parser, or sandbox.

The preregistered escalation gate passed (`+21` tasks and `-32` collapse loops,
against required minima of `+3` and `-3`). A full-164 untuned-base run of the
winning raw-instruction cell is therefore authorized. The canary percentage is
not a final benchmark score and must not replace the 164-task result.

## Full-164 Corrected Untuned-Base Result

The six deterministic shards merged with exact `164/164` identity coverage and
task-ledger SHA-256
`7bb4ed06a3a4c725a9893f38e76087c9d6bf2c3caa8d0c880e061ffecc0a1baa`.

| Protocol | Passed | Collapse loops | Syntax-valid | Cap hits |
|---|---:|---:|---:|---:|
| Existing chat-wrapped instruction | 1/164 (0.610%) | at least 149/164 observed loop forms | not comparable from legacy receipts | legacy predictions omitted finish reason |
| Corrected raw instruction, no chat | **97/164 (59.146%)** | **0/164** | 161/164 | 6/164 |
| Paper untuned-base anchor | 44.5% | not reported | not reported | not reported |

Removing the chat template while preserving the EvalScope instruction changed
the full untuned-base result by `+58.537` percentage points. The corrected local
score being 14.646 points above the paper anchor is evidence that the paper and
local corrected protocols are still not uniquely identified with each other;
it is not evidence that one evaluator is intrinsically better. The defensible
conclusion is narrower and stronger: the prior 0.610% value was dominated by an
untuned-base chat-interface mismatch.

The same exact corrected protocol was rerun on all 164 tasks for TriGLU
step-294 and baseline step-196. Historical chat-protocol values remain
immutable and are displayed beside, not overwritten by, the corrected values.

## Paired Model Comparison

| Model | Chat/parser HumanEval+ | Corrected HumanEval+ | Corrected CodeAvg |
|---|---:|---:|---:|
| Untuned base | 0.610% | 59.146% | 29.714% |
| TriGLU step-294 | 21.951% | **60.976%** | **33.439%** |
| Baseline step-196 | 31.097% | **60.976%** | 32.980% |

Corrected CodeAvg is the equal-weight mean of corrected raw-no-chat HumanEval+,
the unchanged MBPP result, and corrected LiveCodeBench. It is reported beside
the historical chat/parser aggregate and is not substituted into the old OOD
mean.

TriGLU and baseline each pass 100 tasks, but their task-level agreement is
`91 both pass / 9 TriGLU-only / 9 baseline-only / 55 both fail`. Against untuned
base, TriGLU has 16 exclusive passes and 13 regressions; baseline has 13
exclusive passes and 10 regressions. Scalar ties therefore conceal meaningful
behavioral differences.

The three models have four, three, and three sandbox timeouts respectively.
These remain failures. The worker receipts encode the CPU limit as
`execution_status=error, execution_exit_code=-24` (`SIGXCPU`), so timeout audits
must use the exit code rather than the status string alone.

## Evidence Links

- Parser and sandbox recovery:
  `docs/experiment_records/2026-07-17_humanevalplus-parser-and-sandbox-recovery-record.md`
- Untuned collapse compact metrics:
  `docs/experiment_records/compact_metrics/2026-07-17_untuned_base_humanevalplus_collapse_audit.json`
- Baseline step-196 OOD control:
  `docs/experiment_records/2026-07-17_baseline-step196-ood-control-record.md`
- Generator implementation:
  `src/qwen_single_layer_rl/eval/generator_backend.py`

## 继承待办 / Pending Obligations Carried Forward
Canonical registry:
`docs/experiment_plans/PENDING_2026-07-12_mandatory-experiment-gates.md`

- **PENDING-01 Eval Parity Matrix:** still pending. This record narrows one
  evaluator mismatch but does not complete the HF/vLLM matrix.
- **PENDING-02 Pure-BF16 SHS And TriGLU:** deferred and unchanged.
- **PENDING-03 Registered SHS CausalLM Route:** deferred and unchanged.
