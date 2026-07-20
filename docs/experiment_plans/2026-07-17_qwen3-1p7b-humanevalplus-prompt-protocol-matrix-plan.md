# Qwen3-1.7B HumanEvalPlus Prompt-Protocol Matrix Plan

Date: 2026-07-17

Status: **COMPLETE.** The 32-task canary, full-164 untuned-base validation, and
paired full-164 TriGLU step-294 and baseline step-196 reruns completed on
2026-07-17 under the same corrected protocol and exact task ledger.

## Purpose

Determine whether the untuned Qwen3-1.7B-Base HumanEval+ collapse is caused by
the current zero-shot instruction plus Qwen chat-template interface rather than
by the checkpoint, tokenizer, parser, sandbox, or code capability itself.

This is a paper-comparability diagnostic, not a claim to reproduce the paper's
undisclosed evaluator exactly. The paper reports the benchmark and score but
does not provide enough prompt, harness, decoding, stop, or parser detail to
identify its HumanEval+ protocol uniquely.

## Proposed Identity

No command may launch these identities until the owner approves them.

### Canary

- Run ID: `qwen3_1p7b_base_heplus_prompt_matrix_canary32_20260717_v1`
- Screen: `qwen_base_heplus_prompt_matrix_canary32_20260717_v1`
- Output root:
  `runs/eval_protocol/qwen3_1p7b_base_heplus_prompt_matrix_canary32_20260717_v1`
- Model: the exact untuned Qwen3-1.7B-Base revision used by the completed OOD
  control.

### Conditional full evaluation

- Run ID: `qwen3_1p7b_heplus_prompt_matrix_full164_20260717_v1`
- Screen: `qwen_heplus_prompt_matrix_full164_20260717_v1`
- Output root:
  `runs/eval_protocol/qwen3_1p7b_heplus_prompt_matrix_full164_20260717_v1`
- Initial model: untuned Qwen3-1.7B-Base.
- Conditional comparison models: baseline step-196 and TriGLU step-294, only
  after one prompt cell materially repairs the untuned-base generation.

## Frozen Matrix

The only intended independent variable is prompt rendering.

| Cell | Prompt formulation | Chat template | Role |
|---|---|---|---|
| `evalscope_chat_instruction_control` | Existing EvalScope zero-shot instruction and task payload | Qwen `apply_chat_template(..., add_generation_prompt=True)` | Immutable current-protocol control |
| `evalscope_raw_instruction_nochat` | Byte-identical EvalScope instruction and task payload | None; raw text | Isolate chat-template effect |
| `canonical_completion_nochat` | Canonical function signature/docstring completion prompt | None; raw text | Test base-model completion interface |

The matrix must not change model weights, tokenizer revision, task identities,
greedy decoding, response cap, parser-v2, privilege-dropped sandbox, tests, or
scorer between cells. If a cell requires different stop sequences, that is a
separate follow-up matrix and must not be folded into these results silently.

## Canary Sample Contract

- Select 32 of the 164 HumanEval+ task IDs by sorting the SHA-256 of
  `20260707:<task_id>` and taking the first 32.
- Persist the ordered task-ID ledger and its SHA-256 before generation.
- Run all three cells on the same ordered ledger.
- Use greedy decoding (`temperature=0`, no sampling), seed `20260707`, and
  `max_tokens=3072`.
- Preserve rendered prompt text and SHA-256, tokenized prompt IDs and SHA-256,
  raw output, generated-token count, `finish_reason`, parser strategy, sandbox
  verdict, and final pass/fail for every row.
- Require exactly 32 unique rows per cell, zero missing rows, zero duplicates,
  and byte-stable rerendering before a score is published.

The full 164-task untuned-base evaluation is justified when a non-control cell
both improves execution passes by at least three tasks over the control and
materially reduces no-code Hebrew/fragment loops. The full run remains
diagnostic even if its score approaches the paper anchor; matching one scalar
does not prove exact evaluator parity.

## 2026-07-17 Canary Result And Full Gate

The approved canary ran all three cells on the frozen 32-task ledger with SHA-256
`e9a901f7c91f96e2651a80e806c279cf43e6ed3c8c9ffc393392bd3ffab79d0d`.
The source prediction SHA-256 remained
`4976c4442b404a2a5a36bbd7287434bfc46a1be3b912a67666a4a542f48a29b6`.

| Cell | Passed | Collapse loops | Syntax-valid completions | Cap hits |
|---|---:|---:|---:|---:|
| Chat-wrapped EvalScope instruction control | 0/32 | 32 | 32 | 32 |
| Raw EvalScope instruction, no chat template | **21/32** | **0** | 31 | 1 |
| Canonical completion, no chat template | 6/32 | 0 | 14 | 16 |

The preregistered gate required at least three additional passes and at least
three fewer collapse loops than the control. The winning raw-instruction cell
delivered `+21` passes and `-32` collapse loops, so the gate passed decisively.

The full-164 validation keeps the winning cell fixed and uses six deterministic
shards under run ID `qwen3_1p7b_heplus_prompt_matrix_full164_20260717_v1` and
screen `qwen_heplus_prompt_matrix_full164_20260717_v1`. Before launch, the
material-recovery gate is fixed at at least `49/164` passes and no more than
`16/164` collapse loops. Passing this gate supports a new separately named
base-completion protocol candidate; it still does not prove exact paper parity.

## 2026-07-17 Full Untuned-Base Result

The full untuned-base run preserved all 164 identities under ledger SHA-256
`7bb4ed06a3a4c725a9893f38e76087c9d6bf2c3caa8d0c880e061ffecc0a1baa`.
The winning raw EvalScope instruction without a chat template scored
`97/164 = 59.146%`, with zero collapse loops, 161 syntax-valid completions, and
six cap hits. This exceeds the preregistered recovery threshold of `49/164`
passes and at most 16 collapse loops.

The old untuned-base chat score `1/164 = 0.610%` therefore measures a severe
interface mismatch, not the checkpoint's HumanEval+ capability under a suitable
base-model prompt. The local corrected value is also above the paper's 44.5%
anchor, so it must not be described as exact paper parity; prompt, stop, harness,
and scorer details remain underdetermined by the paper.

The authorized comparison stage reruns every HumanEval+ task for TriGLU
step-294 and baseline step-196. Both use the same raw-no-chat prompt, seed,
3,072-token cap, parser-v2, sandbox, six-way shard rule, and exact ledger SHA as
the untuned-base run. TriGLU executes first and must produce a valid registered
custom-FFN dispatch receipt on every rank before merge; baseline follows only
after the TriGLU merge passes exact coverage.

## Final All-Model Result

| Model | Legacy chat/parser | Raw no-chat corrected | Change | Syntax-valid | Cap hits | Sandbox timeouts |
|---|---:|---:|---:|---:|---:|---:|
| Untuned base | 1/164 (0.610%) | 97/164 (59.146%) | +96 tasks / +58.537 pp | 161 | 6 | 4 |
| TriGLU step-294 | 36/164 (21.951%) | **100/164 (60.976%)** | +64 / +39.024 pp | 159 | 6 | 3 |
| Baseline step-196 | 51/164 (31.097%) | **100/164 (60.976%)** | +49 / +29.878 pp | 164 | 3 | 3 |

TriGLU and baseline tie in aggregate score but not per task: 91 tasks pass for
both, nine pass only for TriGLU, nine pass only for baseline, and 55 fail for
both. The corrected result therefore does not support behavioral equivalence.
All timeout verdicts remain failures; the Linux sandbox records its CPU timeout
as `execution_status=error` with `exit_code=-24` (`SIGXCPU`).

Using each model's unchanged MBPP score and corrected LiveCodeBench score, the
equal-weight corrected CodeAvg is 29.714% for untuned base, 33.439% for TriGLU,
and 32.980% for baseline. Historical chat/parser CodeAvg and the original OOD
means remain preserved separately because replacing one benchmark's prompt
inside an old aggregate would silently create a mixed protocol.

## Full-Evaluation Decision Logic

1. Run the winning prompt formulation on all 164 untuned-base tasks.
2. If the full score and failure distribution recover materially, freeze that
   formulation as a separately named `base_completion_protocol` candidate.
3. Re-evaluate baseline step-196 and TriGLU step-294 with exactly the same
   candidate protocol so architecture comparisons remain paired.
4. Preserve all current `vllm_greedy_pass_at_1_max_tokens_3072` results as the
   `internal_chat_protocol`; never overwrite or relabel them.
5. If no cell recovers, stop before model comparison and audit model/tokenizer
   revision, generation stops, response extraction boundaries, and external
   harness differences. Do not add parser heuristics to compensate for outputs
   that contain no implementation.

## Required Report

For each cell and model, report:

- exact pass count and denominator;
- task-level agreement and disagreements against the control;
- no-code, Hebrew-loop, English-fragment-loop, valid-code, syntax-valid, and
  cap-hit counts;
- finish-reason distribution and generated-length quantiles;
- parser-strategy distribution and sandbox failure categories;
- prompt/token hashes, model/tokenizer revision hashes, and task-ledger hash;
- wall time, generated tokens/s, peak GPU memory, and six-rank merge coverage;
- paper anchor beside, but never merged with, the local protocol scores.

## Scientific Boundaries

- This experiment evaluates prompting and harness compatibility only.
- It does not modify, train, merge, or convert any checkpoint.
- Parser-v2 and sandbox repairs are fixed evaluator infrastructure, not matrix
  variables.
- The paper-reported untuned-base HumanEval+ value is an orientation anchor,
  not an acceptance oracle, because the paper does not disclose the exact
  evaluator protocol.
- LiveCodeBench's near-exact paper-anchor match is evidence against gross
  checkpoint corruption, but it does not prove HumanEval+ protocol identity.

## Pending Obligations Carried Forward

Canonical registry:
`docs/experiment_plans/PENDING_2026-07-12_mandatory-experiment-gates.md`

- **PENDING-01 Eval Parity Matrix:** in scope only as a focused prompt-protocol
  diagnostic. This plan does not close the broader HF-versus-vLLM matrix.
- **PENDING-02 Pure-BF16 SHS And TriGLU:** deferred; no architecture or dtype
  path changes are permitted here.
- **PENDING-03 Registered SHS CausalLM Route:** deferred; SHS is out of scope.
