# 2026-07-17 HumanEvalPlus Parser And Sandbox Recovery Record

Date: 2026-07-17

Status: **PASS for cached-prediction parser recovery.** The corrected
parser-only score is `36/164 = 21.95%`. No model was loaded, no generation was
invoked, and no checkpoint or prediction was modified.

## Fixed Scope

- Source run: `qwen3_1p7b_ood_6x5090_step294_20260716_v1/triglu`.
- Source prediction SHA-256:
  `d25bbd9e581f35a7a4b8bd6f4c6cf1dff1f897281bf34e3c43fd3187ab67e2ff`.
- Historical raw report SHA-256:
  `bb5a4a52dd1615845e1ed624b27b8f3df4d1337597e1a564c6293b92a395aa19`.
- Rows: 164 fixed HumanEval+ predictions.
- Allowed operations: deterministic output parsing and privilege-dropped
  sandbox review only.
- Forbidden operations: loading the model, generation, checkpoint mutation,
  prediction mutation, or changing the active OOD/GRPO processes.

## Root Causes

The historical `0.0` was not a valid statement that the model solved zero
HumanEval+ tasks. Two evaluator defects were compounded:

1. EvalScope's built-in postprocessor extracted fenced code only. Unfenced
   outputs prefixed by Hebrew response labels were sent to Python unchanged.
2. The local sandbox allowed OpenBLAS to request 64 threads while enforcing
   `RLIMIT_NPROC=32`. HumanEval+ imports NumPy in its expanded tests, so NumPy
   initialization failed before otherwise valid solutions could be judged.

The sandbox repair pins OpenBLAS, OMP, MKL, NumExpr, vecLib, and BLIS to one
thread. Isolation remains `uid65534 + no_new_privs + rlimits +
seccomp_no_network_no_exec`. A focused control imported NumPy 1.26.4
successfully after the repair.

## Conservative Parser Contract

The parser is fail-closed and applies only these deterministic rules:

- select the first fenced Python code block;
- remove a complete `<think>...</think>` block;
- strip a non-code prefix when a Python declaration is present;
- strip a non-code prefix before an indented function body.

Clean function text and already-indented completion bodies remain byte-exact.
Outputs with no code anchor remain unchanged. The syntax audit moved from
`76/164` raw syntax-valid programs to `122/164`; the remaining 42 syntax
failures were no-code Hebrew boilerplate and were not guessed or fabricated.

## Controlled Results

| Review condition | Passed | Score | Interpretation |
|---|---:|---:|---|
| Historical EvalScope report | 0/164 | 0.00% | Confounded by parser and sandbox defects |
| Raw text, fixed sandbox | 1/164 | 0.61% | Sandbox-only control |
| Improved parser, fixed sandbox | **36/164** | **21.95%** | Corrected parser-only cached review |

The isolated parser gain over the fixed-sandbox raw control is **+35 tasks** or
**+21.34 percentage points**. Passing rows by parser strategy were:

- first fenced block: 14;
- prefix to Python declaration: 20;
- prefix to indented function body: 1;
- unchanged: 1.

The other 128 rows remain genuine failures under this review protocol. They
must not be reclassified merely because parsing improved.

## Durable Evidence

Local ignored run artifacts preserve summaries and all 164 row receipts under:

- `runs/ood_eval/qwen3_1p7b_ood_6x5090_step294_20260716_v1/triglu/diagnostics/humanevalplus_parser_v2_reviewonly_20260717_v2/`;
- `runs/ood_eval/qwen3_1p7b_ood_6x5090_step294_20260716_v1/triglu/diagnostics/humanevalplus_raw_text_fixed_sandbox_control_20260717_v1/`;
- `runs/ood_eval/qwen3_1p7b_ood_6x5090_step294_20260716_v1/triglu/diagnostics/humanevalplus_parser_v2_revised_syntax_audit/`.

The local SHA-256 values match the remote receipts. Source and tests are in
`humanevalplus_parser.py`, `local_code_sandbox.py`,
`rescore_humanevalplus_parser_only.py`, and their focused unit tests.

## Cross-Benchmark Hebrew Output Audit

A read-only follow-up scanned every currently available prediction under the
step-294 TriGLU OOD root. The available root contains TriGLU only; baseline and
untuned-base OOD predictions do not yet exist there, so this audit cannot
attribute the behavior specifically to TriGLU or to RL.

Across 13,336 predictions, 218 contained at least one Hebrew code point. No
input prompt contained a literal Hebrew code point, and the persisted assistant
message matched the raw `model_output` content for every row. The Hebrew text
therefore originated during generation rather than prompt rendering, UI
display, or prediction serialization.

| Benchmark family | Hebrew rows | Exact finish evidence | Existing review observation |
|---|---:|---|---|
| HumanEval+ | 148/164 | 92 stop, 56 length | Historical raw reviews all zero; corrected parser review 36/164 overall |
| MBPP | 42/500 | 5 stop, 37 length | 7/42 pass |
| C-Eval | 18 rows | 3 stop, 15 length | 9/18 pass with current last-answer extraction |
| IFEval | 10/541 | 2 stop, 8 length | 1 strict pass among 8 rows with available row review |

No Hebrew output was found in the currently available GPQA, MMLU-Pro, MGSM,
or other prediction files outside the C-Eval subsets listed by the audit.

HumanEval+ contained 43 Hebrew-only generations. Of these, 42 were genuine
short EOS terminations rather than hidden or truncated code:

- 22 produced `Your answer:` in Hebrew and stopped after five generated tokens
  including EOS;
- 19 produced `Your answer / Your answer is:` and stopped after ten generated
  tokens including EOS;
- one produced a Hebrew sentence saying that the response would contain only
  code and stopped after fourteen generated tokens including EOS.

The remaining Hebrew-only row repeated a Hebrew word for exactly 3,072 tokens
and ended with `finish_reason=length`. Thus both short-EOS collapse and
length-cap repetition exist. The tokenizer round trip, receipt token counts,
and EOS ID agree; no unseen suffix was dropped.

For MBPP, the Hebrew rows correlate mainly with genuine generation collapse,
not a recoverable parser-only problem. Of 35 failed Hebrew rows, 29 raw outputs
did not contain the expected target function at all. The other six contained
and extracted the expected function name but failed functionally. A parser
must not fabricate implementations for these rows.

For C-Eval, all 18 Hebrew rows repeated answer-bearing material. The stock
adapter selects the last `Answer: [A-D]` match. The first and last choices
differed in two rows; first-choice scoring would yield 10/18 instead of the
current 9/18. One observed loss is therefore extraction-order sensitive, while
the other failures remain answer errors or degeneration rather than missing
text.

Nine of the ten IFEval prompts were unrelated to Hebrew. One prompt asking for
names used for God could plausibly elicit a Hebrew term, but the remaining rows
showed the same repetition/template-collapse pattern. Raw instruction-following
evaluation should not silently strip those characters because language and
format compliance are part of IFEval itself.

The shared qualitative pattern is a greedy-decoding multilingual/template
attractor: the model emits a response-header phrase, sometimes transitions to
code, sometimes emits EOS immediately, and sometimes repeats to the cap. The
standard chat template contains no Hebrew. A controlled untuned-base and
single-layer-baseline comparison is required before assigning this behavior to
TriGLU expressivity or the RL trajectory.

## Untuned-Base Cached-Prediction Control

The subsequently completed untuned Qwen3-1.7B-Base OOD cell makes the
TriGLU-only interpretation untenable. Its frozen HumanEval+ predictions scored
`1/164 = 0.61%` in the historical report and remained exactly `1/164 = 0.61%`
after the conservative parser-v2 plus repaired sandbox review. The passing task
was `HumanEval/50`. The parser changed only two rows: one first fenced-code
block and one prefix-to-declaration case. It could not recover any additional
passing implementation.

The raw-generation audit found:

- 153/164 outputs contained Hebrew and no recoverable Python declaration;
- 139/164 outputs were the same Hebrew answer-label token loop repeated for
  1,024 lines;
- 10/164 outputs repeated the English fragment `umably` for 1,536 lines;
- one output contained a fenced Python implementation and was the sole pass;
- zero raw outputs began with a top-level Python `def` declaration.

The configured response cap was 3,072 tokens. The repeated line counts are
consistent with cap-reaching token loops, but the prediction JSONL did not
preserve `finish_reason`, so exact stop-versus-length attribution is not claimed
for every row. Six short outputs and the other non-cap-shaped rows likewise
cannot be assigned an EOS reason from the persisted prediction alone.

This is a genuine generation/protocol collapse under the evaluated greedy
prompt, not an output-extraction miss. The `0.61%` is therefore the correct
execution score for these frozen generations, while remaining an unreliable
estimate of normal untuned-base HumanEval+ capability. It also shows that the
Hebrew attractor is not sufficient evidence of TriGLU- or RL-induced damage.

Untuned control evidence:

- source prediction SHA-256:
  `4976c4442b404a2a5a36bbd7287434bfc46a1be3b912a67666a4a542f48a29b6`;
- raw report SHA-256:
  `8a4db7d72928a45d2c9c057c690fc7d47a53fe5cf5e4bc7e0b150c19002e794b`;
- local summary SHA-256:
  `7ec2b9a1f99524d600de8c8a3401c95a0b24eb2c4185728d5e6316aa642ec7c3`;
- local 164-row review receipt SHA-256:
  `9423ce846c4e0eaf9f317a92da66c35a263c7f6a7f7b92ed219d694c6d1ce33e`.

The local ignored evidence is under
`runs/ood_eval/qwen3_1p7b_ood_6x5090_step294_20260716_v1/untuned_base/code_heplus_mbpp/diagnostics/humanevalplus_parser_v2_reviewonly_untuned_base_20260717_v1/`.

## Prompt-Protocol Interpretation

The local generator unconditionally wraps evaluator conversations with the
Qwen chat template and `add_generation_prompt=True`. EvalScope supplies
HumanEval+ as a zero-shot instruction asking for function code. This creates an
instruction-style chat interface for the untuned base checkpoint.

The paper reports untuned-base HumanEval+ at `44.5%`, while the local frozen
generation scores `0.61%`. In contrast, the local untuned-base LiveCodeBench
score is `7.396%`, effectively matching the paper's `7.4%` anchor. The combined
evidence favors a benchmark-specific prompt/harness mismatch over global model
or tokenizer corruption. This is an inference, not proof of the paper's hidden
evaluation recipe.

The separately proposed prompt matrix is recorded in
`docs/experiment_plans/2026-07-17_qwen3-1p7b-humanevalplus-prompt-protocol-matrix-plan.md`.
It is not launched by this record and does not alter the current OOD outputs.
