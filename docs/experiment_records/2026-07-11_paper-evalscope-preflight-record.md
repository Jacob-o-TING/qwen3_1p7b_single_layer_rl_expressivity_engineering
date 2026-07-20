# 2026-07-11 Paper-Pinned EvalScope Preflight Record

## Scope

This preflight verifies the complete checkpoint-to-report path for the two
initial custom local benchmark adapters required by the paper-aligned evaluation:

- `paper_olympiadbench`: 675 English text-only OlympiadBench problems.
- `paper_amc23`: 40 AMC 2023 problems.

The integration checkpoint was the completed two-step whole-layer baseline
SFT smoke checkpoint at
`runs/sft_milestone_smoke_20260711/checkpoints/step_00000002`. Each preflight
used one sample and at most 64 generated tokens. Scores from this bounded smoke
run are not research results.

## Environment

- EvalScope: 1.8.1 in `envs/evalscope181`.
- Model/checkpoint reconstruction: training environment Transformers/Torch
  site packages injected into the isolated EvalScope environment.
- Device: one NVIDIA RTX PRO 6000 Blackwell Server Edition.
- Generation: greedy, temperature 0, batch size 1.

## Findings And Fixes

### Local JSONL loader

EvalScope's default adapter recognized that the pinned JSONL path existed but
then delegated to its remote loader. That loader treated the file path as a Hub
dataset root and failed before generation. The shared pinned-local adapter now
forces EvalScope's `LocalDataLoader`, which directly supports JSONL and still
applies the benchmark-specific `record_to_sample` functions.

### Report metadata

After local loading was fixed, the Olympiad preflight completed model loading,
generation, answer extraction, and MathJudger scoring. Report construction then
failed because EvalScope 1.8.1 requires a string dataset description rather
than `None`. Both custom benchmark metadata objects now provide explicit
descriptions.

Both were ordinary integration issues. Neither met the economically abnormal
shutdown condition.

## Passing Results

OlympiadBench:

- Evaluation time for one bounded sample: 1.98 seconds.
- Full path passed: local load, prompt formatting, compact checkpoint restore,
  CUDA generation, answer extraction, MathJudger scoring, JSON report, and HTML
  report.
- Remote artifact root: `eval_artifacts/preflight_paper_olympiad_v3`.
- Final marker: `SFT_FINAL_EVAL_END`.

AMC 2023:

- Evaluation time for one bounded sample: 1.89 seconds.
- Full path passed: local load, prompt formatting, compact checkpoint restore,
  CUDA generation, numeric answer extraction/scoring, JSON report, and HTML
  report.
- Remote artifact root: `eval_artifacts/preflight_paper_amc23_v1`.
- Final marker: `SFT_FINAL_EVAL_END`.

The custom benchmark runtime gate is therefore complete. Full production
evaluation remains configured for the exact dataset counts and the separate
AMC Average@32 treatment.

The final production registration also replaces EvalScope's built-in MATH-500
and GSM8K entries with `paper_math500` and `paper_gsm8k`. All four math tasks
therefore read only the independently hashed local snapshots; no production
math score depends on a mutable Hub revision or a built-in GSM8K few-shot
default.

### Average@32 sampling addendum

Greedy repetition would produce 32 identical outputs and would not implement a
meaningful Average@32. The production AMC task therefore uses seeded
multinomial sampling. Because the paper does not publish a temperature or
nucleus threshold, the pinned neutral protocol is `temperature=1.0` and
`top_p=1.0`: this preserves the model's native softmax distribution and adds no
reshaping or truncation beyond the random draw itself.

A bounded two-repeat runtime preflight confirmed that EvalScope reports two
samples and that the generated responses differ. The exact 32-repeat production
recipe is recorded in each evaluation manifest.

### All-four pinned adapter gate

The final combined gate ran from
`eval_artifacts/preflight_all_paper_math_v1` against the same two-step smoke
checkpoint. It produced four independent JSON reports and the expected bounded
prediction counts:

- `paper_math500`: 1 prediction and 1 review.
- `paper_gsm8k`: 1 prediction and 1 review.
- `paper_olympiadbench`: 1 prediction and 1 review.
- `paper_amc23`: 2 sampled predictions and 2 reviews.

The evaluation manifest records the three pinned main datasets, the pinned AMC
dataset, and `amc_repeats=2` for this bounded gate. The repository test suite
also passed remotely with 40 tests before the runtime gate. This completes the
local-snapshot data-to-report verification for every production math adapter.
