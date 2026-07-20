# 2026-07-11 AMC Greedy Modal-Path Diagnostic Plan

Date: 2026-07-11

Status: approved and queued inside the active serial controller before TriGLU
training. The diagnostic must not overlap baseline evaluation or any training
process.

## Question

Does the SHS checkpoint still assign the highest probability to a substantially
correct AMC reasoning path, while temperature-1.0 sampling exposes a flatter or
poorly calibrated distribution of incorrect paths?

## Primary Diagnostic

Proposed run label:

```text
amc_greedy_modal_path_shs_sft50k_v1
```

Evaluate the completed SHS checkpoint on the same 40-question AMC23 snapshot
with:

```text
do_sample=false
temperature=0-equivalent greedy generation
repeats=1
max_new_tokens=3072
same prompt templates
same tokenizer and answer extractor
same checkpoint and benchmark revisions
```

In the local Hugging Face evaluator, `do_sample=false` is true greedy token
selection. It should not be described merely as a low-temperature sample.

The existing paper-aligned AMC Average@32 score remains primary for the current
wave. It is not paper-identical: the paper publishes neither its sampling
temperature nor its top-p setting. Greedy AMC is a secondary mechanism
diagnostic and must never replace the sampled score.

## Approved Queued Launch

The active wave uses this exact scoped insertion:

```text
trigger config: layer10_whole_layer_triglu_side_ffn_sft.yaml
hook script: scripts/launch_greedy_amc_controls_before_triglu.sh
control order: SHS greedy, whole-layer baseline greedy, untuned base greedy,
  untuned base Average@32
SHS output: <run-root>/diagnostics/amc_greedy_modal_path_shs_sft50k_v1
baseline output: <run-root>/diagnostics/amc_greedy_modal_path_baseline_sft50k_v1
base output: <run-root>/diagnostics/amc_greedy_modal_path_untuned_qwen3_1p7b_base_v1
base sampled output: <run-root>/diagnostics/amc_average_at_32_untuned_qwen3_1p7b_base_v1
per-control preflight: 16 AMC prompts, 64 max tokens, eval_batch_size=16
per-control production: 40 AMC prompts, 3072 max tokens, eval_batch_size=16
checkpoints: completed SHS and whole-layer baseline step_00003916
base weights: untuned Qwen3-1.7B-Base with no architecture surgery or checkpoint load
post-eval controls:
  amc_greedy_modal_path_triglu_sft50k_v1
  amc_greedy_modal_path_oft_sft50k_v1
```

The pre-TriGLU hook runs all four controls synchronously in the existing serial process
immediately before TriGLU training. It does not create a second screen or
overlap the baseline evaluation. A separate hash-bound
`diagnostic_complete.json` with exactly 40 review rows is required for each
control before TriGLU begins. Each completed control is independently and
idempotently skipped on resume.

TriGLU and OFT each receive one greedy AMC pass@1 control only after their
primary final-evaluation receipt has been written. Their normal final-evaluation
order is main benchmarks followed by AMC Average@32, so these hooks are
specifically post-Average@32. They use separate work directories and receipts.

For the already-running 2026-07-11 wave, the parent ordered-loop body was parsed
before this extension was deployed. A current-wave bridge therefore verifies
the TriGLU primary receipt immediately before OFT starts, then runs the TriGLU
greedy control. A lightweight waiter verifies the OFT primary receipt before
running the terminal OFT greedy control. Both use the same idempotent diagnostic
receipts as the future-wave controller hooks and cannot overlap the gated
training/evaluation phases.

## Comparison Set

Minimum comparison:

1. SHS SFT checkpoint under greedy decoding.
2. SHS SFT checkpoint under the completed temperature-1.0 Average@32 run.

Queued paired controls:

3. Whole-layer baseline SFT checkpoint under greedy decoding.
4. Whole-layer baseline SFT checkpoint under temperature-1.0 Average@32.
5. Untuned Qwen3-1.7B-Base under the identical greedy harness. This path records
   `base_model_only=true` and `checkpoint_dir=null`; it applies no variant and
   loads no SFT state.

6. Untuned Qwen3-1.7B-Base under the sampled Average@32 harness.
7. TriGLU SFT checkpoint under greedy decoding after its primary Average@32.
8. OFT SFT checkpoint under greedy decoding after its primary Average@32.

These three additions were approved on 2026-07-11. The untuned-base sampled run
uses an AMC-only evaluator phase, so it produces exactly 1,280 sampled reviews
without redundantly re-running a greedy main phase.

## Metrics

Report:

- greedy exact-answer accuracy over 40 questions;
- per-question greedy correctness;
- sampled Average@32 accuracy and per-question success frequency;
- whether the greedy answer equals the empirical modal sampled answer;
- empirical answer-frequency entropy for each question, with extraction-aware
  normalization and an explicit unresolved-format bucket;
- response length, cap-proxy status, extraction fidelity, and failure mode;
- easy-question slices, especially AMC 12A Problem 1 and AMC 12B Problem 1;
- paired SHS-versus-baseline differences on the same questions.

Do not claim true model-distribution entropy from only 32 text samples. Label it
empirical answer-frequency entropy.

## Interpretation Matrix

| Result | Interpretation |
|---|---|
| Greedy much better than Average@32 | Correct modal knowledge may remain while sampled probability mass is too diffuse or unstable. |
| Greedy remains poor on easy questions | The failure is not solved by removing sampling; skill interference, overthinking, or model changes are more likely. |
| SHS greedy poor but baseline greedy healthy | Supports an SHS-specific architecture effect. |
| SHS and baseline greedy both poor after SFT | Supports a common SFT-recipe or prompt/evaluator effect. |
| Untuned base healthy under the same harness | Strengthens a training-induced interference claim. |
| Untuned base also poor under the same harness | Strengthens a protocol/model-scale explanation. |

Greedy success does not prove that the entire reasoning distribution is healthy.
It establishes only that the locally highest-probability token path performs
better under this decoding policy.

## Optional Temperature Sweep

Only after the greedy result, consider a compact sweep such as temperatures
0.2, 0.5, and 0.7 with a pre-approved repeat count. Do not launch a broad sweep
by default. The greedy control has the highest information-to-cost ratio and
should come first.

## Validation And Fairness Gates

- Use a separate work directory and run label; never write into the primary
  Average@32 artifact tree.
- Bind results to checkpoint, config, tokenizer, dataset, prompt, and code
  hashes.
- Confirm `do_sample=false` in the saved generation manifest.
- Use one repeat for greedy decoding; repeated deterministic outputs add no
  information unless nondeterminism is explicitly being measured.
- Reuse the same extractor and verifier, then run the semantic disagreement
  audit on any score changes.
- Do not inspect answers to tune decoding parameters before reporting the
  pre-registered greedy result.
- Preserve all generated traces and compact summaries.

## Launch Gate

The exact scope, output roots, IDs, and inclusion of the untuned base greedy
control were approved on 2026-07-11. The active ordered SFT wave must remain
uninterrupted.
