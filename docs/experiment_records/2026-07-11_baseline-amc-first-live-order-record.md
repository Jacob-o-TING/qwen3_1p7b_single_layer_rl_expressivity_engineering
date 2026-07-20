# 2026-07-11 Baseline AMC-First Live Order Record

## Approved Change

During the active ordered SFT wave, the final-evaluation phase order was changed
only for `layer10_whole_layer_baseline_sft.yaml`:

```text
AMC23 Average@32 -> MATH-500 + GSM8K + OlympiadBench
```

SHS, TriGLU, and OFT retain the default main-benchmarks-first order. The ordered
runner itself was not edited or restarted. No second screen or independent
evaluation process was launched.

## Implementation

- `run_evalscope.py` accepts `--amc-first` and selects the phase tuple
  `("amc", "main")`; the default remains `("main", "amc")`.
- `launch_sft_final_eval.sh` passes the flag only when the config basename is
  exactly `layer10_whole_layer_baseline_sft.yaml`.
- Evaluation output paths and the four-report completion receipt contract are
  unchanged. The ordered runner still blocks synchronously until all four
  reports exist and their hashes are recorded.

Commit: `44402cb` (`Evaluate baseline AMC before main benchmarks`).

## Verification

Before deployment, seven focused local unit tests passed. After deployment:

- remote `bash -n scripts/launch_sft_final_eval.sh` passed;
- three remote eval-config tests passed;
- remote phase-order assertions passed for both default and AMC-first modes;
- baseline training remained healthy through step 3786/3916 before handoff;
- the original detached screen remained active after handoff;
- the baseline batched preflight evaluated AMC before the three main datasets;
- the production evaluation then started `paper_amc23` with 1,280 samples as
  its first benchmark.

The live confirmation observed production AMC progress from 0 to 9 of 1,280.
This establishes that the active serial process consumed the updated evaluator
without interruption and without launching a competing GPU job.
