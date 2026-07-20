# 2026-07-11 Eval Microbatch And Live Trend Record / 中英混搭版

<!-- bilingual-companion-v1 -->

> **English original / 英文原件：** [2026-07-11_eval-microbatch-and-live-trend-record.md](../2026-07-11_eval-microbatch-and-live-trend-record.md)
> **中文导读 / Bilingual guide：** 这份 companion 的主题是 **2026-07-11 Eval Microbatch And Live Trend Record**。下面完整保留 original report 的 metrics、tables、commands、paths、hashes 与 scientific claims，同时加入中文 narrative layer 和 bilingual section handles，方便项目负责人阅读与后续 Agent 检索。
> **Evidence contract / 证据契约：** English original 保持 immutable；如果 companion 与原件出现 wording tension，以原件中的 exact numbers、machine-readable artifacts 和 source paths 为准，companion 不得 retroactively 改写实验事实。


## Motivation

EvalScope 1.8.1 uses `eval_batch_size` as the number of worker threads for the
native evaluator. The initial custom model protected single-example HF
`generate()` with one lock. Increasing `eval_batch_size` therefore created more
waiting threads but did not create GPU inference batches, which left most of
the RTX PRO 6000 memory unused and risked making full paper evaluation dominate
the experiment runtime.

## Static Microbatch Design

The custom model now places synchronous EvalScope requests into a 10 ms
rendezvous queue. Requests with an identical generation signature are grouped
up to the configured evaluation batch size and executed in one HF
`model.generate()` call. Different greedy/sampling configurations never share a
batch.

The default production evaluation batch size is 8. The tokenizer uses left
padding, and generated text is sliced after the common padded prompt width.
This is static batching rather than vLLM continuous batching; it preserves the
custom SHS/TriGLU/OFT model implementation and the existing EvalScope scoring
path without requiring a custom inference-engine kernel.

Every unlimited call through `launch_sft_final_eval.sh` now first runs a bounded
gate with one problem per benchmark, two sampled AMC responses, 64 maximum new
tokens, and the selected batch size. Limited preflights do not recursively
launch another preflight. Full evaluation starts only after that gate exits
successfully.

## 验证 / Verification
- Concurrent batcher test: four same-signature requests formed one batch and
  returned results to the correct callers.
- Signature isolation test: greedy and sampled requests formed separate
  batches.
- Real Qwen3 tokenizer integration: two differently sized conversations
  produced batched shape `[2, 22]`; their non-padding token IDs exactly matched
  individual lengths 15 and 22.
- Remote full repository suite: 49 passed.
- Shell syntax checks passed for final-eval and ordered launchers.

A second full model was intentionally not loaded on the actively training GPU.
The real CUDA microbatch gate is scheduled automatically at the SHS
train-to-eval handoff, so training timing and loss measurements remain
uncontaminated.

## Live Training Trend

The monitor now reads the append-only per-step metrics and reports first-window
and recent-window loss means, their delta, recent median step time, training
ETA, and all recorded validation points.

At the first post-implementation observation:

- SHS step: `614 / 3916` (`15.68%`).
- First-100 training loss mean: `0.609055`.
- Last-100 training loss mean: `0.612808`.
- Delta: `+0.003754`.
- Last-100 median optimizer step: `1.484` seconds.
- Remaining pure-training ETA: `01:21:39`.
- 10% validation: step 392, loss `0.643324`, validation time `1.096` seconds.
- 10% checkpoint: three required files present, total size approximately
  547 MB, and `latest.json.global_step=392`.

One validation point and a near-flat noisy training window do not establish a
quality trend. The process remains healthy and economically meaningful; later
25%, 50%, 75%, and 100% validation points are required for interpretation.

## SHS 50% Milestone

The bounded 04:14 +08:00 heartbeat observed step `2054 / 3916` (`52.45%`).
The remote screen remained detached and active, GPU utilization was 94%, peak
training allocation remained 18.39 GB, and the data volume retained 220 GB
free. No training or evaluation errors were present.

Validation loss improved at every recorded milestone:

- Step 392 (10%): `0.643324`.
- Step 979 (25%): `0.632509`.
- Step 1958 (50%): `0.625547`.

The last-100 training loss mean was `0.598982`, versus first-100 mean
`0.609055` (delta `-0.010073`). Median optimizer-step time remained `1.484`
seconds, with an estimated 46 minutes of SHS training remaining. All three
scheduled checkpoint manifests were present. The run remained healthy, so no
intervention or additional polling was performed.

## SHS 75% Milestone

The bounded 04:45 +08:00 heartbeat observed step `3271 / 3916` (`83.53%`).
The detached screen remained active, GPU utilization was 95%, training memory
remained 18.39 GB, 220 GB of data-volume space remained free, and no errors
were present.

The 75% validation extended the monotonic milestone trend:

- Step 392 (10%): `0.643324`.
- Step 979 (25%): `0.632509`.
- Step 1958 (50%): `0.625547`.
- Step 2937 (75%): `0.621546`.

The last-100 training loss mean was `0.580312`, down `0.028743` from the
first-100 mean. Median optimizer-step time was `1.483` seconds and the
remaining SHS training ETA was approximately 16 minutes. The four scheduled
checkpoint manifests through 75% were present. No intervention or repeated
check was needed.

## SHS Training Completion And Evaluation Handoff

The bounded 05:14 +08:00 heartbeat observed exact SHS training completion at
step `3916 / 3916`. Training-loop wall time was `5822.787` seconds, or about 1
hour 37 minutes. The final checkpoint, `train_result.json`, and all five
scheduled checkpoint manifests were present, and the exact-completion handoff
resolved the final checkpoint before evaluation started.

Validation loss improved monotonically across the full run:

- Step 392 (10%): `0.643324`.
- Step 979 (25%): `0.632509`.
- Step 1958 (50%): `0.625547`.
- Step 2937 (75%): `0.621546`.
- Step 3916 (100%): `0.620852`.

The final last-100 training loss mean was `0.584501`, down `0.024554` from the
first-100 mean. The ordered controller entered the SHS paper-pinned evaluation
at 05:01 +08:00. At the heartbeat, MATH-500 was at 96/500 after approximately
13 minutes, GPU utilization was 66%, evaluation memory use was 11.96 GB, and
219 GB of data-volume space remained free. The evaluation screen remained
active and no real runtime error was present.

The monitor's previous broad case-insensitive `Error` pattern matched the
benign EvalScope config field `ignore_errors=false`. The pattern was narrowed
to actual error/exception forms so later heartbeats do not report this false
positive. No training or evaluation process was interrupted.

## SHS MATH-500 Completion

The bounded 06:45 +08:00 heartbeat confirmed that SHS completed the first
paper-pinned benchmark, MATH-500, and automatically advanced to GSM8K. At the
observation point GSM8K was at `472 / 1319` after approximately 20 minutes.
The detached screen remained active, the GPU and data volume were healthy, and
no runtime errors were present. The bounded monitor did not perform an
additional report read; the final score remains in the EvalScope artifact and
will be collected with the normal evaluation receipt.

## SHS GSM8K Completion

The bounded 07:45 +08:00 heartbeat confirmed that SHS completed GSM8K and
automatically advanced to the third paper-pinned benchmark, OlympiadBench. At
the observation point OlympiadBench was at `80 / 675` after approximately 17
minutes. The detached screen, GPU, and data volume remained healthy, and no
runtime errors were present. As with MATH-500, the bounded monitor left report
collection to the final evaluation receipt and performed no extra remote read.

## SHS OlympiadBench Completion

The bounded 10:15 +08:00 heartbeat confirmed that SHS completed OlympiadBench
and automatically entered the final paper-pinned phase, AMC23 Average@32. At
the observation point AMC sampling was at `177 / 1280` after approximately 23
minutes. Sampling used about 37.7 GB of GPU memory; GPU, disk, and the detached
screen remained healthy, with no runtime errors. The final AMC report and the
three completed main-task reports remain delegated to the normal hashed
evaluation receipt.

## SHS Evaluation Completion And Baseline Handoff

The bounded 13:16 +08:00 heartbeat confirmed completion of the full SHS
paper-pinned evaluation. AMC Average@32 completed all `1280 / 1280` sampled
responses in `3:19:47`; EvalScope wrote the AMC report, the final-eval launcher
exited successfully, and the ordered controller wrote the hash-bound
`evaluation_complete.json` receipt.

The controller then advanced automatically to the second variant, the
whole-layer identity baseline. At the observation point baseline training was
at step `275 / 3916` (`7.02%`), with a recent median optimizer-step time of
`0.943` seconds, 10.59 GB peak training allocation, and an estimated 57 minutes
of pure training remaining. The screen, GPU, and data volume were healthy and
no errors were present.

The bounded monitor did not perform an additional report read. SHS benchmark
scores remain in the four EvalScope reports bound by the completion receipt and
will be collected through the planned compact result-analysis step.

## SHS Primary Math Scores

The four hash-bound EvalScope reports contain the following primary scores:

| Benchmark | Samples | SHS SFT score |
|---|---:|---:|
| MATH-500 | 500 | 59.00 |
| GSM8K | 1,319 | 76.95 |
| OlympiadBench | 675 | 22.96 |
| AMC23 Average@32 | 1,280 | 13.44 |
| Unweighted math average | - | 43.09 |

For orientation only, the paper's Qwen3-1.7B base row is 57.4, 74.4, 18.7,
26.1, and 44.1 respectively. The reconstructed SHS SFT run is higher on the
first three tasks but much lower on AMC, leaving its unweighted average about
1.01 points below the paper base row. This is not a controlled causal
comparison: the paper does not publish its AMC sampling hyperparameters or
complete evaluator recipe, and the local untrained base checkpoint has not yet
been evaluated under this exact reconstructed protocol.

The active whole-layer baseline SFT run will provide the first controlled
same-data, same-seed, same-evaluator comparison. Cap-hit, extraction-missing,
and continuation diagnostics remain secondary analyses and must not alter
these primary scores.
