# Qwen3-1.7B Layer-10 SFT DDP And Compile Build Record

Date: 2026-07-10

Plan: `docs/experiment_plans/2026-07-10_sft-ddp-compile-short-benchmark-plan.md`

## Approved Scope

- Build the deterministic packed SFT pipeline through a production-ready state.
- Preserve the architecture-development direction and all initialization,
  shuffle-map, data-order, and seed contracts.
- Run the controlled four-case matrix: baseline eager, baseline compile, SHS
  eager, and SHS compile.
- Produce human-readable live logs and a monitor script.
- Keep resumable checkpoints at a storage-conscious cadence.
- Continue on the available single RTX PRO 6000 unless an absolutely abnormal
  condition requires stopping.

## Shutdown Policy

Normal implementation failures, a compile failure, a missing dependency, one
SSH timeout, or a recoverable OOM do not justify shutting down. An automatic or
remote shutdown is reserved for an absolutely abnormal condition where
continued remote execution has clearly lost economic meaning: for example, a
single batch/step takes tens of minutes, the projected experiment duration is
grossly beyond a reasonable rental window, or repeated OOM/restart cycles make
no effective progress. Correct ordinary problems in place and continue.

If such a condition occurs:

1. Stop the affected process safely.
2. Preserve logs, manifests, and the latest valid checkpoint.
3. Record the evidence and decision in this file.
4. Commit and push source/docs only.
5. Verify no training/evaluation process remains, then request remote shutdown.

## Initial Repository Audit

- The approved SFT/compile plan exists and is committed.
- The project has `docs/experiment_plans/` but previously had no dedicated
  `docs/experiment_records/` directory.
- The older `experiment record` TXT files located in a neighboring workspace
  are cross-project notes with uncertain text encoding, not dedicated Qwen3
  record Markdown files; they were intentionally not moved into this repo.
- The complete paper-style final evaluation pipeline is not yet implemented.
  Existing code only includes layer-contribution arithmetic plus planning notes.
  Checkpoint-to-eval handoff and a pinned external evaluator recipe are therefore
  part of the production-complete gate, while the short performance benchmark
  can proceed independently.

### EvalScope handoff decision

- Primary framework: EvalScope 1.8.1, isolated from the training/vLLM
  environment.
- Stock local-model loading is insufficient for SHS/TriGLU because their
  input-dependent architecture cannot be merged into ordinary Qwen weights.
- The handoff therefore reconstructs model surgery from the committed config,
  loads the compact trainable checkpoint, and exposes it through EvalScope's
  official custom `ModelAPI` interface.
- Main task: GSM8K, MATH-500, and OlympiadBench. AMC is a separate task with 32
  repeats to preserve the paper's Average@32 treatment.
- Before a full evaluation, save EvalScope `benchmark-info --format json`
  output to pin the exact dataset IDs, splits, prompts, filters, and metrics.

### Initial eager/compile smoke

- Baseline eager warm step: 0.0459 seconds at sequence length 512.
- SHS eager warm step: 0.1060 seconds, approximately 2.31x baseline eager;
  peak allocation was 7.33 GB. This is not economically abnormal.
- Baseline compile cold step: 36.64 seconds; warm step: 0.0259 seconds.
- SHS compile cold step: 24.96 seconds; warm step: 0.4814 seconds, slower
  than SHS eager.
- Dynamo reported 8 graph breaks from dynamic-shape `aten.nonzero` operations
  inside the SHS column-block loops and 11 unique graphs. The first compile
  optimization preserves the four persistent shuffle maps but precomputes
  non-persistent per-column index buffers during module initialization.
- First-step loss drift versus eager was approximately 0.23% for baseline
  compile and 1.06% for SHS compile. Formal timing is blocked until the SHS
  graph breaks are removed and numerical parity is remeasured.
- Static per-column index buffers removed the SHS dynamic-shape graph breaks:
  Dynamo captured one graph with no `graph_break` counter. The second SHS
  compile smoke had a 127.87-second cold step and a 0.0877-second warm step,
  approximately 1.21x faster than SHS eager. Loss drift fell to approximately
  0.32%, near the baseline compile control. Formal cases use independent
  Inductor cache directories and save initial loss plus timed token throughput.

## Execution Log

### 2026-07-10: build resumed

- User approved continuing through implementation and the single-GPU benchmark.
- No GPU job or package installation had started at this point.
- Status: repository/environment audit in progress.

## Abnormal Events

None at build start.

## Normal Build Findings

### SeetaCloud SSH gateway instability

- Several connections timed out during SSH banner exchange, before
  authentication. Successful connections with the same key and the observed
  9-18 KB/s SCP rate confirm gateway/port-proxy instability rather than an SSH
  key failure.
- Local alias `autodl-qwen` now pins IPv4, identity-only authentication, a
  60-second connect timeout, and keepalives.
- Operational rule: reduce repeated short handshakes, retain an interactive
  control connection when practical, and leave long jobs in remote `screen`.
- One banner timeout is normal infrastructure noise, not a training abnormality
  and not a shutdown condition.

### Overlong prompt in the 512-token benchmark cache

- The first baseline smoke stopped during cache construction before any
  optimizer step because source row 579 had a prompt longer than the temporary
  512-token benchmark sequence length.
- This is a normal data-boundary finding, not an economically abnormal run and
  not a shutdown condition.
- Active policy: preserve each complete prompt, truncate only the solution when
  the combined example is too long, and deterministically skip a row when its
  prompt alone cannot fit. Record all skipped source indices and their hash in
  the packed-cache manifest so every architecture uses the same retained rows.

## Checkpoint And Trend Policy

- Emit lightweight human-readable and JSONL metrics every optimizer step so the
  complete loss/throughput/memory trend is retained.
- Permanently save compact trainable-weight checkpoints at approximately 10%,
  25%, 50%, 75%, and 100% of each variant's optimizer steps.
- Align validation snapshots with these milestones.
- Do not use a `keep latest two only` policy: it supports resume but erases the
  architecture trajectory. Five sparse, whole-run milestones are the active
  storage/observability compromise.

### Real milestone smoke result

- Hardware: one RTX PRO 6000 Blackwell Server Edition.
- Workload: real Qwen3-1.7B baseline surgery, 16 deterministic packed
  sequences at length 512, gradient accumulation 8, and two optimizer steps.
- Step 1/2 and step 2/2 each emitted a train metric, validation metric, and
  compact checkpoint at the configured 50% and 100% milestones.
- Training-loop wall time was 2.254 seconds. The two validation losses were
  0.540156 and 0.540979.
- Each checkpoint occupied approximately 289 MiB: 100,676,682 bytes of
  trainable state, 201,368,368 bytes of optimizer/RNG/sampler state, plus a
  small manifest.
- Restarting the completed run loaded step 2 and exited in 0.001 seconds. It
  emitted no duplicate optimizer step or validation record, confirming the
  saved sampler cursor is honored at the end of the run.
- The smoke-only `sft.max_packed_sequences` cap is recorded in the run
  manifest and is absent from every production config.

### Config-loader portability finding

- A local no-PyYAML test exposed that the minimal fallback parser treated
  inline YAML lists as strings. The fallback now parses JSON-compatible inline
  lists, with tests covering both `train_layers: [10]` and checkpoint fractions.
- This did not affect the remote run because PyYAML was already installed, but
  fixing it prevents silent config drift in a reduced local environment.
