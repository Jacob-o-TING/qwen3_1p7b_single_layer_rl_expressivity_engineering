# Qwen3-1.7B Layer-10 SFT DDP And Compile Short-Benchmark Plan

Date: 2026-07-10

Status: approved planning artifact. Writing this document does not launch a GPU
job, install packages, regenerate data, or modify the active RL experiment wave.

## Objectives

Build a deterministic supervised fine-tuning path beside the existing GRPO
path, then use it to answer one narrow performance question before committing
to custom kernels:

1. How fast is the existing whole-layer baseline in eager and compiled modes?
2. How much overhead does the active high-capacity SHS variant add?
3. How much of that overhead can `torch.compile` remove?

The first benchmark matrix is deliberately limited to four cases:

| Case | Architecture | Execution mode |
|---|---|---|
| A | Layer-10 whole-layer baseline | eager |
| B | Layer-10 whole-layer baseline | `torch.compile` |
| C | Layer-10 SHS (`32 x 32`, generator hidden `2048`, add rank `64`) | eager |
| D | Layer-10 SHS (`32 x 32`, generator hidden `2048`, add rank `64`) | `torch.compile` |

TriGLU and OFT remain part of the production SFT wave, but they are outside the
first compile benchmark so that the initial profiling question stays small.

## Non-Negotiable Comparison Contract

- Use the same Qwen3-1.7B-Base checkpoint and model revision in all cases.
- Use seed `20260707` for model initialization, data order, and distributed
  sampler construction.
- Replay the same pre-tokenized batches in all four benchmark cases.
- Keep global token budget, sequence packing, micro-batch size, gradient
  accumulation, dtype, optimizer, learning rate, and loss mask identical.
- Apply model surgery before DDP wrapping and before `torch.compile`.
- Preserve the existing architecture initialization invariants:
  - SHS grid-generator output weight and bias are exactly zero at initialization.
  - SHS multiply and additive deltas retain separate deterministic shuffle maps.
  - TriGLU has an exact identity multiplier at initialization.
  - OFT starts at identity and leaves its wrapped base SwiGLU weights frozen.
- Audit trainable names and parameter counts before every run.
- Do not compare throughput until the initial loss and a fixed-batch loss agree
  between eager and compiled modes within a recorded BF16 tolerance.
- Do not change batch size independently per architecture merely to make a
  slower architecture appear faster. An additional maximum-throughput sweep
  may be reported separately after the controlled comparison.

## Proposed File Layout

Keep SFT separate from the veRL command path while reusing config loading,
seeding, model surgery, freeze policy, and data provenance utilities:

```text
configs/sft/
  base_qwen3_1p7b_layer10_sft.yaml
  layer10_whole_layer_baseline_sft.yaml
  layer10_whole_layer_shs_sft.yaml
  layer10_whole_layer_triglu_side_ffn_sft.yaml
  layer10_whole_layer_oft_sft.yaml

src/qwen_single_layer_rl/sft/
  data.py                 # schema validation, tokenization, labels, packing
  distributed.py          # rank-aware deterministic batch schedule
  trainer.py              # BF16 DDP training and validation loop
  checkpoint.py           # custom-architecture-aware save/resume
  benchmark.py            # warmup, timed steps, profiler/compile metrics

scripts/
  launch_sft_single_node.sh
  launch_sft_ordered_variants.sh
  launch_sft_compile_short_benchmark.sh

tests/
  test_sft_data.py
  test_sft_distributed_order.py
  test_sft_loss_and_compile.py
```

Names may be consolidated where doing so removes trivial files without mixing
SFT logic into the GRPO/veRL launcher.

## Phase 1: Data And Loss Contract

1. Inspect the materialized NuminaMath records and identify the authoritative
   full chain-of-thought solution field. Do not silently train on the extracted
   final answer stored for the RL reward verifier.
2. Reuse the already decontaminated and seeded 50K row IDs. SFT must not perform
   a new independent sample.
3. Render each sample with the pinned Qwen3 tokenizer/chat template.
4. Mask prompt, padding, and packing-boundary tokens with label `-100`; compute
   cross-entropy only on assistant solution tokens.
5. Implement deterministic greedy packing into a fixed sequence length. Store a
   packing manifest containing source row IDs and offsets for every packed item.
6. Split training and validation before packing and record hashes of both
   source-order manifests.
7. Add tests for label masking, EOS handling, no cross-example label leakage,
   deterministic packing, truncation accounting, and row-order preservation.

The first performance run should use packed sequences. An unpacked dynamic-
padding run is useful only as a separately labeled packing ablation.

## Phase 2: Four-GPU DDP Trainer

Use one complete model replica per GPU and DDP rather than FSDP for the first
SFT implementation. Qwen3-1.7B fits comfortably on each 96 GB RTX PRO 6000,
while only the selected layer and variant modules require optimizer state.

Required behavior:

- Launch with `torchrun --nproc_per_node=4`; permit `1` for local smoke tests.
- Use BF16 autocast, AdamW, gradient accumulation, gradient clipping, and a
  configurable scheduler/warmup.
- Start with two SFT epochs; permit a third epoch only as a declared extension.
- Prefer DDP with `find_unused_parameters=False` after verifying every declared
  trainable parameter receives a gradient.
- Use a deterministic global batch schedule, then assign disjoint slices to
  ranks. Do not let each rank or variant create an independent permutation.
- Record global sample IDs for the first batches and hash the complete epoch
  schedule so equality across variants is directly auditable.
- Reduce token-weighted loss and token counts across ranks correctly.
- Save optimizer, scheduler, RNG, epoch, global step, sampler cursor, packing
  manifest hash, architecture config, and trainable-parameter audit for resume.
- Resume at the next exact global batch without repeating or skipping samples.
- Keep evaluation and checkpoint cadence step-based and identical across
  variants.

Gradient checkpointing should be configurable, not automatically enabled. On a
96 GB card it may reduce throughput without providing useful batch capacity;
the controlled benchmark should initially leave it off unless the chosen fixed
batch does not fit.

## Phase 3: Ordered Production SFT Wave

After smoke and determinism gates pass, preserve the architecture order already
used by the RL project:

1. SHS.
2. Layer-10 whole-layer baseline.
3. TriGLU side FFN.
4. OFT on SwiGLU.

Each run performs train, validation, checkpoint export, and the existing final
evaluation handoff before the next run starts. Large checkpoints and generated
predictions remain outside Git.

## Phase 4: Four-Case Short Benchmark

### Fixed workload

- Hardware for the first measurement: the available single RTX PRO 6000.
- Use one fixed packed sequence length and one fixed micro-batch that fits all
  four cases without gradient checkpointing.
- Replay identical in-memory/pre-tokenized batches.
- Run optimizer steps, not forward-only timing, so backward and optimizer cost
  are represented.
- Use the same trainable-parameter policy as the corresponding SFT run.

### Timing protocol

For each case:

1. Recreate the model from the same base checkpoint and seed.
2. Verify initialization and trainable-parameter audit.
3. Run one untimed correctness step on the same fixed batch and record loss.
4. For compiled cases, record compilation/cold-start wall time separately.
5. Run at least 5 warmup optimizer steps after compilation stabilizes.
6. Time 20-50 optimizer steps with CUDA synchronization at measurement
   boundaries, using the exact same batch sequence in every case.
7. Repeat the timed section if coefficient of variation exceeds 5%.

### Required metrics

- Cold initialization time and compile time.
- Median, mean, p10, and p90 optimizer-step time.
- Assistant training tokens per second and total non-padding tokens per second.
- Forward, backward, optimizer, and data-wait time where profiler overhead is
  not included in the headline timing.
- Peak allocated and reserved GPU memory.
- GPU utilization and achieved power during the timed window.
- Initial/final loss, gradient norm, and eager-vs-compile loss delta.
- Dynamo graph breaks, recompilations, and generated Inductor kernel count.
- SHS overhead versus baseline in eager mode and in compiled mode.
- Compile break-even steps: compile time divided by steady-state time saved.

The benchmark report must distinguish cold wall time from steady-state training
time. A compile path that is faster after warmup but expensive to initialize is
still useful for a multi-epoch run; the break-even point makes that explicit.

## Phase 5: Kernel Decision Gate

Do not implement a custom kernel merely because SHS is novel. Use the measured
profile to choose the next level of optimization:

1. Keep eager PyTorch if SHS overhead is small enough for the experiment wave.
2. Use `torch.compile` if it removes most elementwise/indexing overhead without
   graph breaks or numerical drift.
3. Prototype a Triton fusion only if a stable SHS operation remains at least
   roughly 10-15% of total step time, or its intermediate tensors materially
   constrain batch size.
4. Consider CUDA/CUTLASS only after Triton is insufficient and the architecture
   has demonstrated useful quality/scaling behavior.

Likely optimization targets are SHS block-ID gathers, generated multiply/add
delta application, and elimination of materialized intermediate tensors. Do not
replace cuBLAS GEMMs without profile evidence. For TriGLU, first target fusion
around activation, gating, and residual application rather than rewriting the
GEMMs themselves.

Any Triton/CUDA implementation must retain a PyTorch reference path and pass
forward, backward, initialization, determinism, mixed-precision, odd-shape, and
distributed smoke tests before performance results are accepted.

## Dependencies And Installation Policy

The minimal SFT path should not require a new training framework. Prefer the
existing environment plus PyTorch DDP:

- Required and likely already present: `torch`, `transformers`, `datasets`,
  `pyyaml`, `pandas`, `pyarrow`, and `safetensors`.
- Provided by PyTorch: DDP, `torchrun`, profiler, Dynamo, Inductor, and
  `torch.compile`.
- Optional for logging only: TensorBoard or the existing JSONL logger.
- Optional after profiling: `triton` (normally installed with the CUDA PyTorch
  wheel) for an SHS kernel prototype.
- Not required initially: TRL, DeepSpeed, FlashAttention, Liger Kernel, a new
  evaluator framework, or a custom CUDA extension toolchain.

Before installation, run an import/version audit inside the isolated AutoDL
environment and write the result to the run manifest. Install only packages
shown to be missing. Pin every newly installed package and preserve the prior
environment lock/audit; do not upgrade Torch, Transformers, CUDA libraries, or
vLLM as a side effect of adding the SFT path.

Expected dependency outcome: no mandatory heavyweight download. At most, a
small missing utility such as `safetensors` or a logger may need installation.

## Verification Gates Before GPU Timing

- CPU/unit tests pass for SFT masking, packing, schedule hashes, and resume.
- A tiny model or tiny batch completes forward/backward and checkpoint resume.
- All four architecture configs pass trainable-name and parameter-count audits.
- SHS and TriGLU remain exact no-ops at initialization; OFT remains identity.
- Additive and multiplicative SHS shuffle maps remain distinct and reproducible.
- Rank slices are disjoint and reconstruct the exact global order.
- Eager and compiled fixed-batch losses agree within recorded BF16 tolerance.
- No benchmark/eval sample occurs in the decontaminated 50K training rows.
- The exact benchmark command and environment manifest are saved before launch.

## Deliverables

- Reusable deterministic DDP SFT pipeline and four SFT configs.
- Ordered four-variant launcher with train-to-eval chaining.
- Four-case eager/compile benchmark script and machine-readable results.
- Short Markdown benchmark report with timing, memory, graph-break, and
  compile-break-even tables.
- Updated dependency/environment manifest and AutoDL run instructions.
- Unit and single-GPU smoke coverage for data, model surgery, training, resume,
  and compile compatibility.

## Estimated Engineering Schedule

- Data/loss contract and deterministic packing: 2-4 hours.
- DDP trainer, checkpoint/resume, and variant configs: 3-5 hours.
- Tests, launchers, manifests, and train-to-eval chaining: 2-4 hours.
- Four-case single-GPU smoke/benchmark and report: 1-3 hours of engineering
  plus the measured GPU runtime.

Expected total to a production-ready first version: approximately 8-14 focused
engineering hours, with the first runnable single-GPU smoke available earlier.

## Launch Hold

After this planning commit, stop. Do not install dependencies, modify the remote
environment, or launch the SFT/benchmark jobs until the project owner returns and explicitly
asks to continue.
