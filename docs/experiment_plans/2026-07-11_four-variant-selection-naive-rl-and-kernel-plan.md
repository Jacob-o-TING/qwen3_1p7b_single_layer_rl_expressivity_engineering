# 2026-07-11 Four-Variant Selection, Naive RL, And Kernel Plan

Date: 2026-07-11

Status: approved research direction and planning artifact. This document does
not authorize a new GPU wave or finalize future run IDs. Exact configs, run
names, seeds, and launch gates must be approved after the active SFT wave.

## Objective

Compare four Layer-10 SFT schemas under the current controlled contract, select
one architecture schema for a direct comparison against naive single-layer RL,
then optimize the selected schema's rollout and actor-compute paths enough to
make RL experimentation practical.

The four schemas are:

1. SHS over the original SwiGLU, with original SwiGLU and the rest of Layer 10
   trainable.
2. Naive whole-layer baseline with no adapter or added architecture.
3. TriGLU with side FFN over the original SwiGLU, with both paths and the rest
   of Layer 10 trainable.
4. OFT on SwiGLU while the original SwiGLU weights remain frozen and the other
   Layer-10 components follow the approved training contract.

## Stage 1: Finish The Four-Variant SFT Comparison

Primary evidence:

- MATH-500, GSM8K, OlympiadBench, and AMC Average@32;
- unweighted four-benchmark math average;
- per-item paired outcomes wherever prompts and decoding are identical;
- greedy AMC pass@1 as a decoding diagnostic, never as a replacement for the
  primary sampled AMC score;
- final validation loss and training trend;
- trainable parameter count, wall time, peak memory, and evaluation throughput;
- extraction failures, cap-proxy responses, response length, and audited
  failure modes.

Selection must not be based on MATH-500 alone. The winner should improve the
four-task aggregate or a pre-declared priority profile without introducing a
material regression elsewhere. Small score gaps should be treated as uncertain
until paired analysis and, where justified, repeated evaluation are complete.

Interpretation gates:

- SHS and TriGLU both beat baseline: supports added expressivity as a useful
  factor.
- SHS beats both baseline and TriGLU: supports SHS-specific update geometry or
  inductive bias rather than parameter count alone.
- OFT beats baseline: supports constrained update geometry without requiring
  more freely trainable SwiGLU weights.
- all three variants beat baseline: suggests the naive parameterization or its
  optimization geometry is the main SFT bottleneck.

## Stage 2: Select The RL Comparison Schema

The selected architecture should be compared directly with a naive Layer-10 RL
baseline. Both RL runs must share:

- the same Qwen3-1.7B base revision and Layer-10 scope;
- the same decontaminated NuminaMath 50K manifest and deterministic prompt order;
- the same initialization, sampling, and trainer seeds;
- identical rollout prompts, group membership, response caps, reward verifier,
  and decoding parameters;
- identical GRPO objective, KL coefficient, clip range, optimizer schedule,
  effective batch, update count, checkpoint gates, and evaluator revisions;
- identical hardware count and an explicitly recorded concurrency policy.

The naive control trains the unmodified complete Layer 10 under GRPO. The
selected schema preserves its approved architecture-specific train/freeze
contract. Common sampled rollouts should be reused only if doing so remains
mathematically valid for the compared policy updates; otherwise, deterministic
prompt/group schedules should be shared while each policy generates its own
on-policy responses.

Recommended launch sequence after naming approval:

1. CPU/static and one-batch numerical gates.
2. Short rollout and update smoke tests for both schemas.
3. A small matched RL pilot with trend and throughput checkpoints.
4. One complete matched run only after the pilot passes economic and numerical
   gates.
5. Additional seeds only after the first complete comparison establishes that
   the effect size warrants replication.

Use the staged budget gates recorded in
`2026-07-11_sft-loss-eval-misalignment-and-budgeted-rl-record.md`: 2, 20, 50,
98, 196, and 391 global 512-prompt batches. The 98-batch gate is approximately
one pass through 50K unique prompts; only matched schemas that remain promising
should advance to half and full paper-shaped budgets.

## Stage 3: Profile Before Writing Custom Kernels

Profile the selected architecture in two distinct workloads:

1. Rollout inference: prompt prefill plus autoregressive decode with KV cache.
   Decode is often dominated by small-batch matrix-vector work, launch overhead,
   memory traffic, and custom-layer dispatch.
2. Actor training: packed-sequence forward, loss computation, backward, and
   optimizer update. This path is more GEMM-heavy and has different fusion and
   recomputation tradeoffs.

Collect at minimum:

- end-to-end rollout tokens per second and time per completed response;
- prefill tokens per second and decode tokens per second separately;
- actor forward, backward, optimizer, and weight-sync wall time;
- kernel launch counts, graph breaks, achieved occupancy, memory bandwidth, and
  temporary allocation peaks;
- eager versus `torch.compile` full-graph/partial-graph behavior;
- time outside GPU kernels, including scheduler, tokenizer, verifier, and data
  movement overhead.

No custom kernel should be started solely because a custom architecture exists.
The profiler must identify a stable, material bottleneck first.

### Required Runtime Benchmark Matrix

Run matched measurements before selecting the production rollout path. The
minimum matrix is:

| Schema | Eager HF | Compiled HF | vLLM reference graph | vLLM optimized custom graph |
|---|---:|---:|---:|---:|
| Naive Layer 10 | required | required | required | not applicable |
| SHS | required | required | required after onboarding | required only if selected |
| TriGLU | required | required | required after onboarding | required only if selected |

OFT may be added if its SFT quality warrants RL selection. Each applicable cell
must distinguish cold initialization/compilation from warm steady state and
must use the same checkpoint, prompt IDs, response cap, decoding parameters,
and generated-token accounting.

Measure three nested workloads:

1. **Operator and decode microbenchmarks.** Measure the custom MLP alone, one
   cached decode step, and prompt prefill over token-batch sizes 1, 8, 16, 32,
   and 64. These measurements isolate graph breaks, launch overhead, temporary
   tensors, and kernel ceilings.
2. **Fixed-token generation benchmark.** Generate a preregistered number of
   tokens for a fixed prompt panel, separately reporting prefill and decode
   throughput, cap-hit behavior, and per-request latency distribution.
3. **One complete GRPO global batch.** Include rollout scheduling, four samples
   per prompt, verifier/reward work, log-probability computation, actor forward
   and backward, optimizer update, weight synchronization, and checkpoint/log
   overhead. Run at least naive eager, naive compiled, naive vLLM, selected-
   schema eager, selected-schema compiled, and selected-schema vLLM.

A decode-only result may be multiplied by the already measured generated-token
count to form a provisional rollout estimate after continuous batching is
working. That estimate must be labelled as a projection, not a measured GRPO
wall time. It does not capture prefill, length variance, scheduler bubbles,
stragglers, verifier work, actor updates, weight synchronization, compilation,
or checkpoint overhead. The final rental-cost decision requires the complete
one-batch measurement and should extrapolate full budgets only from its
rollout/update component timings.

For the one-batch comparison, record both total wall time and this decomposition:

```text
engine/model load
compile and CUDA-graph warmup
rollout prefill
rollout decode
reward and answer extraction
old-policy/reference log probabilities
actor forward/backward/optimizer
actor-to-rollout weight synchronization
checkpoint and orchestration
```

Do not count compile warmup as steady-state throughput, but do include its
amortized cost when estimating a 2-, 20-, or 50-batch pilot.

## Stage 4: Kernel And Runtime Tracks

### Rollout Runtime Track

- Replace runtime-only model surgery with explicit Hugging Face custom model
  classes and configs. Record the architecture variant, target layer,
  dimensions, SHS shuffle maps/seeds, and checkpoint parameter names in the
  exported model contract.
- First try the pinned vLLM Transformers backend with tensor parallel size 1.
  If correctness is preserved but performance is inadequate, register an out-
  of-tree vLLM model plugin that reuses the native Qwen3 attention/model path
  and replaces only the selected Layer-10 MLP.
- Retain one complete rollout replica per GPU. Continuous batching is a
  scheduler/runtime capability, but it becomes usable only after vLLM can
  instantiate, load, flatten-token execute, compile/capture, and synchronize
  the custom model graph.
- Implement correct custom weight loading and checkpoint-to-rollout weight
  synchronization. The first correct version may rebuild or reload an engine at
  an update boundary; an in-place update path is a later throughput optimization.
- Preserve paged KV cache, continuous batching, prefix/prompt handling, stop
  conditions, and deterministic sampling contracts.
- Validate dynamic flattened token counts, CUDA Graph/compile sizes, and
  chunked-prefill behavior before increasing `max_num_seqs` or
  `max_num_batched_tokens`.
- Benchmark against Hugging Face generation and the existing baseline vLLM
  path at matched prompts, lengths, seeds, and batch pressure.

### Forward/Backward Kernel Track

- First attempt graph capture and operator decomposition compatible with
  `torch.compile` and Inductor.
- Use Triton or a suitable CUTLASS-backed extension only for hot regions that
  remain unfused or materialize expensive intermediates.
- Keep large dense GEMMs on cuBLAS/cuBLASLt or vLLM parallel-linear operators
  unless profiling proves that replacing them is necessary.
- For SHS, first fuse the generator activation/reshape path. Then prototype a
  modulated projection kernel that combines the base projection and dynamic
  multiplicative grid without 32 sliced GEMMs or materialized `base_col`
  tensors. Implement the additive low-rank path as a two-stage block reduction:
  token-by-column-block rank features followed by an output projection modulated
  by the per-token grid.
- SHS additive and multiplicative shuffle maps remain separate. Inference-only
  weight prepacking may group channels by deterministic block ID after each
  actor-to-rollout synchronization, but it must maintain distinct layouts for
  the two maps and preserve the original parameter/checkpoint layout.
- For TriGLU, leave the six dense side-path projections on tuned GEMM kernels.
  First use Inductor to fuse casts and elementwise operations, then consider a
  Triton fusion for the three-way activated product and a second fusion for the
  final `tanh`, scale, and trunk multiplier. Do not expect these epilogue
  fusions to remove the cost of the additional GEMMs.
- For OFT, avoid rebuilding orthogonal transforms per generated token; cache
  transforms for an immutable rollout weight snapshot and separately design the
  differentiable training path.
- Evaluate activation checkpointing or selective recomputation only after
  measuring its compute-memory tradeoff.

Start with inference-only custom kernels because rollout generation is the
expected dominant cost. Keep the actor-training path on eager/compiled PyTorch
until profiling shows that custom forward/backward kernels would materially
change the pilot budget. A custom backward implementation requires gradient
parity for inputs, original Layer-10 weights, grid-generator weights, SHS
low-rank factors/scales, and TriGLU side-path parameters.

### Recommended Engineering Order

1. Freeze and test the reference algebra and checkpoint naming contract.
2. Package explicit SHS and TriGLU Hugging Face model classes without changing
   numerical behavior.
3. Obtain continuous batching through the vLLM Transformers backend at TP=1.
4. Add a native out-of-tree vLLM model only if the generic backend leaves a
   material performance gap.
5. Run the complete runtime benchmark matrix and one-batch GRPO comparison.
6. For TriGLU, try `torch.compile` before writing Triton epilogues.
7. For SHS, prioritize an inference Triton prototype for the shuffled dynamic
   projection; write backward kernels only after RL quality and profile gates.

Provisional engineering estimates, excluding queue/rental availability, are
one to two engineering days for packaging plus a first continuous-batching
prototype, one to three GPU-debugging days for useful TriGLU inference fusion,
and three to seven GPU-debugging days for an SHS inference kernel. A complete
SHS backward kernel may require an additional one to two weeks and is not a
default commitment.

## Durable Pause, Clone, And Resume Workflow

The production SFT wave may be stopped while a cloned instance is used for
kernel and continuous-batching development. This is artifact-level resume, not
process-memory suspension: shutdown destroys the PID, Python queues, CUDA
context, and `screen` session. Recovery depends on durable training checkpoints,
EvalScope JSONL caches, evaluation receipts, deterministic seeds, and the
ordered launcher.

Before shutdown:

1. terminate the active trainer/evaluator gracefully and wait for the main
   `screen` session to exit;
2. run `sync` and verify zero GPU compute/memory use;
3. record each partial prediction/review row count, unique index count, and
   SHA-256;
4. verify final training checkpoints and completed evaluation receipts;
5. run the cache-resolution and resume-launcher unit/static tests;
6. commit and push source, scripts, plans, and records, never model/data/output
   artifacts.

Resume with:

```bash
cd /root/autodl-tmp/qwen3_1p7b_single_layer_rl
bash scripts/launch_sft_ordered_resume.sh
```

The ordered launcher must:

- skip training only after `resolve-checkpoint` verifies an exact final
  checkpoint;
- otherwise resume training from the latest checkpoint and deterministic
  sampler/RNG state;
- skip evaluation only after validating its hashed completion receipt;
- otherwise select the EvalScope cache with the greatest unique-row coverage,
  not merely the newest timestamp;
- pass separate `main-use-cache` and `amc-use-cache` directories so interruption
  during either phase is resumable;
- keep aborted or lower-coverage timestamp directories for audit but exclude
  them from the authoritative result.

At a training interruption, work since the last intentionally sparse checkpoint
may need to be repeated. At an evaluation interruption, only in-flight requests
not yet appended to JSONL may be repeated. Completed cached rows must not be
regenerated.

The cloned instance is a separate kernel-development environment. It must use
new run IDs and output roots for compile, vLLM, continuous-batching, and rollout
benchmarks. It must not write into
`sft_ordered_20260711_sft50k_v1`. Local Git plus the pushed private repository
remain the source of truth for code; the original instance remains the source
of truth for the paused production artifacts until they are archived locally.

## 2026-07-11 SHS Multiplicative Kernel Execution Update

The approved `shs_modproj_triton_decode_bench_20260711_v1` implementation and
controlled microbenchmark completed on the kernel-development clone. The
inference-only Triton kernel fuses the base projection and SHS multiplicative
map; it does not optimize the independent additive low-rank path and implements
no backward. The production module keeps the reference backend by default and
requires explicit inference opt-in for Triton.

All 32 FP32/BF16 parity cases passed across gate, up, down, an odd-tail shape,
and token batches 1/8/16/32. Twelve warm BF16 timing cases passed without
fallback. Relative to the original 32-sliced-GEMM multiplicative reference,
the measured operator speedup ranged from 3.62x to 21.01x. This is an operator
microbenchmark, not an end-to-end decode, continuous-batching, or GRPO result.

The vLLM 0.10.2 static scaffold now provides an explicit Hugging Face config
and model class plus a lazy `vllm.general_plugins` registry entry. Registry
discovery and checkpoint-key contract tests passed. Engine loading remains
blocked on constructing an exported SHS model directory from a real checkpoint
and validating meta/buffer placement, flattened-token execution, attention
integration, weight loading, CUDA graphs, and weight synchronization. See
`docs/experiment_records/2026-07-11_shs-modproj-triton-and-vllm-scaffold-record.md`
for exact evidence and the next commands.

## Definition Of The Built SHS Runtime

The project continues beyond the first Triton operator benchmark. The final
engineering objective is a reproducible SHS training-and-rollout runtime, not a
standalone kernel. Advancement follows these gates in order:

1. **Full-model integration parity.** Export a disposable real SHS checkpoint;
   load it through the explicit custom HF class; verify exact loaded, missing,
   and unexpected key sets; enable the Triton multiplicative backend; compare
   Layer-10 outputs, model logits, greedy generations, and seeded sampled
   generations against the reference model.
2. **vLLM eager onboarding.** Load the export through the registered vLLM 0.10.2
   path at TP=1 and `--enforce-eager`; validate persistent shuffle-buffer device
   placement, flattened token counts, PagedAttention/KV cache, stop conditions,
   and explicit Triton dispatch.
3. **Continuous batching.** Remove eager-only restrictions after parity; test
   request pressure 1/8/16/32, heterogeneous prompt/response lengths, chunked
   prefill, compile/CUDA-graph sizes, memory, throughput, and latency. Compare
   naive baseline vLLM, SHS reference vLLM, and SHS Triton vLLM.
4. **Rollout-only GRPO smoke.** Run a tiny deterministic prompt group with four
   on-policy samples, reward extraction, response-cap accounting, and compact
   traces, but no actor update. Confirm backend dispatch and no silent fallback.
5. **SFT optimizer-step smoke.** First prove the unchanged reference training
   path still matches. A kernel-enabled label requires Triton forward with
   gradient-correct reference-recompute or custom backward; compare loss,
   gradients, updated parameters, optimizer state, and checkpoint resume.
6. **Complete one-batch GRPO smoke.** Include rollout, reward, old/reference
   log probabilities, actor loss/backward/update, custom-weight synchronization,
   checkpoint, and deterministic resume. Compare eager, compiled, and vLLM
   timing components against the naive Layer-10 control.
7. **Economic gates.** Only after the complete batch passes may the 2-, 20-,
   50-, and 98-batch ladder be considered. Extrapolate from measured component
   timings, never from the isolated operator speedup alone.

The independent additive SHS path remains reference code during the earliest
runtime gates. It should receive its own two-stage Triton prototype only after
full-model profiling shows that it remains a material bottleneck. Training
backward optimization is a separate milestone and must not block inference
rollout validation unless the one-batch actor profile shows it dominates cost.

### Proposed Next Run Names

These names reserve separate output roots and prevent operator, rollout, SFT,
and GRPO evidence from being mixed:

| Gate | Run ID | Screen |
|---|---|---|
| Full-model parity/export | `shs_fullmodel_kernel_parity_20260712_v1` | `qwen_shs_fullmodel_parity_20260712_v1` |
| vLLM continuous-batched rollout | `shs_vllm_contbatch_rollout_smoke_20260712_v1` | `qwen_shs_vllm_rollout_smoke_20260712_v1` |
| SFT optimizer-step smoke | `shs_triton_sft_step_smoke_20260712_v1` | `qwen_shs_triton_sft_smoke_20260712_v1` |
| Complete GRPO one-batch smoke | `shs_triton_grpo_onebatch_smoke_20260712_v1` | `qwen_shs_triton_grpo_smoke_20260712_v1` |

Each run must write a human-readable log and machine-readable manifest under
its own `runs/` subtree. The names are planning reservations; launching a new
controlled wave still requires explicit approval after its exact config, seed,
prompt subset, response cap, backend, and checkpoint source are reviewed.

## Correctness Gates For Every Optimization

- exact initial no-op invariants for SHS and TriGLU;
- deterministic and distinct SHS additive/multiplicative shuffle maps;
- forward parity against the reference implementation at FP32 and BF16
  tolerances;
- backward/gradient parity for every trainable parameter group;
- generation parity under greedy decoding and deterministic sampled decoding;
- no change to data order, prompt grouping, reward extraction, or checkpoint
  semantics;
- no silent fallback that makes benchmark labels claim a custom kernel when the
  reference path actually ran;
- memory and throughput measurements include warmup, compilation, steady state,
  and end-to-end wall time separately.

## Decision Gates

Kernel work advances only if all of the following hold:

1. The chosen schema shows a scientifically meaningful advantage over the naive
   SFT baseline.
2. Its matched RL smoke test is numerically stable.
3. Profiling shows custom architecture compute is a material share of rollout
   or actor wall time.
4. A reference implementation and parity suite exist before optimization.
5. The projected speedup is large enough to alter RL iteration time or rental
   cost, not merely a microbenchmark.

The final research comparison should report both model quality and system cost:
benchmark scores, reward/training trends, response-length behavior, tokens per
second, GPU-hours, peak memory, and the speedup relative to the unoptimized
naive single-layer RL baseline.

## 2026-07-12 SFT Wave Decision

The complete two-epoch SFT wave is now recorded in
`docs/experiment_records/2026-07-12_sft50k-four-variant-completion-record.md`.
The four-task averages were 43.09% SHS, 43.17% whole-layer baseline, 42.85%
TriGLU, and 13.04% OFT. These results do not support a broad SFT winner among
the first three variants; the 0.32-point range is too small for a categorical
single-run claim. They do expose different benchmark profiles and a decisive
loss/evaluation mismatch in the implemented OFT configuration.

The next matched RL wave therefore keeps the whole-layer baseline as the
mandatory control, advances SHS as the primary custom candidate because its
MATH-500 result and runtime integration are strongest, and retains TriGLU as a
secondary candidate pending a larger greedy/modal-path check. The current OFT
configuration is held for parity and checkpoint-curve diagnostics rather than
advanced directly. All future comparisons must preserve the canonical data
order, seed policy, response caps, and evaluator protocol across variants.

## 2026-07-12 Runtime Execution Status

The four approved runtime smokes were executed on the RTX PRO 6000 clone. The
disposable export, vLLM Transformers-backend loading, Paged KV cache,
continuous batching, reference SFT step/resume, and tiny GRPO update/rebuild
paths are operational. The strict built boundary is nevertheless not met:

| Run | Status | Decisive result |
|---|---|---|
| `shs_fullmodel_kernel_parity_20260712_v1` | failed | Layer-10 mixed parity and greedy parity passed, but full-logits cosine 0.999755 was below the preregistered 0.9999 threshold. |
| `shs_vllm_contbatch_rollout_smoke_20260712_v1` | passed | TP=1 V1 eager engine, Paged KV, HF token parity, 3/3 Triton receipts, and pressure 1/8/16/32 passed. |
| `shs_triton_sft_step_smoke_20260712_v1` | passed | Honest reference-training integration, optimizer step, checkpoint load, and deterministic resumed step passed; no Triton training claim. |
| `shs_triton_grpo_onebatch_smoke_20260712_v1` | passed | G=4 rollout, reward/advantage, old/ref log probabilities, actor update, checkpoint resume, updated export, engine rebuild, and post-sync Triton dispatch passed. |

No 20-, 50-, or full-budget wave is authorized from these results. The next
correctness task is to reduce the remaining deep-logit drift without changing
the preregistered threshold after observing the result. Exact evidence is in
`docs/experiment_records/2026-07-12_shs-runtime-execution-record.md`.

## 2026-07-13 TriGLU Topology Correction And Shared Runtime Optimization Plan

Status: **approved planning update only**. It records the architecture identity,
measured bottleneck, owner proposals, and engineering proposals. It does not
alter the active six-GPU GRPO wave, authorize a new run, or reserve final run,
screen, config, or output names. Those identities require separate owner
approval before launch.

### Architecture identity correction

The active architecture historically called `TriGLU with side FFN` is not a
standard standalone SwiGLU and is not the simpler vanilla side-FFN multiplier.
It is the FineWeb ablation's **compressed side TriGLU with an embedded
GeLU-FFN-derived third factor**:

```text
base trunk: original Qwen SwiGLU

x -> side_down 2048 -> 512
     |-- side_value 512 -> 2048
     |-- side_gate  512 -> 2048 -> SiLU
     '-- side_ffn_1 512 -> 2048 -> GeLU
                         -> side_ffn_2 2048 -> 2048 -> SiLU
     three-way product -> side_up 2048 -> 6144
                       -> 1 + scale * tanh(side)
                       -> multiply original SwiGLU intermediate
                       -> original down projection
```

It contains six side Linears and begins as an exact identity because
`side_up.weight` and `side_up.bias` are zero. The active checkpoint, optimizer
state, run ID, and evaluation identity remain unchanged. All later records must
call it `compressed side TriGLU with embedded GeLU-FFN third factor`; the short
historical label may appear only as an alias.

The owner-intended architecture is a distinct existing FineWeb ablation:

```text
base trunk: original Qwen SwiGLU

x -> side_down 2048 -> 512
  -> GeLU(side_ffn_1 512 -> 2048)
  -> GeLU(side_ffn_2 2048 -> 512)
  -> side_up 512 -> 6144
  -> 1 + scale * tanh(side)
  -> multiply original SwiGLU intermediate
  -> original down projection
```

This is the **vanilla GeLU-FFN multiplier side branch**. It has four side
Linears and must be introduced under a new architecture/config/checkpoint/run
identity after explicit naming approval. It may reuse `side_dim=512` and
`side_hidden=2048`, but no code or loader may reinterpret a compressed-TriGLU
checkpoint as this simpler graph.

### Compute and production timing evidence

For the current dimensions, side-only matrix work is:

```text
compressed side TriGLU:
  2048*512 + 3*(512*2048) + 2048*2048 + 2048*6144
  = 20.97M MAC/token

vanilla side FFN:
  2048*512 + 512*2048 + 2048*512 + 512*6144
  = 6.29M MAC/token
```

The simpler side FFN therefore has approximately **3.33x less side matrix
work**. The current 20.97M side MAC/token is 55.6% of one Qwen SwiGLU layer's
approximately 37.75M MAC/token, but only about 1.98% of the 28-layer model's
aggregate FFN matrix work and a still smaller fraction after attention is
included. This arithmetic never counts the original SwiGLU as extra work.

Matched medians from the active production wave exposed an execution-path
bottleneck rather than an actor-compute explosion:

| Component | Baseline steps 16-20 | Compressed TriGLU steps 38-42 | Ratio |
|---|---:|---:|---:|
| vLLM rollout | 58.85 s | 137.92 s | 2.34x |
| actor update | 276.96 s | 282.85 s | 1.02x |
| complete update | 655.45 s | 766.00 s | 1.17x |

The actor path is consistent with the small whole-model theoretical overhead.
The rollout penalty is primarily an implementation tax: six FP32 side GEMMs,
explicit casts, unfused activation/elementwise kernels, generic Transformers
execution inside vLLM, and `enforce_eager=True`, repeated at every
autoregressive decode token. The earlier matched pressure-64 measurement of
2,872.1 tok/s/GPU versus 5,442.1 for vanilla Qwen, or 52.8% of vanilla,
provides an independent anchor for the same diagnosis.

### Owner proposals preserved

1. **Use pure BF16 for custom architecture math.** SHS and both side-branch
   candidates should receive separately named BF16 paths if parity holds.
   BF16 changes storage/execution policy, not the mathematical topology.
2. **Keep ordinary SwiGLU execution ordinary.** The native Qwen trunk should
   retain its normal optimized vLLM/cuBLASLt path. Only architecture-owned
   side/delta math should use a special kernel or wrapper.
3. **Localize special execution.** Tokens cannot and need not avoid cuBLAS;
   cuBLAS/cuBLASLt is the normal GEMM path. The target split is native trunk,
   localized custom side math, then native down projection.
4. **Generalize improvements to SHS.** BF16, native-trunk preservation,
   compilation, fusion, concurrency, and shape-dependent dispatch should be
   evaluated for both TriGLU-family and SHS paths rather than built as one-off
   architecture patches.
5. **Restore the intended simpler experiment.** The vanilla side-FFN
   multiplier is a new controlled architecture candidate, not a rename or
   repair of the current compressed-TriGLU wave.

### Engineering proposals and dependency order

#### 1. Pure-BF16 reference paths

- Implement explicit `reference_fp32_custom` and `pure_bf16_custom` policies,
  satisfying PENDING-02 rather than silently casting historical checkpoints.
- Keep custom weights, inputs, intermediate activations, and returned deltas
  in BF16; report GEMM/optimizer accumulation and master-state dtypes
  separately.
- Use the same policy in actor and rollout. A BF16-only vLLM side paired with an
  FP32 actor is not an accepted optimization because it creates avoidable
  policy/log-probability drift.
- Preserve exact initial no-op, export/reload, backward, optimizer, checkpoint,
  and weight-sync behavior.

BF16 alone is estimated, not promised, to improve the current compressed-
TriGLU rollout by approximately 1.15-1.35x, moving the 137.92-second median to
roughly 102-120 seconds. This corresponds to approximately 3-6% complete-step
improvement, with about 8% as an optimistic upper bound. The estimate must not
be used as acceptance evidence.

#### 2. Preserve a native vLLM Qwen trunk

- Base the serving model on vLLM's native Qwen implementation and replace only
  the target Layer-10 MLP behavior.
- Retain native gate/up projection, fused SwiGLU activation, attention, KV
  cache, scheduler, and native down projection wherever the topology permits.
- Insert the BF16 custom side multiplier between the native SwiGLU intermediate
  and native down projection. A post-`base_mlp(x)` correction is mathematically
  invalid because the multiplier acts before `down_proj`.
- Emit resolved model class, target layer, dtype, backend, and
  `fallback=false` receipts. Generic `TransformersForCausalLM` is a historical
  control, not evidence that the native-trunk objective is complete.

#### 3. Remove eager-only execution after graph-safety gates

- Make custom operations compatible with vLLM compilation and CUDA Graph
  capture, then remove `enforce_eager=True` under a separately measured path.
- Record graph breaks, captured batch sizes, warmup/compile time, and steady
  decode throughput separately.
- Retain an eager fallback for correctness diagnostics; never silently label an
  eager execution as compiled.

Autoregressive decode repeatedly pays kernel-launch latency, so graph capture
may be at least as important as the dtype change.

#### 4. Pack projections without changing checkpoint semantics

- `side_value`, `side_gate`, and `side_ffn_1` consume the same `side_z` and can
  be packed into one `512 -> 6144` rollout GEMM, then split into three logical
  outputs.
- Keep canonical logical tensors and checkpoint keys stable during the first
  serving optimization. Construct a packed rollout buffer after load and after
  every actor-to-vLLM weight synchronization rather than changing training
  parameter ownership prematurely.
- Hash the packed buffer against its source tensors and reject stale packs.

#### 5. Fuse the elementwise chain

Fuse the applicable GeLU, SiLU, three-way product, `tanh`, scale/add, and
`trunk *= multiplier` work to reduce tiny launches and intermediate tensors.
Fusion must preserve the exact activation ordering and output dtype. It must
not turn the vanilla side FFN into compressed TriGLU or vice versa.

#### 6. Use shape-dependent hybrid dispatch

- Use a fused Triton or CUTLASS small-M path for low-occupancy decode shapes
  only if it beats the reference path end to end.
- Keep cuBLASLt for larger prefill, packed actor, and high-pressure shapes when
  it wins.
- Select by preregistered shape/backend rules and emit a dispatch receipt.
  One kernel is not expected to dominate every workload.

For SHS, the corresponding strategy is native base GEMM plus localized custom
delta math, BF16 grid/delta execution, fused multiplicative/additive correction
where parity allows, graph capture, and low-M/high-M hybrid dispatch. The
deterministic additive and multiplicative shuffle maps remain separate. Packed
maps/weights are rebuilt only after verified weight synchronization. The
strict-failed historical grouped SHS Triton path is not reauthorized by this
plan.

#### 7. Tune continuous-batching pressure after memory gates

Test `max_num_seqs=96/128` only after KV-cache, activation, cap, and OOM gates.
Larger live token batches may amortize side GEMMs and launch overhead for both
SHS and side-FFN variants. Throughput gains must be reported together with
response-length distribution, cap rate, overlap/straggler tail, and semantic
sampling drift; a changed output distribution is not a clean hardware speedup.

#### 8. Defer custom backward until profiling justifies it

The observed actor ratio is only 1.02x, so the initial training path should keep
reference PyTorch/cuBLASLt backward and test `torch.compile` before writing a
custom backward kernel. Re-profile after rollout optimization. A fused backward
becomes a priority only if actor forward/backward is then a material share of
iteration cost and passes full gradient/optimizer/resume parity.

### Staged execution and decision gates

1. Complete and preserve the active 98-step compressed-TriGLU and baseline
   wave unchanged; archive its exact architecture manifest and timing evidence.
2. Implement the vanilla side-FFN multiplier as a separately named exact-no-op
   architecture with independent config, loader, export, and tests.
3. Complete PENDING-02 for pure-BF16 SHS and compressed TriGLU, then extend the
   same dtype contract to the vanilla side-FFN candidate.
4. Establish native-Qwen-trunk serving paths before claiming that architecture
   FLOPs explain end-to-end throughput.
5. Test eager versus compiled/CUDA-graph execution under identical prompts,
   caps, seeds, concurrency, and checkpoints.
6. Add projection packing and elementwise fusion one change at a time, retaining
   an immutable reference cell and attribution for every speedup.
7. Add hybrid kernels only for profiler-confirmed shapes, then retune
   continuous-batching pressure.
8. Run the required HF/vLLM runtime and Eval Parity matrices plus one complete
   production-shaped GRPO batch before authorizing a new multi-step quality
   comparison.

The provisional combined target for the unchanged compressed-TriGLU math is
70-95 seconds per rollout after BF16, native trunk, and graph capture, compared
with the observed 58.85-second baseline. Side fusion may narrow the remaining
gap. This range is a planning hypothesis, not a launch promise. The simpler
vanilla side FFN has 3.33x less side matrix work but requires direct measurement;
its end-to-end speed must not be inferred by linearly dividing wall time.

### Correctness and reporting gates

- exact topology name and graph in every manifest and report;
- exact initial no-op for every side/delta variant;
- no missing, unexpected, or silently remapped checkpoint keys;
- actor/rollout dtype and version receipts, sampled-token log-probability drift,
  KL, ratio, and clipping checks;
- HF reference forward/backward and fixed-seed generation comparisons under
  preregistered BF16 and backend tolerances;
- native-route and custom-kernel dispatch receipts with `fallback=false`;
- unchanged dataset ledger, seeds, prompt/group order, caps, reward, checkpoint,
  and evaluator contracts across compared variants;
- cold load/compile, warm prefill, warm decode, actor, synchronization, and
  complete-update timing reported separately;
- memory, KV capacity, cap-hit rate, response-length distribution, and
  concurrency reported with throughput;
- no active-run modification and no retroactive relabeling of current
  compressed-TriGLU evidence as vanilla side-FFN evidence.

## 2026-07-14 MathAvg And GRPO Continuation Addendum

The primary aggregate for cross-model quality decisions is reactivated as:

```text
MathAvg = (GSM8K + MATH-500 + OlympiadBench + AMC Average@32) / 4
```

The Whole-50K training-mix weighted proxy remains a useful secondary metric
for training-distribution alignment and continues to use AMC greedy pass@1.
AMC greedy pass@1 alone is a diagnostic of the modal answer path; it must not be
substituted into `MathAvg`. Raw per-benchmark scores remain mandatory beside
both aggregates.

Current complete observations are:

| Checkpoint | MathAvg | Weighted proxy | AMC greedy |
|---|---:|---:|---:|
| Baseline step 20 | 48.6771 | 63.7890 | 15/40 |
| TriGLU step 20 | 48.7262 | 64.2849 | 15/40 |
| TriGLU step 98 | 51.3401 | 64.0703 | 15/40 |

TriGLU step 20 exceeds baseline step 20 by only `+0.0490` MathAvg points,
while TriGLU rises `+2.6139` points from step 20 to step 98. AMC greedy is
unchanged at `15/40` in every currently complete cell, so it has not supplied a
useful learning-curve signal. Baseline step 98 remains pending as of the
2026-07-14 step-75 snapshot and must not be inferred.

Six-GPU vLLM parallel evaluation has reduced a complete checkpoint evaluation
to approximately five to nine minutes after export. This supersedes the old
serial-evaluator assumption that full evaluation is unaffordable at early
gates. After the active wave, retroactively evaluate both variants' immutable
step-30 and step-60 checkpoints to form a `20/30/60/98` curve. Continue an
existing positive run to cumulative `128/158/196`; evaluate a new architecture
from the untuned base at prospective `30/60/98`.

The continuation is paired and interleaved in this exact order:
`TriGLU-128 eval`, `baseline-128 eval`, `TriGLU-158 eval`, `baseline-158
eval`, `TriGLU-196 eval`, `baseline-196 eval`. A first-member result is exposed
immediately as a self-trend with its pair marked pending; completing the second
member emits the same-step architecture delta. The monitor must report current
phase, milestone ETA, raw benchmarks, MathAvg, weighted proxy, reward, KL,
response length, and cap-hit rate without waiting for the full 98-update block.
All durable names use cumulative steps; `+30/+60/+98` are explanatory aliases
only. Paired sampler/RNG and consumed-row ledger hashes are mandatory at every
boundary.

For the current `504`-prompt update contract, step 98 is explicitly a
`98-step near-one-pass`: it consumes 49,392 of 50,000 rows from the matched
deterministic shuffle and leaves 608 unconsumed. Preserve the omitted row-ID
ledger/hash and source composition. Do not create an irregular tail batch or
call the milestone an exact epoch. Exact all-row carry-over requires a
separately validated sampler protocol.

## Pending Obligations Carried Forward

Canonical registry:
`docs/experiment_plans/PENDING_2026-07-12_mandatory-experiment-gates.md`

- **PENDING-01 Eval Parity Matrix:** still PENDING. It is required before a new
  backend-dependent quality comparison is decision-grade and is in scope at
  Stage 8 of this roadmap.
- **PENDING-02 Pure-BF16 SHS And TriGLU:** still PENDING. It is an immediate
  runtime-optimization dependency in Stages 3-4; planning or one architecture
  alone does not complete it.
- **PENDING-03 Registered SHS CausalLM Route:** still PENDING. It is deliberately
  deferred from the active TriGLU/baseline wave but required for the shared
  native/custom-FFN runtime claim.
