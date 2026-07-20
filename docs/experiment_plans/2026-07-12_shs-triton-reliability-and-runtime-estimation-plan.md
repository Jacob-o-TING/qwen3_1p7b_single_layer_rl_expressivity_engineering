# 2026-07-12 SHS Triton Reliability And Runtime Estimation Plan

Date: 2026-07-12

Status: single-GPU runtime execution completed through a real 128-prompt,
group-size-4 veRL optimizer step. The runtime is not yet a production candidate
because fast-evaluator semantic parity failed and the real shard's 441/512
zero-reward outputs have not been classified. No multi-GPU experiment wave is
authorized.

## Pending Obligations Carried Forward

Canonical registry:
`docs/experiment_plans/PENDING_2026-07-12_mandatory-experiment-gates.md`.

- **PENDING-01 Eval Parity Matrix:** deferred from the TriGLU transplantation
  and still required before backend-dependent scores become decision-grade.
- **PENDING-02 Pure-BF16 SHS And TriGLU Architecture Paths:** planned below as
  a separate dtype-policy and vLLM gate; neither architecture is complete.
- **PENDING-03 Registered SHS CausalLM Generation Route:** planned below as a
  separately approved implementation gate; historical generic SHS execution
  does not close it.

These obligations remain visible in every later experiment plan until their
canonical registry status is explicitly changed to `COMPLETE` with evidence.

## Core Objective

Turn the existing SHS inference prototype into a numerically trustworthy
training-and-rollout runtime, while obtaining decision-grade SFT and GRPO wall-
time estimates before committing to a complete custom backward kernel.

The immediate success criteria are:

1. explain and either remove or deliberately bound the current full-logit
   numerical drift without weakening the preregistered gate after observing it;
2. separate vLLM continuous-batching gains from the SHS Triton-kernel gain with
   matched checkpoints, prompts, generated-token counts, and concurrency;
3. measure a production-shaped SFT step sequence and a production-shaped GRPO
   per-GPU shard with complete timing decomposition;
4. validate the final four-GPU extrapolation with one real four-GPU global batch
   before authorizing the 20/50/98-batch ladder.

## Current Evidence And Limits

- The grouped SHS multiplicative Triton operator is 3.62x-21.01x faster than
  the 32-sliced-GEMM reference in isolated warm BF16 tests, but at production-
  relevant pressure 64 the SHS reference path reached 2,262.8 tokens/s versus
  1,725.1 for grouped Triton. The selected runtime remains reference
  PyTorch/cuBLAS inside vLLM continuous batching.
- Drift localization found amplification beginning after Layer 10, visible by
  Layer 11. FP32 accumulation did not repair the fixed full-logit cosine gate;
  the fast Triton backend therefore remains experimental.
- Triton-forward/reference-recompute autograd passed local VJP, finite-gradient,
  optimizer, reload, and resume gates. It did not pass full-model gradient
  parity because the forward drift remains, and it is not a custom backward.
- The matched 50-step SFT matrix selected reference eager at 1.4526 seconds per
  step, or approximately 1.58 hours for 3,916 steps. Triton recompute compiled
  projected to approximately 3.59 hours, making actor backward a material
  optimization target rather than a current speedup.
- A native vLLM evaluator backend, stable sample identities, resumable partial
  outputs, live status, and topology-neutral replica sharding/merge now exist.
  Production-pressure baseline evaluation projects to 16.66 minutes, but the
  expanded semantic-parity panel failed and blocks full fast evaluation.
- Real veRL actor construction, completed-checkpoint overlay, production
  `math_verify`, SHS-reference vLLM synchronization, one optimizer step, compact
  checkpoint, and project-level completion resume all executed successfully.
- The real 128-prompt, group-size-4 shard took 745.60 seconds for the measured
  trainer step and 826 seconds shell wall. It is timing evidence, not a quality
  or production-candidate pass, because output-quality failure modes remain
  unclassified and fast-evaluator parity remains unresolved.

## Track A: Triton Correctness Before More Fusion

### A1. Localize The Deep-Logit Drift

Run matched reference/Triton captures at the output of each Layer-10
projection, after SwiGLU, after the Layer-10 residual, and after every later
transformer block. Record absolute error, relative L2, cosine, top-k overlap,
and the first layer at which error amplifies.

Compare, without changing model weights or prompts:

- current grouped BF16 reduction order;
- FP32 accumulation with BF16 output;
- deterministic reference-equivalent reduction order;
- base GEMM plus Triton delta versus fully reference base/delta paths;
- pressure 1 and flattened-token pressure 32.

The gate remains logits cosine at least 0.9999. If an exact-reduction variant is
too slow, preserve both a strict-parity backend and a separately labelled fast
backend; do not silently redefine parity.

Execution status: completed. The first material amplification appears after
Layer 10 and is visible by Layer 11. FP32 accumulation did not recover the gate;
no strict-parity Triton production backend is currently available.

### A2. Establish A Training-Capable First Version

Do not begin with a hand-written full backward. First implement a custom
autograd integration whose forward uses the proven Triton multiplicative path
and whose backward recomputes the reference PyTorch algebra. This is allowed to
be slower than the eventual target, but it must provide an honest kernel-
enabled training path and gradients for:

- inputs and all original gate/up/down projection weights;
- grid-generator parameters;
- multiplicative scales/parameters;
- additive low-rank factors, scales, and inputs.

Check FP32 and BF16 loss, input gradients, every trainable parameter group,
optimizer state, checkpoint/reload, and deterministic resumed updates. Only
after profiling this path should a fused custom backward be considered.

Execution status: completed as an honest recompute-autograd integration. Local
VJP and checkpoint/update tests passed, while full-model gradient parity remains
blocked by the A1 forward drift. `custom_backward=false` remains mandatory.

### A3. Profile The Remaining SHS Work

Measure the generator, additive path, multiplicative path, base GEMMs, and
elementwise/residual work independently during prefill, decode, actor forward,
and actor backward. Implement the two-stage additive Triton path only if it is a
material end-to-end bottleneck. Keep large dense GEMMs on cuBLAS/cuBLASLt.

## Track B: Matched SFT Timing

Use the same real SHS checkpoint contract, deterministic train-item manifest,
sequence length, microbatch, gradient accumulation, optimizer, and precision as
the production 50K SFT run. Compare:

| Path | Required purpose |
|---|---|
| Reference eager | reproduce the known production anchor |
| Reference `torch.compile` | quantify compiler-only gain and compile cost |
| Triton-forward/reference-backward eager | isolate the first trainable kernel path |
| Triton-forward/reference-backward compiled | test whether composition improves or regresses |

For each path, report cold setup/compile separately, discard warmup from the
steady-state median, and collect at least 50 measured optimizer steps after
warmup. Record median, p10/p90, peak memory, tokens/s, graph breaks, and the
projected 3,916-step wall time. Run one sparse checkpoint/reload continuation
for the selected path.

The present honest anchor remains approximately 1 hour 37 minutes for the
current Layer-10 SHS SFT run on one RTX PRO 6000. No Triton SFT speedup may be
claimed until this matrix is complete.

Execution status: matrix complete. Reference eager was fastest at a 1.4526-
second median step and approximately 1.58 projected loop hours. The Triton-
recompute compiled cell projected to approximately 3.59 hours. Production SFT
therefore retains reference eager; a fused backward is optional future work.

## Track C: Matched Rollout And GRPO Timing

### C1. Separate vLLM And Triton Contributions

Run one fixed prompt panel and fixed generated-token budget through:

1. naive Qwen vLLM;
2. SHS reference vLLM;
3. SHS Triton vLLM strict-parity backend;
4. SHS Triton vLLM fast backend, only if distinct from strict parity.

Use TP=1, one complete replica per GPU, the same engine settings, prompt IDs,
token caps, stops, seeds, and pressure sweep 1/8/16/32/64. Report scheduler-
native prefill/decode timings where available, completed tokens/s, request
latency p50/p90/p99, cap hits, peak memory, and dispatch receipts. Include long
responses near the observed 800-1,000-token production range; the previous tiny
phase proxies are not sufficient.

Execution status: complete for naive Qwen, SHS reference, and grouped-Triton-
fast. At pressure 64, rates were 5,329.7, 2,262.8, and 1,725.1 tokens/s,
respectively. No strict-Triton cell exists because A1 failed.

### C2. Production-Shaped One-GPU Shard

On one RTX PRO 6000, run the exact work assigned to one replica in a four-GPU
paper-shaped global batch: 128 prompts, group size 4, production response cap,
real reward/extraction, old/reference log probabilities, actor forward and
backward, optimizer contribution, synchronization, and checkpoint/log overhead.

Record:

```text
engine/model load
compile and warmup
rollout prefill
rollout decode
reward and answer extraction
old/reference log probabilities
actor forward/backward/optimizer
all-reduce estimate or measurement
actor-to-rollout weight synchronization
checkpoint and orchestration
```

This shard should reduce the current full-run uncertainty from roughly +/-25%
to about +/-15%, provided generated-length and pressure distributions match the
production manifest.

Execution status: completed through one real optimizer step after correcting
two integration defects. Twelve deterministic SHS block-ID buffers are now
excluded from FSDP-to-vLLM weight synchronization while remaining available at
runtime. The one-GPU PPO micro-batch is two, matching the paper's global micro-
batch eight across four GPUs. The successful shard changed all 29 expected
trainable tensors and wrote a verified checkpoint. Its completion gate prevents
`resume_mode=auto` from repeating an already completed optimizer step.

### C3. One Real Four-GPU Global Batch

The final estimate gate is one complete 512-prompt, group-size-4 batch on four
identical GPUs with one rollout replica per GPU and the actual distributed actor
path. Compare measured wall time with the sum of the four single-GPU shard
projections. A discrepancy above 10% requires profiling scheduler imbalance,
all-reduce, reward bottlenecks, or synchronization before extrapolation.

Only this measurement may authorize a +/-10% full-budget estimate and the
2/20/50/98/196/391-batch ladder.

## Provisional Runtime Projection

The earlier pressure-32 proxy of 777.9 decode tokens/s is superseded by the
matched long-response C1 benchmark. With 800-1,024 generated tokens, pressure
64, and otherwise identical vLLM settings, SHS reference vLLM reached 2,262.8
generated tokens/s per RTX PRO 6000. The grouped-Triton-fast path reached only
1,725.1 tokens/s at pressure 64, so the current production candidate is the
strict-correct SHS reference PyTorch/cuBLAS projection inside vLLM continuous
batching, not plain sequential Hugging Face generation and not the current
grouped Triton kernel.

Four independent TP=1 SHS-reference replicas provide a measured-rate aggregate
of approximately 9,051 generated tokens/s. For 512 prompts, group size 4, and
800-1,000 generated tokens per response:

| Budget | SHS pure rollout | Provisional rollout wall incl. scheduling margin |
|---:|---:|---:|
| 20 global batches | 1.0-1.3 h | 1.2-1.6 h |
| 50 global batches | 2.5-3.1 h | 3-4 h |
| 98 global batches | 4.9-6.2 h | 6-8 h |
| 391 global batches | 19.7-24.6 h | 23-30 h |

The rollout-only table remains useful for locating generation cost, but it is
superseded as a complete-GRPO estimate by the real C2 shard. The successful
single-GPU shard measured:

| Component | Seconds |
|---|---:|
| rollout generation | 266.54 |
| reward and extraction | 9.84 |
| old-policy log probabilities | 90.67 |
| reference-policy log probabilities | 93.71 |
| actor update | 284.67 |
| checkpoint save | 10.40 |
| measured trainer step | 745.60 |
| complete shell wall | 826.00 |

The shard processed 516,253 tokens, with mean response length 897.60, cap-hit
rate 8.79%, and mean binary reward 0.13867. On four GPUs, each TP=1 replica is
intended to receive the same 128-prompt, group-size-4 workload concurrently.
The 745.60-second trainer step is therefore the current per-global-batch timing
anchor, not one quarter of the four-GPU wall time. DDP all-reduce, rank
imbalance, and shared-host contention remain unmeasured.

| Budget | Linear measured-step projection | Practical four-GPU planning range |
|---:|---:|---:|
| 20 global batches | 4.14 h | 4.5-5.5 h |
| 50 global batches | 10.36 h | 11-14 h |
| 98 global batches | 20.30 h | 21-27 h |
| 196 global batches | 40.59 h | 42-54 h |
| 391 global batches | 80.99 h | 82-105 h |

The practical range amortizes engine/Ray startup, assumes intentionally sparse
checkpoints rather than saving every step, and reserves margin for distributed
communication and stragglers. It is still provisional until one real four-GPU
global batch runs, but it replaces the earlier 40-70-hour full-GRPO range.

For comparison, naive Qwen vLLM reached 5,329.7 generated tokens/s per GPU at
pressure 64. Four replicas project to 77-96 seconds of pure rollout per global
batch and 8.4-10.4 hours over 391 batches. SHS reference rollout is therefore
approximately 2.35x slower than naive Qwen at the measured high-pressure point.
This makes further SHS rollout optimization material, but the next design must
target large flattened-token batches. A pressure-aware hybrid may use Triton at
small M and cuBLAS/reference projections at large M, subject to matched parity
and end-to-end benchmarks.

The real veRL path exercised actor-to-vLLM `update_weights` rather than full
export plus engine rebuild at every step. The earlier fixed 2.7-hour rebuild
penalty no longer applies to this selected path. Weight synchronization remains
part of the four-GPU timing audit, including deterministic-buffer exclusions and
post-sync dispatch verification.

## Time To A Reliable Estimate

The single-GPU SFT matrix, long-response rollout matrix, real veRL construction,
and production-shaped C2 shard are complete. They provide a measured component
anchor and an approximately +/-15% planning range, subject to generation-
quality and semantic-parity blockers.

The remaining timing gate is one real four-GPU global batch. Budget one to two
rental hours for environment/topology preflight, warmup, the approximately
12.5-minute measured-step workload, checkpoint verification, and one bounded
diagnostic rerun if needed. Only that gate can reduce the estimate toward +/-10%.

A fused custom backward is not required to obtain the estimate, but the real
shard assigned 284.67 seconds, or about 38% of measured trainer-step wall, to
the actor update. Backward optimization is therefore economically material if
profiling attributes a substantial fraction to the custom SHS path.

## Run Status

| Purpose | Run ID | Status |
|---|---|---|
| Deep-logit localization | `shs_triton_drift_localize_20260712_v1` | complete; strict gate failed |
| Matched long-response rollout matrix | `shs_vllm_matched_longdecode_20260712_v1` | complete |
| Trainable autograd parity | `shs_triton_autograd_parity_20260712_v1` | local pass; full-model fail |
| Matched SFT 50-step matrix | `shs_sft_runtime_matrix_20260712_v1` | complete; reference eager selected |
| Fast-evaluator parity | `vllm_eval_single_gpu_parity_20260712_v1` | complete; semantic parity failed |
| Fast-evaluator speed | `vllm_eval_single_gpu_speed_20260712_v1` | complete; 16.66-minute baseline projection |
| Replica sharding/merge dry run | `vllm_eval_multireplica_dryrun_20260712_v1` | complete on logical ranks |
| One-GPU real veRL shard | `shs_grpo_replica_shard_20260712_v2_realverl` | one step complete; candidate blocked |
| Four-GPU complete global-batch gate | `shs_grpo_globalbatch_4gpu_20260712_v1` | not authorized |

Every run must write a preregistered manifest, human-readable log, machine-
readable component timings, dispatch/fallback receipts, environment versions,
checkpoint hashes, prompt-manifest hashes, and generated-token counts. Compact
source/docs/metrics may be committed; models, datasets, exports, and checkpoints
must remain ignored.

## Stop And Decision Gates

- Do not touch or interrupt the old production SFT/evaluation instance.
- Do not weaken numerical thresholds after observing failures.
- Do not call a reference-backward path a custom backward kernel.
- Do not claim a Triton gain without a matched SHS reference-vLLM cell.
- Do not extrapolate a cap-16 smoke or isolated operator timing to full GRPO.
- Stop kernel expansion if profiling shows the custom SHS path is below 10% of
  end-to-end rollout/actor wall time or if its projected gain cannot change the
  rental decision materially.
- Continue through recoverable implementation failures. Shutdown remains a
  user-controlled action unless continued execution has clearly lost economic
  value under the documented abnormality gate.

## Production-Readiness Boundary

Perfect completion of the selected single-GPU inference backend, vLLM, matched-
SFT, and production-shaped GRPO-shard work makes the runtime a **production
candidate**, not yet a fully production-ready four-GPU training system. The
distinction prevents a successful kernel or one-batch smoke from being confused
with authorization for a complete 391-batch wave.

At the production-candidate boundary, the system should already provide:

- numerically trustworthy selected SHS inference with strict parity evidence;
- stable vLLM continuous batching for realistic long responses;
- a training-capable Triton-forward path with gradient-correct backward;
- one-GPU production-shaped rollout/update component timings;
- checkpoint/resume, explicit dispatch receipts, and no silent fallback;
- an approximately +/-15% four-GPU SFT/RL runtime projection.

The remaining production-readiness gates are deliberately smaller integration
and stability tasks rather than a new runtime rewrite:

1. **Real GRPO trainer integration: complete.** The actual veRL hook, completed
   SHS checkpoint overlay, production `math_verify`, old/reference log
   probabilities, actor update, vLLM weight synchronization, checkpoint, and
   completion-resume gate all executed in the C2 shard.
2. **Four-GPU distributed validation.** Execute one complete 512-prompt,
   group-size-4 global batch with four TP=1 rollout replicas and the real DDP
   actor path. Verify rank sharding, common seeds/manifests, all-reduce, weight
   synchronization, and absence of rank-specific fallback or failure.
3. **Weight-refresh decision: selected path exercised.** Real veRL
   `update_weights` synchronized the actor to vLLM without per-step engine
   rebuild. Revalidate this path under four ranks and retain the explicit
   exclusion of exactly 12 reconstructible deterministic SHS block-ID buffers.
4. **Longer stability evidence.** Run the launch ladder below to expose memory
   fragmentation, scheduler stalls, pathological long responses, OOM recovery,
   interruption/resume defects, and reward/KL/length instability.
5. **Actor-speed decision.** Triton forward plus reference-recompute backward
   may be production-correct without being production-fast. vLLM accelerates
   rollout inference, not actor backward. Write a fused backward only if the
   production-shaped profile shows actor backward is a material bottleneck.

### Required Launch Ladder

```text
single-GPU production-candidate gates
-> one real four-GPU global batch
-> two-batch canary
-> 20-batch pilot
-> reward/KL/length/throughput/resume review
-> 50- or 98-batch scientific gate
-> 196- and 391-batch continuation only if justified
```

The two-batch canary validates orchestration and recovery cheaply. The 20-batch
pilot is the minimum practical stability and learning-trend gate before a
substantial wave. At the current measured-shard anchor it is expected to cost
roughly 4.5-5.5 four-GPU hours. Passing the real trainer integration, four-GPU
global-batch gate, and 20-batch pilot promotes the runtime from production
candidate to **production ready** for the approved budget ladder.

The remaining work before the four-GPU gate is no longer trainer plumbing. It
is semantic: resolve or formally separate the vLLM evaluation protocol,
classify zero-reward and cap-hit shard generations, and prove that the selected
runtime does not change the scientific decision. After those pass, budget one to two four-
GPU rental hours for the distributed gate and 4.5-5.5 hours for the 20-batch
pilot.

## Post-Production Parallel Evaluation Track

The current serial evaluation path is too expensive for repeated RL research.
Its wall time can approach the same order of magnitude as a 50-batch RL pilot,
which makes every architecture comparison pay a second large compute bill after
training. After the training runtime reaches the production-ready boundary,
build a separate four-GPU evaluation path using the validated Triton inference
kernel and vLLM continuous batching.

The durable onboarding and validation procedure is maintained in
`docs/runtime/custom-architecture-vllm-onboarding.md`. Update that general
workflow only when the reusable runtime contract changes; keep run-specific
evidence in experiment records.

The intended topology is one TP=1 vLLM evaluation replica per GPU. Shard
benchmark items, not model tensors, across the four replicas; allow each vLLM
engine to continuously batch heterogeneous requests locally. The coordinator
must preserve the canonical benchmark item IDs and decoding contract, then
merge outputs by stable item/sample identity rather than completion order.

The parallel evaluator must provide:

- deterministic benchmark/sample sharding for any declared replica count;
- identical prompts, generation parameters, seeds, stop rules, token caps,
  extraction, grading, and Average@32/pass@1 semantics relative to the serial
  evaluator;
- per-rank progress, generated-token counts, throughput, errors, dispatch
  receipts, partial-result files, and resumable completion receipts;
- incremental merge and human-readable partial accuracy while evaluation is
  still running;
- idempotent resume that skips only verified `(benchmark, item, sample)` keys;
- matched serial-versus-parallel parity on a fixed subset before scale-out;
- no duplicate rows or omissions under worker interruption and restart;
- independent benchmark concurrency limits so CPU extraction/grading and GPU
  generation cannot accidentally oversubscribe the host.

Measure one-, two-, and four-replica scaling efficiency. Four GPUs should
ideally reduce item-parallel evaluation wall time toward one quarter of serial
time, but the report must expose load imbalance from response-length variance,
benchmark size, cap hits, tokenizer work, and grading bottlenecks. Evaluation
optimization begins only after the production training path is complete; it
must not distract the current Triton, SFT-timing, and GRPO-timing critical path.

The first single-GPU projection based on completed artifacts and the matched
long-decode matrix suggests approximately 10-20 minutes for naive Qwen and
20-35 minutes for SHS reference, versus observed serial walls of 4 h 45 m and
8 h 10 m. These 14-28x planning ranges are not substitutes for a matched
end-to-end evaluator run. Every model-summary table must include both AMC
Average@32 and the separate greedy AMC pass@1 diagnostic; greedy remains
excluded from the four-benchmark arithmetic mean.

Implementation status: the single-GPU generator backend, stable identities,
partial/live status, completion receipts, and topology-neutral shard/merge path
are complete. Logical multi-rank dry-run coverage passed. The 16.66-minute
baseline projection confirms economic value, but scientific parity failed; the
next work is semantic resolution, not additional scheduler optimization.

## GPU-Count And Topology Audit Before Future Scale-Out

Future experiments may use more than four GPUs. Before any launch on a new GPU
count, globally audit source, configs, scripts, manifests, tests, and resume
logic for hard-coded or implicitly fixed topology. This is required because the
current Single-Layer SFT work previously encountered a GPU-count mismatch that
interrupted training.

The audit must cover at least:

- `CUDA_VISIBLE_DEVICES`, explicit device IDs, `nproc_per_node`, `world_size`,
  local/global rank, rendezvous ports, and launcher process counts;
- vLLM `tensor_parallel_size`, number of independent rollout/evaluation
  replicas, per-replica memory limits, and engine worker counts;
- global batch, per-rank microbatch, gradient accumulation, prompt-group
  assignment, and divisibility assumptions;
- dataset sampler length, epoch boundaries, shuffle/order manifests, dropped
  tails, and resume cursors when world size changes;
- DDP/FSDP collectives, barriers, checkpoint ownership, optimizer-state
  sharding, all-reduce expectations, and rank-zero-only writes;
- evaluation shard/merge logic, cache-directory ownership, duplicate detection,
  and completion-receipt counts;
- monitoring scripts, progress denominators, expected screen/process counts,
  GPU utilization summaries, and abnormality detection;
- tests or conditionals containing literal GPU counts such as 1, 2, 4, or 8.

Add a topology preflight that discovers the visible hardware, compares it with
the declared contract, prints the resolved topology and batch arithmetic, and
fails before loading the model if they disagree. A mismatch must never be
discovered only after training has begun. The preflight receipt should include:

```text
visible GPU count and UUIDs
launcher world size and rank mapping
rollout/eval replica count and tensor-parallel size
per-rank microbatch and gradient accumulation
effective global prompt/update batch
dataset cardinality, shard sizes, and tail policy
checkpoint/resume world-size compatibility
resolved output/cache ownership per rank
```

Keep the scientific contract independent of physical topology wherever
possible. Changing GPU count must not silently change effective batch size,
prompt order, group membership, seeds, decoding, optimizer schedule, or the
number of updates. When exact sample-order or optimizer-state continuation
cannot be preserved across a world-size change, reject the resume explicitly
and require a newly named experiment rather than silently altering semantics.

## Fast-Evaluator Scientific Parity Boundary

Fast evaluation is intended to resolve an evaluation-cost bottleneck without
erasing the small architecture effects under study. Engineering-style
tolerances of one or two percentage points are unacceptable here because the
observed SHS, baseline, and TriGLU aggregate differences are themselves below
one point.

Before a vLLM evaluator may replace the serial evaluator in the same scientific
protocol, it must satisfy all of the following:

- exact checkpoint, architecture/config, item/sample identity, prompt-token,
  chat-template, decode-parameter, seed-derivation, stop/cap, extractor, grader,
  expected-row, and aggregation contracts;
- exact scoring replay when identical saved responses are passed to both
  evaluator paths;
- no duplicate or missing item/sample keys under completion reordering or
  resume;
- absolute score drift no greater than 0.5 percentage points on every primary
  benchmark;
- target four-benchmark-average drift no greater than 0.25 points, with 0.5 as
  a hard rejection boundary;
- no systematic same-direction shift across benchmarks, response lengths,
  cap-hit rates, or extraction failures;
- paired correct-to-wrong and wrong-to-correct transitions reported rather than
  hidden behind a similar aggregate.

Deterministic greedy evaluation is held to a stronger standard. Prompt tokens
must match exactly and extracted-answer agreement should be at least 99.5%.
For the 40-item AMC greedy panel, a single item changes accuracy by 2.5 points;
the 0.5-point scientific boundary therefore requires the aggregate score to be
exact, with paired item disagreements still reported even when they cancel.

AMC Average@32 uses clustered sampling over 40 items. Overall drift must remain
within 0.5 points, but approval also requires paired per-item `correct_count/32`
analysis and a cluster-level paired uncertainty estimate. If backend RNG
consumption prevents a sufficiently precise comparison, add preregistered seed
panels or common-random-number support rather than loosening the score boundary.

If these gates fail, vLLM defines a new evaluation protocol. It may still be
used for fast within-protocol research only after untuned base, whole-layer
baseline, SHS, TriGLU, and OFT are all rerun under the same backend. Its absolute
scores must not be mixed with the existing serial results or used as if they
were directly paper-comparable.

### 2026-07-12 Observed Fast-Evaluator Gate

The first implementation demonstrated the desired speed but failed scientific
parity. A production-pressure baseline projection estimated a 16.66-minute full
evaluation, but the expanded 24-item greedy panel matched only 16/24 extracted
answers and 21/24 scores; aggregate correctness moved from 13/24 under HF to
14/24 under vLLM. SHS-reference vLLM also changed one of two GSM8K extracted
answers from the HF checkpoint/export result.

Identity sets, prompt hashes, review keys, row counts, checkpoint export, and
extraction/accounting contracts passed. The remaining discrepancy is generated
trajectory drift in the vLLM runtime. No full fast evaluation is authorized,
and the runtime must not be labelled production compatible with the existing
serial protocol until the strict boundary above passes or a separately named
all-model vLLM protocol is approved.

## Checkpoint-By-Backend Differential Diagnosis

Do not conflate checkpoint quality with runtime behavior. Run a controlled
two-by-two matrix on the exact same preregistered prompt IDs, prompt tokens,
decode parameters, response caps, stops, seeds, and scoring pipeline:

| Checkpoint state | Serial HF | vLLM reference |
|---|---|---|
| Initial SHS exact-no-op state | A | B |
| Final SHS SFT checkpoint | C | D |

The initial SHS state must preserve the exact-zero/no-op architecture invariant
and be functionally equivalent to the untuned Qwen base while retaining the SHS
module structure. The final state must be the hash-bound completed SFT
checkpoint and deployment export already audited.

Record raw text, extracted answer, score, token IDs where available, response
length, cap hit, extraction status, per-token or sequence log-probability
diagnostics, and paired item transitions for every cell.

Interpret the matrix as follows:

1. **A and B agree; C and D both degrade.** The dominant issue is final-SFT
   checkpoint behavior. Starting scientific RL from the initial exact-no-op SHS
   state is a justified candidate, subject to a matched base control.
2. **HF cells are healthy; both vLLM cells degrade.** The dominant issue is the
   runtime/backend contract. Changing checkpoint does not make production
   rollout acceptable.
3. **A and B agree; C is healthy while D degrades.** The likely mechanism is an
   interaction: SFT makes the policy more sensitive to small backend numerical
   differences, which then amplify autoregressively under vLLM.
4. **All cells show comparable anomalous-output rates on identical prompts.**
   Difficult prompt distribution or ordinary model error is more plausible
   than a checkpoint-specific or vLLM-specific defect.

Aggregate percentage alone is insufficient. A similar score can hide different
correct items, while a small panel can exaggerate one trajectory divergence.
Apply the existing 0.5-point scientific boundary, deterministic greedy gates,
paired outcomes, and response-distribution diagnostics.

## Scientific RL Initialization Contract

The completed SFT checkpoint was used in the real C2 shard as an engineering
stress test for custom-weight loading, overlay, synchronization, actor update,
checkpointing, and resume. That run does not select the final scientific RL
initialization.

For the clean comparison `naive Layer-10 RL` versus `SHS Layer-10 RL`, both
policies should start from the same untuned Qwen base revision and common
initialization seed. SHS must begin at its exact-no-op state, so the initial
functions are identical and the intended causal difference is architecture
parameterization under RL. Starting SHS from its SFT final checkpoint while the
naive control starts from the untuned base would confound architecture, SFT,
and RL effects and requires a separately named experiment if ever desired.

Before any 20-batch pilot, compare vLLM rollout-policy probabilities with the
actor/old-policy log probabilities evaluated on the sampled rollout tokens.
Large implementation drift can violate the effective on-policy assumption,
distort importance ratios and clipping, and cannot be cleared solely by a
similar final benchmark percentage. Record log-probability deltas, approximate
KL, importance-ratio distribution, clip fraction, and any dependence on
checkpoint state.

## Immediate Post-Bacon Execution Plan

Proceed in this order:

1. **Preserve the completed evidence.** Treat the serial evaluator as the
   authoritative historical protocol. Keep all fast-evaluator and real-shard
   outputs under independent run roots.
2. **Classify vLLM trajectory drift.** Execute the checkpoint-by-backend matrix
   above and separate numerical model-forward drift, greedy tie/near-tie
   sensitivity, sampling RNG-order differences, final-SFT fragility, and runtime
   attention/cache behavior. Do not change the 0.5-point gate.
3. **Choose the evaluator branch.** If the strict-compatible path can be fixed,
   rerun the preregistered parity panel before any full evaluation. Otherwise,
   define a separately named all-vLLM protocol and rerun untuned base,
   baseline, SHS, TriGLU, and OFT before making within-protocol comparisons.
4. **Classify shard response failures.** The compact metrics establish 71/512
   reward-one responses, 441/512 reward-zero responses, and 45/512 cap hits;
   they do not establish a malformed count. Audit valid-but-wrong answers,
   format failures, extraction failures, cap hits, repetition/degeneration,
   invalid LaTeX, empty answers, and reward-zero causes by prompt/source/length.
   Compare a fixed shard subset against serial HF generation to distinguish
   model quality from runtime trajectory drift.
5. **Revalidate one single-GPU step only if semantics change.** The trainer,
   reward, overlay, synchronization, optimizer, checkpoint, and resume plumbing
   do not need to be rebuilt. Rerun C2 only after a generation/runtime fix that
   could alter timing or quality.
6. **Authorize four GPUs only after candidate gates pass.** Run one complete
   global batch, compare it with the 745.60-second shard anchor, and update the
   +/-10% budget estimate. Then follow the 2/20/50/98/196/391 ladder.

Kernel work is temporarily secondary. The grouped Triton backend is neither
strict-parity nor faster at production pressure, while reference vLLM already
executes the real pipeline. Resume custom-kernel work only after semantic
acceptance or when profiling a validated run identifies a material bottleneck.

## Shadow Full-Evaluation Execution Update

The approved `shs_vllm_full_eval_shadow_20260712_v1` run is complete. It used
the final SHS SFT checkpoint's hash-bound deployment export, TP=1,
Transformers model implementation, eager execution, and reference
PyTorch/cuBLAS SHS projections. It completed all paper-pinned main benchmarks,
AMC temperature-1.0/top-p-1.0 Average@32, and the separate 40-item greedy AMC
pass@1 diagnostic with exact verified review counts.

This result remains **SHADOW/NEW BACKEND**. The four-task average was 41.7178%
versus the authoritative HF 43.0882%, and benchmark deltas were -0.8000,
+1.1372, -0.7407, and -5.0781 points. It therefore fails the unchanged
0.5-point compatibility boundary and cannot authorize replacement of the HF
protocol.

The first full attempt took 3,202 seconds, while artifact-complete wall after
strict JSONL audit and idempotent repair was 4,497.73 seconds. The latter is the
cost and speed comparison of record: `6.537x` faster than the 8h10m serial HF
wall. Its 571.9 tokens/s artifact-complete throughput was only 25.27% of the
2,262.8 tokens/s long-decode anchor. Before another full shadow evaluation,
fix EvalScope concurrent JSONL serialization, retain the 250 ms gather window
and 128-sequence pressure setting, and profile eager reference-SHS decode,
long-output tails, CPU grading, and engine reload overhead. Do not relax the
semantic boundary or relabel this run as compatible.

## RTX 5090 Pair Bring-Up And Weight-Synchronization Plan

Execution status: completed as a bounded two-GPU systems wave. Gate A passed.
Gate B passed checkpoint, reference-training, isolated Triton, and V1 vLLM
smokes, but the unchanged strict full-logit Triton gate failed at cosine
0.9997553 versus the required 0.9999. Gate C completed ten live veRL sync
cycles plus a checkpoint restart/resync step and passed a repaired fresh-vLLM
reload. It remains partial: cast-normalized audit found 23/29 visibly changed
tensors, per-version receiver timing/hashes and direct-sync versus fresh-reload
full logits were not captured, and Ray emitted a post-checkpoint DataLoader
cleanup traceback. See
`docs/experiment_records/2026-07-12_rtx5090-pair-bringup-and-weight-sync-record.md`.
No production GRPO ladder is authorized by this result.

Two cloned RTX 5090 instances are expected for the next systems gate. This is
enough to separate one actor GPU from one rollout GPU and test real
actor-to-rollout synchronization. It is not the paper's four-replica topology
and does not authorize a production GRPO wave by itself.

Proposed run identities, subject to explicit approval before launch:

| Purpose | Run ID | Screen |
|---|---|---|
| Per-instance environment and architecture gate | `rtx5090_pair_bringup_20260712_v1` | `qwen_rtx5090_bringup_20260712_v1` |
| Actor-to-rollout synchronization gate | `shs_2x5090_actor_rollout_weight_sync_20260712_v1` | `qwen_shs_2x5090_sync_20260712_v1` |

Every instance must use a separate output root and record hostname, GPU UUID,
driver, CUDA runtime, Python environment, source commit, checkpoint hash, and
network role. Do not infer that two rented GPUs share a host or private LAN.
First classify the topology as same-host two-GPU, private-LAN multi-node, or
gateway-only independent instances. A gateway-only topology may validate the
portable export/reload control but cannot be labelled live distributed sync.

### Gate A: Environment Compatibility On Both RTX 5090s

Run this gate independently on each clone before loading a large checkpoint:

1. Verify the GPU is an RTX 5090 with approximately 32 GB VRAM, CUDA capability
   `(12, 0)`, and a driver able to run CUDA 12.8 binaries.
2. Record `torch.__version__`, `torch.version.cuda`,
   `torch.cuda.get_arch_list()`, vLLM, Triton, Transformers, EvalScope, veRL,
   and `math-verify` versions. The expected proven stack is PyTorch
   `2.8.0+cu128`, vLLM `0.10.2`, Triton `3.4`, Transformers `4.57.1`,
   EvalScope `1.8.1`, and `math-verify==0.9.0`.
3. Require `sm_120` in the PyTorch architecture list. Reject an inherited
   PyTorch build based on CUDA 12.6 or older even if imports succeed.
4. Run BF16 allocation, matrix multiplication, loss, backward, optimizer step,
   and CUDA synchronization. Record peak allocated/reserved memory and reject
   NaN/Inf output or gradients.
5. Verify NCCL availability and collect interface/routing information without
   assuming cross-instance reachability. Test a two-rank scalar broadcast only
   after the topology and firewall are known.
6. Use new per-run `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` locations.
   Never copy or reuse compiled RTX PRO 6000 cache artifacts as evidence for a
   5090 run.
7. Import the custom HF model, vLLM plugin, EvalScope adapter, veRL patch, and
   reward adapter from the pinned environment. Any fallback environment or
   mixed Torch/vLLM ABI blocks the GPU smoke.

### Gate B: Quick Checkpoint And Custom-Architecture Smoke

Use the locally archived SHS final checkpoint only after its
`trainable_state.pt` hash equals
`ebb4cd92f6e890c17cf0e14a883557358dff927e901b69588f1867e6dd016712`.

1. Reconstruct Qwen3-1.7B plus SHS and require the exact expected loaded,
   missing, and unexpected key sets.
2. Run one reference eager forward, one greedy decode, one BF16 backward, and
   one optimizer step without retaining the updated model as scientific data.
3. Recompile the SHS Triton kernel from an empty cache. Run FP32 and BF16 gate,
   up, down, and odd-tail parity cases at token batches 1 and 8; require an
   explicit Triton dispatch receipt and no silent reference fallback.
4. Run the bounded full-model parity panel against the reference backend. Keep
   all existing tolerances unchanged, including the 0.9999 full-logit cosine
   gate that the earlier grouped path failed.
5. Export the hash-bound checkpoint and load one TP=1 eager vLLM engine. Begin
   conservatively at `max_num_seqs=16`,
   `max_num_batched_tokens=32768`, and GPU memory utilization at most 0.80.
   Test request pressure 1, 8, and 16 with heterogeneous lengths before trying
   32 or 64. Do not copy the 96 GB RTX PRO 6000 pressure-128 settings.
6. Confirm prompt-token parity, stop/cap behavior, greedy extraction, Paged KV
   cache operation, response counts, peak VRAM, and backend receipts. This is a
   quick compatibility smoke, not a replacement for the fixed semantic panel.

The expected clean-path wall is 20-40 minutes per instance when dependencies
already match. Dependency repair or wheel replacement is an abnormal branch
and must be recorded before continuing; do not silently upgrade the environment
in place and then compare its timings with the RTX PRO 6000 record.

### Gate C: Two-RTX-5090 Actor-To-Rollout Weight Sync

Assign GPU/instance A as the PyTorch/veRL actor and GPU/instance B as the TP=1
vLLM rollout engine. Both begin from the same base revision, SHS architecture,
and checkpoint/export hashes. Preserve a fresh-reload rollout engine as the
correctness oracle.

1. Establish the transport explicitly. Prefer NCCL for CUDA tensors when the
   instances have a reachable private network; use Gloo only for control-plane
   metadata. Record interface, rank mapping, rendezvous endpoint, NCCL debug
   summary, and broadcast throughput. Do not tunnel bulk tensor sync through
   the public SSH gateway and call it a production result.
2. On the actor, run one deterministic tiny optimizer update and prove that all
   29 expected trainable tensors change while frozen parameters and the 12
   deterministic SHS block-ID buffers do not enter the synchronization payload.
3. Save actor tensor names, shapes, dtypes, element counts, pre/post hashes, and
   maximum deltas. Require exact payload agreement before applying the update
   to rollout.
4. Execute the desired direct vLLM `update_weights` path without rebuilding the
   engine. Record transfer, receiver-apply, cache/barrier, and first-post-sync
   generation walls separately. Any missing/unexpected key, stale tensor, rank
   disagreement, timeout, or fallback is a hard failure.
5. Independently construct a fresh rollout engine from the updated actor export.
   Compare direct-sync versus fresh-reload parameter hashes, fixed-prompt
   logits, next-token rankings, greedy tokens, and backend dispatch receipts.
   The direct path passes only if it reproduces the fresh-reload oracle within
   the preregistered BF16 tolerance; merely changing the generated text is not
   evidence of correct synchronization.
6. Verify a pre-update prompt changes when the actor update is intentionally
   large enough to be observable, while direct-sync and fresh-reload agree on
   the post-update result. This catches a no-op or stale-engine false pass.
7. Repeat ten bounded update/sync cycles without optimizer accumulation beyond
   the disposable smoke. Track sync latency distribution, GPU memory growth,
   receiver stalls, stale KV/cache behavior, and exact step IDs. Require no
   monotonic leak and exactly-once application per actor version.
8. Rebuild both processes from the saved one-step checkpoint and verify the
   same actor version is restored and re-synchronized once. No scientific GRPO
   batch follows until this resume check passes.

For a same-host two-GPU topology, also compare direct peer/NCCL sync with the
known full-export-plus-engine-rebuild control. For two independent instances
without private reachability, record the topology limitation, validate the
export/reload oracle through durable artifact transfer, and defer the live-sync
claim rather than weakening the gate.

### Required Evidence And Stop Conditions

Each run writes a human-readable log, machine-readable manifest, environment
lock, topology receipt, tensor ledger, timing JSON, GPU-memory samples, dispatch
receipts, and checkpoint/source hashes. Pull compact evidence locally and
commit source/docs only; never commit weights, exports, caches, or checkpoints.

Stop before GRPO and report if either GPU lacks `sm_120`, Torch/vLLM ABI differs,
the custom model falls back, fresh compile or full-model parity fails, 32 GB
VRAM cannot hold the conservative actor/rollout role, direct sync differs from
fresh reload, synchronization requires the public SSH gateway, or any rank
hangs. Passing this plan authorizes only a later named two-GPU one-batch GRPO
canary; it does not authorize the 20/50/98/196/391 ladder.

## Owner Decision: Parallel Evaluation Only

Future full evaluations must not use the historical single-GPU serial-HF wall
as the operational path. Existing serial-HF scores remain immutable historical
anchors, but no new model-selection wave should spend another approximately
8h10m per model on that implementation. A future evaluator must take one of two
explicitly labelled paths:

1. **Strict parallel-HF bridge.** Preserve the exact HF model, decode, prompt,
   seed, extraction, and grading semantics while identity-sharding the dataset
   across independent TP=1 GPU workers. Merge by expected item/sample identity,
   reject duplicates or gaps, and prove the merged result matches a bounded
   unsharded control. This removes serialization without changing backend.
2. **All-model vLLM protocol.** Evaluate untuned base, whole-layer baseline,
   SHS, TriGLU, and OFT under one pinned vLLM implementation and compare only
   within that protocol. Keep historical HF values in a separate column; never
   mix a vLLM score from one model with an HF score from another as if backend
   were controlled.

The checkpoint-by-backend diagnosis remains incomplete:

| Checkpoint state | Parallel HF | vLLM |
|---|---|---|
| Initial SHS exact-no-op state | A: pending full cell | B: pending full cell |
| Final SHS SFT checkpoint | C: complete historical cell | D: complete full shadow cell |

The RTX 5090 bounded fixed-prompt parity is a smoke and does not complete A or
B. Bacon implemented parity/evaluator scaffolding, Leibniz completed D, and no
agent has yet completed the full four-cell matrix. The remaining cells must use
the same preregistered identities and record paired text, extraction, score,
length, cap, and log-probability diagnostics.

Proposed run identities, requiring explicit approval before launch:

| Purpose | Run ID |
|---|---|
| Complete parallel checkpoint/backend matrix | `shs_checkpoint_backend_matrix_parallel_20260712_v1` |
| Establish all-model vLLM comparison protocol | `qwen3_variants_vllm_parallel_eval_20260712_v1` |

On two RTX 5090s, first prove two-way identity sharding and exact aggregation.
On eight RTX 5090s, prefer eight independent TP=1 evaluator shards for one
model or explicitly partition the pool across models; choose from measured
tail latency and model-load amortization, not a fixed static split. Every shard
must emit resumable per-identity JSONL through a single writer or independently
named files followed by a deterministic merge, avoiding the earlier concurrent
JSONL interleaving defect.

vLLM is the rollout/serving/evaluation runtime, not the backward/optimizer
engine. `Use vLLM to train and serve` means that on-policy samples come from the
pinned vLLM rollout policy and updated actor weights are synchronized back to
it; PyTorch/veRL still computes log probabilities, loss, backward, and optimizer
updates. Production acceptance therefore also requires actor-versus-vLLM
log-probability/KL receipts rather than assuming identical distributions from
identical parameter files.

## General Custom-FFN vLLM Onboarding Program

SHS must become the first worked example of a reusable onboarding method, not a
one-off custom-model patch. The next architecture is TriGLU. Applying the method
to a structurally different FFN augmentation is the test that distinguishes a
real workflow from an SHS-specific recipe. The current scope is custom FFN,
adapter, normalization, and local residual changes that preserve Qwen3 causal
attention and KV-cache semantics; arbitrary attention/state architectures are
outside this first generalization claim.

### Phase 1: SHS Retrospective And Failure Ledger

Reconstruct the complete SHS path from reference module to bounded vLLM and
weight-sync execution. For every step, record the initial assumption, observed
failure, diagnosis, architecture-specific fix, and reusable invariant. At
minimum the retrospective covers:

1. replacing runtime surgery with explicit HF config/model construction;
2. self-describing export metadata, `auto_map`, architecture names, and pinned
   dimensions/seeds;
3. out-of-tree `vllm.general_plugins` registration and lazy model loading;
4. checkpoint-key ownership, duplicate base-linear references, missing and
   reconstructible buffers, and safetensors export;
5. persistent shuffled-map device placement and non-persistent derived buffers;
6. flattened dynamic token batches, Paged KV cache, chunked prefill, stop/cap,
   and heterogeneous-request execution;
7. explicit reference/custom backend selection and no-fallback receipts;
8. HF/vLLM numerical, greedy, sampled, and long-response semantic drift;
9. FSDP merge metadata, actor-to-rollout `update_weights`, deterministic-buffer
   exclusions, versioning, receiver receipts, and fresh-reload oracles;
10. resumable evaluation output, deterministic identity merge, and concurrent
    JSONL writer failure prevention.

The output is an expanded durable workflow in
`docs/runtime/custom-architecture-vllm-onboarding.md`, plus a compact failure
ledger that links each invariant to its source test and experiment receipt.

### Phase 2: Extract A Fixed Architecture-Neutral Contract

Factor only behavior proven common across at least SHS and TriGLU. The reusable
contract should provide:

- declarative serialization of variant name, target layers, dimensions,
  initialization/no-op contract, and variant parameters;
- explicit HF construction and a stable export/reload key manifest;
- plugin/registry helpers that do not require editing installed vLLM source;
- architecture-owned buffer classification and device-finalization hooks;
- reference-backend dispatch receipts and optional custom-kernel receipts;
- shared HF/vLLM parity, pressure, long-decode, memory, and stop/cap harnesses;
- actor-to-rollout tensor ledgers, versioned synchronization receipts, and
  fresh-reload comparison;
- identity-sharded evaluation, single-writer shard output, deterministic merge,
  resume, and completeness validation.

Do not force each FFN into one mathematical base class merely to reduce file
count. The stable abstraction is lifecycle and runtime contracts; SHS HyperGrid
math and TriGLU side-FFN math remain architecture-owned modules.

### Phase 3: Apply The Workflow To TriGLU

Use the completed Layer-10 TriGLU SFT checkpoint whose trainable-state SHA-256
is `8a463168b4dce0f698357a821dfaca2d7b7fa90032841adb717251e323c48ab8`.
The implementation retains the original SwiGLU and adds the TriGLU side FFN
with `side_dim=512`, `side_hidden=2048`, and an exact-zero initial side return.

Proceed through the same fixed ladder. The currently authorized transplantation
slice covers steps 1-6 plus a training-reference regression. Steps 7-8 remain
future gates because the owner explicitly deferred the Eval Parity Matrix and
did not authorize production GRPO in this wave:

1. serialize a unique TriGLU config/architecture and reconstruct it during HF
   `__init__`, with no hidden post-load surgery;
2. export/reload and audit exact expected/missing/unexpected/duplicate keys;
3. prove HF reference logits, greedy tokens, seeded samples, and no-op behavior;
4. register the explicit model through the out-of-tree vLLM plugin;
5. load TP=1 enforce-eager and prove pressure-1 HF parity, Paged KV, and device
   placement before increasing concurrency;
6. run heterogeneous pressure 1/8/16/32 and a production-length panel, recording
   aggregate generated tokens/s, latency, VRAM, lengths, caps, and receipts;
7. later, run a bounded versioned actor-to-rollout sync plus fresh-reload
   oracle as part of the production canary;
8. later, run the separately approved parallel-evaluation protocol rather than
   folding an Eval Parity Matrix into onboarding.

The first TriGLU path uses ordinary PyTorch/cuBLAS modules inside the vLLM
Transformers backend. A custom Triton kernel is not part of onboarding success
and is considered only after end-to-end profiling identifies a material TriGLU
bottleneck.

### Phase 3B: Explicit Pure-BF16 SHS And TriGLU Variants

The initial custom-runtime implementations intentionally preserve historical
mixed-precision behavior. TriGLU currently casts the side-branch input to FP32
and keeps its side modules in FP32. SHS keeps architecture-owned generator,
adapter, and scale state in FP32 while the native Qwen projections continue to
use their BF16 GEMMs. These statements concern the custom FFN paths; they do not
mean that the complete Qwen backbone executes in FP32.

Add a second, explicit and serializable dtype policy for each architecture:

- `reference_fp32_custom`: preserve the existing checkpoint/runtime semantics;
- `pure_bf16_custom`: keep custom-path parameters, inputs, intermediate
  activations, and returned deltas in BF16 without explicit FP32 upcasts.

Optimizer master state may remain FP32 and reductions may use backend-selected
accumulation; both must be reported separately from forward-runtime dtype. Do
not silently reinterpret historical checkpoints. Every export must include the
selected dtype policy, a per-component dtype manifest, conversion receipts, and
the source trainable-state hash. Both policies must preserve the exact initial
no-op invariant.

Run the same bounded gate for pure-BF16 TriGLU and pure-BF16 SHS:

1. load the historical checkpoint under its reference policy, construct the
   BF16 runtime copy explicitly, and verify expected/missing/unexpected keys;
2. record parameter, persistent-buffer, input, intermediate, output, and
   optimizer-state dtypes rather than inferring precision from autocast alone;
3. compare reference-policy and pure-BF16 HF logits, greedy tokens, fixed-seed
   samples, memory, and throughput under preregistered BF16 tolerances;
4. load the pure-BF16 export through each architecture's out-of-tree vLLM
   registration and prove the custom FFN path dispatched without fallback;
5. run HF-versus-vLLM greedy/logit parity plus pressure 1/8/16 and the existing
   matched long-decode cell using `max_num_batched_tokens=32768` and
   `gpu_memory_utilization=0.85`;
6. compare pure-BF16 TriGLU with the same-profile vanilla control and compare
   pure-BF16 SHS with its historical reference-PyTorch/cuBLAS anchor;
7. retain the reference policy as the production fallback if BF16 drift,
   instability, or quality receipts fail. This gate does not revive the
   strict-failed grouped SHS Triton backend.

Proposed experiment family and identities, pending explicit owner approval
before GPU launch:

| Purpose | Run ID | Screen |
|---|---|---|
| Shared dtype-policy/source gate | `custom_ffn_pure_bf16_vllm_gate_20260712_v1` | none; CPU/source task |
| TriGLU pure-BF16 vLLM smoke | `triglu_pure_bf16_vllm_smoke_20260712_v1` | `qwen_triglu_pure_bf16_smoke_20260712_v1` |
| SHS pure-BF16 vLLM smoke | `shs_pure_bf16_vllm_smoke_20260712_v1` | `qwen_shs_pure_bf16_smoke_20260712_v1` |

Use separate config, result, compact-metrics, and dated-record identities for
the TriGLU and SHS cells. Reuse the common onboarding helpers and existing
weights; do not create a second framework. The Eval Parity Matrix, full
benchmark evaluation, and 50/98-batch GRPO remain outside this dtype gate.

### Phase 3C: Registered SHS Generation-Interface Implementation

The TriGLU transplantation retrospective found that historical SHS vLLM runs
resolved generic `TransformersForCausalLM`; the registered SHS wrapper was not
actually exercised. Those runs remain valid evidence for the generic
Transformers execution path and their recorded throughput, but they are not
evidence that the intended registered SHS route satisfies vLLM's generation
interface. The TriGLU wave correctly treated this as report-only and reverted
its temporary out-of-scope SHS edit.

Create a separately approved SHS implementation gate that applies the reusable
TriGLU lifecycle fix without silently changing historical results:

1. preserve the generic historical SHS route as an immutable control;
2. make the registered SHS wrapper satisfy the causal-LM generation contract,
   using `TransformersForCausalLM` rather than a hidden-state-only wrapper where
   required by the pinned vLLM runtime;
3. select `model_impl=auto` and prove that vLLM resolves the explicit SHS
   architecture instead of generic `TransformersForCausalLM`;
4. reapply architecture-owned device/dtype finalization only after vLLM weight
   loading and conversion have completed;
5. emit a semantic receipt containing wrapper class, architecture, target
   layer, SHS dimensions/seeds, dtype policy, backend, device, and
   `fallback=false`;
6. verify export/reload keys, exact initial no-op behavior, HF reference logits,
   eight-token greedy parity, pressure 1/8/16, and one matched long-decode cell;
7. compare the registered route with the immutable generic control. Any speed
   or token difference must be reported as a new protocol result, not used to
   rewrite the historical SHS anchor;
8. keep the reference PyTorch/cuBLAS SHS projection backend. This gate does not
   reauthorize the strict-failed grouped Triton implementation.

The implementation should reuse the architecture-neutral export metadata,
plugin registration, post-load finalization, backend-neutral weight-shape
inspection, generation-interface tests, and dispatch-receipt helpers proven by
TriGLU. Any new issue that is not required for the SHS registered route remains
report-only unless the owner expands the exact scope.

Proposed identities, pending explicit owner approval before implementation or
GPU launch:

| Purpose | Proposed identity |
|---|---|
| Run ID | `shs_registered_generation_interface_smoke_20260712_v1` |
| Screen | `qwen_shs_registered_generation_smoke_20260712_v1` |
| Runtime config | `configs/runtime/shs_registered_generation_interface_smoke_20260712_v1.yaml` |
| Result directory | `runs/runtime_smokes/shs_registered_generation_interface_smoke_20260712_v1/` |
| Durable record | `docs/experiment_records/2026-07-12_shs-registered-generation-interface-record.md` |
| Compact metrics | `docs/experiment_records/compact_metrics/2026-07-12_shs_registered_generation_interface_smoke.json` |

This gate is independent of the pure-BF16 variants. It may test the historical
SHS reference dtype policy first, then reuse the confirmed registered route in
the separately approved BF16 gate. It does not include the Eval Parity Matrix,
actor synchronization, production GRPO, or full benchmark evaluation.

### Phase 4: Failure-Driven Workflow Revision

Every TriGLU failure must be classified before patching:

- **architecture-specific:** TriGLU math, state, initialization, or key layout;
- **workflow defect:** an implicit SHS assumption in export, registration,
  buffer ownership, flattened execution, synchronization, or evaluation;
- **backend limitation:** unsupported vLLM/Transformers behavior that requires a
  native out-of-tree model or a documented scope reduction.

When a workflow defect is found, fix the generic contract and tests first, then
rerun both the affected TriGLU cell and the corresponding SHS regression cell.
No TriGLU success may regress SHS checkpoint loading, continuous batching,
semantic receipts, or weight synchronization. Repeat until the same written
workflow onboards both architectures without undocumented manual repair.

### Completion Boundary And Approved Identities

The program is complete only when a third FFN variant can follow the documented
workflow with architecture-owned math plus declarative metadata, without
copying an entire SHS runtime stack. Until that triangulation, claims are limited
to `SHS and TriGLU supported by the common FFN onboarding contract`.

The owner approved the following identities for execution on 2026-07-12:

| Purpose | Run ID | Screen |
|---|---|---|
| SHS retrospective and reusable-contract extraction | `custom_ffn_vllm_onboarding_methodology_20260712_v1` | none; CPU/source task |
| TriGLU vLLM onboarding smoke | `triglu_vllm_onboarding_smoke_20260712_v1` | `qwen_triglu_vllm_smoke_20260712_v1` |

Complete naming manifest:

| Artifact | Approved identity |
|---|---|
| Runtime config | `configs/runtime/triglu_vllm_onboarding_smoke_20260712_v1.yaml` |
| Python harness | `scripts/run_triglu_vllm_onboarding_smoke.py` |
| Launch wrapper | `scripts/run_triglu_vllm_onboarding_smoke_20260712_v1.sh` |
| Remote result directory | `runs/runtime_smokes/triglu_vllm_onboarding_smoke_20260712_v1/` |
| Durable record | `docs/experiment_records/2026-07-12_triglu-vllm-generality-transplantation-record.md` |
| Compact metrics | `docs/experiment_records/compact_metrics/2026-07-12_triglu_vllm_onboarding_smoke.json` |

The two-GPU harness assigns TriGLU to GPU 0 and vanilla Qwen to GPU 1 as
independent TP=1 replicas. It first proves explicit-HF export/reload parity,
then runs short pressures 1/8/16 and matched 800-1,024-token pressures 16/32/64
with the accepted 5090 scheduler starting point: `max_num_batched_tokens=32768`
and `gpu_memory_utilization=0.85`. The full evaluation matrix, 50/98-batch
GRPO, Triton research, and shutdown are explicitly out of scope.

For TriGLU, `model_impl=auto` is intentional: the exported architecture string
must resolve through the out-of-tree registry to the custom wrapper, which is
itself a vLLM Transformers-backend subclass. Forcing `model_impl=transformers`
selects generic `TransformersForCausalLM` and bypasses architecture-owned
post-load finalization.

Required deliverables are the revised durable onboarding guide, generic
contract/helpers and regression tests, explicit TriGLU HF/vLLM implementation,
compact run manifests, a dated experiment record, and a clear list of remaining
architecture-specific code. Commit source/docs/compact evidence only; never
commit base weights, SFT checkpoints, deployment exports, caches, or raw full
evaluation traces.

### TriGLU Transplantation Result

The approved transplantation completed with bounded status **PASS**. The final
custom route resolved `Qwen3TriGLUForCausalLM`, emitted one no-fallback Layer-10
FP32 dispatch receipt, matched the explicit HF eight-token greedy continuation,
and completed pressures 1/8/16 plus matched long-decode pressures 16/32/64.

At pressure 64, TriGLU reached 2,872.1 generated tok/s/GPU versus 5,442.1 for
vanilla Qwen and the historical 1,892.1 SHS anchor. TriGLU is 52.8% of vanilla
and 151.8% of SHS. It therefore does not qualify as near-baseline speed under
the provisional 70-80% threshold, but baseline plus TriGLU remain the preferred
low-cost first pair for production canaries before SHS. The one/two/20-batch
gates still apply; this result does not authorize 50/98-batch GRPO.

The run corrected two reusable workflow defects in the TriGLU path: registered
generative wrappers must inherit vLLM `TransformersForCausalLM`, and
`model_impl=auto` is required to resolve the registered architecture rather
than forcing the generic wrapper. Historical SHS measurements remain generic-
wrapper measurements and were not rerun here. The same concern was observed in
the registered SHS source, but the owner explicitly required SHS to remain
unchanged; it is report-only pending separate approval. Full details are in
`docs/experiment_records/2026-07-12_triglu-vllm-generality-transplantation-record.md`.

## RTX 5090 Production-Length Throughput Gate Result

The approved `shs_2x5090_production_length_throughput_20260712_v1` gate is
complete. Two concurrent TP=1 RTX 5090 replicas ran the reference SHS backend
at pressure 16, 32, and 64. The final pressure-64 production cell used real
Numina prompts, group size four, temperature 1.0, top-p 1.0, and cap 3072. It
produced a representative mean length of 870.5 tokens and cap-hit rate 10.16%.

The measured pair rate was only 446.75 generated tokens/s. The matched
800-1024 pressure-64 cell reached 494.3 tokens/s/GPU, 21.84% of the RTX PRO
6000 C1 anchor. Engine profiling explains much of the gap:
`max_num_batched_tokens=131072` consumed approximately 20.6 GiB of peak
activation space and left only 1.22 GiB / 11,440 tokens of KV cache. The
current 5090 setup is scheduler/KV constrained and is rejected for economical
GRPO despite passing correctness, distribution, dispatch, and OOM gates.

The next runtime experiment requires a new approved name. It should reduce
`max_num_batched_tokens` to approximately 32,768, allocate substantially more
KV cache, and rerun the same matched and production pressure-64 cells. Do not
launch GRPO from the current 125.5-hour pure-rollout projection for 391 batches.
The dated record and compact JSON are the authoritative receipts.

### RTX 5090 KV-Balanced Follow-Up Result

The approved `shs_2x5090_kv_balanced_throughput_20260712_v1` follow-up is
complete. Reducing `max_num_batched_tokens` from 131,072 to 32,768 and raising
memory utilization from 0.80 to 0.85 reduced the vLLM activation profile from
20.6 to 5.16 GiB and increased KV capacity from 11,440 to 170,720 tokens.
Reported full-4096 concurrency increased from 2.79x to 41.68x, so the
conditional 16,384 profile was correctly not executed.

The controlled matched pressure-64 cell improved from 959.0 to 3,742.1 pair
tokens/s, a 3.90x configuration speedup. Mean per-GPU throughput reached
1,892.1 tokens/s, or 83.62% of the RTX PRO 6000 C1 anchor. The production cell
improved from 446.8 to 899.1 pair tokens/s, but its mean sampled length was
668.7 and missed the existing distribution boundary. Identical per-request
seeds did not preserve high-pressure sampled traces under the PyTorch-native
top-p fallback, so the final status remains `degraded` and the clean method
comparison is the matched-length cell.

This result confirms that the original 5090 control was KV/scheduler starved,
not a compute-ceiling measurement. The current operational-distribution
projection is 9.57 minutes of pure rollout per global batch and 62.36 pure-
rollout hours for 391 batches on eight 5090s. It still excludes actor, log-prob,
reward, synchronization, checkpoint, and evaluation work and does not authorize
production GRPO. A separate async instrumentation protocol is required for
per-request latency because synchronous vLLM V1 returned null timestamps.

### Owner Throughput Closeout Decision

The owner has ended additional standalone throughput testing and accepted the
current evidence as sufficient for concept-level planning. Select the
32,768/.85 KV-balanced profile for production staging. Do not run the optional
16,384 profile, a queue-depth-only follow-up, a FlashInfer-only benchmark, or
separate async latency instrumentation before the first production canary.

The planning envelope is intentionally broad: approximately 15 hours of pure
rollout at matched sustained capability versus 62.36 hours under the measured
finite stochastic distribution for the full 391-batch eight-RTX-5090 wave.
Do not manufacture a narrower prelaunch estimate. Instrument the real canary's
rollout, reward, old/reference log probabilities, actor update, synchronization,
checkpoint, and shell wall, then replace the envelope with cumulative live ETA.

This decision moves work from performance characterization to production
launch preparation; it does not waive correctness. The next GPU run must have a
new explicitly approved production/canary identity, preserve exact resume and
data-order contracts, begin bounded rather than at 391 batches, and retain the
existing actor/vLLM semantic, weight-sync, memory, cleanup, and evaluation
risks as monitored stop conditions.

## Master Pre-50-Batch Critical Path

This section is the current condensed execution order and supersedes older
four-GPU wording where it conflicts with the owner's selected eight-RTX-5090
production direction. Throughput characterization is closed. The selected
starting rollout profile is `max_num_batched_tokens=32768` and
`gpu_memory_utilization=0.85`; no additional standalone throughput benchmark is
required.

### Mandatory Before The First Production Canary

1. **Freeze the scientific initialization contract.** The primary comparison is
   naive Layer-10 RL versus exact-no-op SHS Layer-10 RL. Both start from the
   same untuned Qwen revision, seed, prompt order, group membership, reward,
   optimizer schedule, and number of updates. The completed SHS SFT checkpoint
   remains an engineering/runtime stress artifact unless a separately named
   SFT-initialized RL study is approved.
2. **Audit and parameterize eight-GPU topology.** Remove or gate every implicit
   one-, two-, or four-GPU assumption in launchers, veRL overrides, FSDP/DDP
   ranks, rollout replicas, effective global batch, microbatch/accumulation,
   dataset sharding, output ownership, checkpoint ownership, monitor expected
   processes, and resume world size. Emit a fail-fast topology/batch-arithmetic
   receipt before model load.
3. **Create the production canary contract.** Propose and obtain approval for a
   new run/config/screen/log/output naming set. Pin source commit, model/export
   hashes, dataset/provenance hashes, vLLM environment, KV-balanced settings,
   sparse checkpoint cadence, human-readable monitoring, and economic stop
   conditions. Do not reuse a smoke or throughput run identity.
4. **Preserve exact data order across variants and ranks.** Materialize the
   global prompt/group ledger independently of world size, then prove that rank
   shards are disjoint, complete, deterministic, and reconstruct the same
   global order. Changing GPU count must not change effective batch size or
   silently drop/repeat a tail.

### Mandatory During One- And Two-Batch Canaries

1. **Actor-versus-rollout on-policy receipt.** For vLLM-generated tokens,
   recompute actor/old-policy log probabilities and record per-token/sequence
   deltas, approximate KL, importance-ratio distribution, clip fraction, and
   nonfinite/outlier counts. This is mandatory before a 20-batch pilot; similar
   benchmark accuracy alone cannot clear effective off-policy drift.
2. **Eight-rank weight-refresh evidence.** Record actor version, exact 29-tensor
   payload ledger, exclusion of the 12 deterministic SHS block-ID buffers,
   receiver-side version/hash/apply timing, barriers, and post-sync dispatch.
   Compare a bounded direct-sync result with a fresh-reload oracle. The earlier
   23/29 BF16-observable result is not an SHS-learning failure, but the receiver
   path still needs versioned production receipts.
3. **Memory and full-component timing.** Measure rollout, reward/extraction,
   old and reference log probabilities, actor forward/backward/update,
   collectives, synchronization, checkpoint, and shell wall. Validate actor
   headroom on 32 GB cards with real response lengths; the prior short gate
   reserved nearly all actor memory.
4. **Checkpoint, interruption, and cleanup.** Prove exactly-once step tracking,
   eight-rank checkpoint completeness, deterministic resume without repeated
   optimizer steps, and safe worker/Ray teardown. The reproducible post-save
   DataLoader-worker warning must be fixed or explicitly contained so it cannot
   hide a real failure.

### Mandatory Before A 50-Batch Scientific Run

1. **Complete the 20-batch pilot and review.** Inspect reward trend and source
   mix, KL/ratio/clip behavior, response lengths and cap hits, extraction and
   verifier failures, gradient/update norms, per-component timing, memory,
   sync receipts, checkpoint/resume, and rank imbalance. A 50-batch wave starts
   only from a written go/no-go review.
2. **Complete the evaluator decision.** Finish the pending initial exact-no-op
   SHS cells A/B in the checkpoint-by-backend matrix and establish either the
   strict identity-sharded parallel-HF bridge or a separately labelled all-model
   vLLM protocol. No future full single-GPU serial evaluator is allowed.
3. **Classify existing C2 response failures.** Audit the saved 512-response
   shard into valid-but-wrong, extraction/format failure, cap hit,
   repetition/degeneration, invalid LaTeX, empty answer, and verifier/reward
   categories. This is a CPU/artifact task and should not consume rollout GPU
   time. It must distinguish ordinary model error from runtime or reward-pipeline
   defects before interpreting a 50-batch trend.
4. **Parallel-evaluator smoke and resume.** Prove deterministic item/sample
   sharding, per-rank single-writer output, gap/duplicate rejection, incremental
   merge, partial human-readable accuracy, and idempotent resume. A full
   all-model evaluation can run after or alongside training once this plumbing
   gate passes.

### Parallel And Variant-Selection Tracks

- SHS retrospective, generic custom-FFN onboarding contract, and TriGLU vLLM
  onboarding are now a critical variant-selection input. If matched TriGLU
  decode is close to vanilla Qwen, baseline plus TriGLU become the preferred
  low-cost candidates for one/two/20-batch canaries and subsequent 50-batch
  comparison; SHS remains a more expressive but more expensive follow-up.
- TriGLU custom kernels, SHS grouped-Triton strict-logit repair, fused actor
  backward, FlashInfer sampling, async request-latency instrumentation, and
  further throughput optimization. Reopen only from production profiling or a
  changed scientific requirement.
- OFT checkpoint-curve and pure-OFT diagnostics. The current mixed OFT SFT
  configuration is not a production RL candidate.

### Baseline And TriGLU Two-GPU Prelaunch Result

The approved
`baseline_triglu_2x5090_grpo_prelaunch_gate_20260712_v1` gate is complete and
recorded in
`docs/experiment_records/2026-07-12_baseline-triglu-2x5090-grpo-prelaunch-gate-record.md`.
Both variants started from the same untuned Qwen3-1.7B-Base revision and passed
the bounded actor-vLLM on-policy, production reward-wiring, two-step exact
resume, memory, and timing gates. TriGLU additionally passed a 23-tensor live
weight refresh with sender/receiver hashes and a bit-exact greedy/logprob fresh-
reload oracle. Its final trainable contract includes all 12 side tensors plus
the 11 selected Layer-10 backbone tensors.

This closes the two-GPU prelaunch work for baseline and TriGLU. The next
blocking execution is the eight-GPU topology/data-order preflight and one
production-shaped global batch. The result does not authorize 20, 50, or 98
batches, and all three mandatory PENDING registry items remain carried forward.

### Current Launch Ladder

```text
eight-GPU topology/data-order/config preflight
-> one complete global-batch canary
-> two-batch interruption/resume canary
-> actor/vLLM + sync + component-timing review
-> 20-batch pilot
-> evaluator/failure-audit + 20-batch go/no-go review
-> 50-batch scientific run
-> 98/196/391 continuation only from observed trends and economics
```
