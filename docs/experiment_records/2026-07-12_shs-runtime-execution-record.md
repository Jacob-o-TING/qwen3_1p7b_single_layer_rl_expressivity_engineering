# 2026-07-12 SHS Runtime Execution Record

Date: 2026-07-12

Status: runtime execution substantially operational; strict built definition
not reached because the full-model logits cosine gate failed.

## Fixed Inputs And Environment

- NEW kernel-development clone, SSH port 36348 only.
- NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB; driver 580.82.09.
- PyTorch 2.8.0+cu128; Triton 3.4.0; vLLM 0.10.2; Transformers 4.57.1.
- Base: `/root/autodl-tmp/qwen3_single_layer_rl/models/Qwen3-1.7B-Base`.
- Completed SHS checkpoint: `runs/sft_ordered_20260711_sft50k_v1/layer10_whole_layer_shs/checkpoints/step_00003916`.
- Trainable-state SHA-256: `ebb4cd92f6e890c17cf0e14a883557358dff927e901b69588f1867e6dd016712`.
- Trainer-state SHA-256: `396e87c52a37105500d49f2381f2123886e733efab5bc8419953906ef65e8815`.
- Runtime seed: 20260712.

Every run wrote a preregistered manifest before execution. Compact files were
pulled under `runs/runtime_smokes/`; exports, weights, optimizer payloads, and
datasets remain remote and ignored.

## Implementation Corrections

Execution found and fixed four deployment-contract defects:

1. Base projections were registered twice through `base_mlp` and
   `shs.*.base_linear`, making safetensors export ambiguous. SHS now keeps a
   non-registered strong reference and rebinds after vLLM linear replacement.
2. Custom loading cast the FP32 SHS generator/adapters to BF16. The explicit
   classes now preserve the canonical mixed-precision contract.
3. The export now provides `Qwen3SHSModel` for AutoModel, standard local
   remote-code shims, and deterministic map-buffer hashes. Reconstructible map
   buffers are omitted from safetensors because the generic vLLM loader handles
   parameters rather than buffers.
4. Runtime buffer migration now occurs before local map capture. Child-process
   JSONL receipts record actual Triton dispatch and buffer devices.

The final runtime uses the native base GEMM plus one grouped Triton
multiplicative-delta kernel. A prepacked deterministic column permutation lets
the kernel follow the 32-column-group BF16 reduction order in one launch. The
independent additive path remains reference code. No backward kernel exists.

## Gate 1: Full-Model Export And Parity

Run/screen: `shs_fullmodel_kernel_parity_20260712_v1` /
`qwen_shs_fullmodel_parity_20260712_v1`.

Result: **failed** only on the preregistered full-logits cosine threshold.

- disposable export: about 3.4 GB; canonical checkpoint unchanged;
- export approximately 2.6 seconds; explicit HF load approximately 1.0-1.2 seconds;
- missing, unexpected, mismatched, and overlay-missing keys: zero;
- deterministic map hashes matched; SHS buffers were on `cuda:0`;
- actual dispatch: Triton for all three projections;
- Layer-10 max/mean absolute difference: 0.0625 / 0.000656;
- Layer-10 mixed-tolerance mismatches: 0 of 36,864;
- logits max/mean absolute difference: 0.3125 / 0.037274;
- logits relative L2: 0.004647, below the 0.01 limit;
- logits top-1 and greedy token IDs/text: equal;
- logits cosine: 0.999755, below the fixed 0.9999 minimum;
- diagnostic mixed-tolerance misses: 264 of 2,734,848 logits.

Earlier compact logs preserve shared-serialization, dtype, AutoModel, buffer,
device-reference, and numerical failures. Thresholds were not relaxed to force
a pass.

## Gate 2: vLLM Continuous Batching

Run/screen: `shs_vllm_contbatch_rollout_smoke_20260712_v1` /
`qwen_shs_vllm_rollout_smoke_20260712_v1`.

Result: **passed** on the grouped-kernel revision.

- vLLM V1 Transformers backend, TP=1, enforce-eager;
- FlashAttention, Paged KV cache, and chunked prefill active;
- engine load: 20.216 seconds; weights: 3.247 GiB;
- steady memory approximately 25.7 GiB;
- one-prompt token IDs exactly matched HF reference;
- child-process dispatch receipts: 3/3 Triton;
- heterogeneous pressures 1/8/16/32 all completed.

| Pressure | Full wall s | Prefill proxy tok/s | Decode proxy tok/s |
|---:|---:|---:|---:|
| 1 | 0.137 | 223.7 | 45.8 |
| 8 | 0.390 | 1,779.4 | 189.6 |
| 16 | 0.420 | 2,417.7 | 425.5 |
| 32 | 0.500 | 3,114.9 | 777.9 |

The synchronous API returned null native TTFT timestamps. The manifest labels
prefill as a separate uncached one-token probe and decode as residual full wall
minus that probe. These are phase proxies, not scheduler-native timers.

## Gate 3: Reference SFT Integration

Run/screen: `shs_triton_sft_step_smoke_20260712_v1` /
`qwen_shs_triton_sft_smoke_20260712_v1`.

Result: **passed**, honestly labelled `reference_training_integration`.

- validation-cache item 0, tail 256 tokens; deterministic math SDPA;
- step-1 loss 2.004079; step time 0.606 seconds;
- gradient norm 3.2466; 26 parameters changed;
- checkpoint reload maximum parameter difference: 0;
- original/resumed step-2 loss: 1.98854124546 for both;
- step-2 final maximum parameter difference: 0;
- peak allocated memory: 10,281,164,288 bytes.

The manifest records `triton_forward=false`, `triton_backward=false`, and
`optimized_training=false`. This proves only that reference training,
optimizer, checkpoint, and resume were not regressed.

## Gate 4: Tiny One-Group GRPO

Run/screen: `shs_triton_grpo_onebatch_smoke_20260712_v1` /
`qwen_shs_triton_grpo_smoke_20260712_v1`.

Result: **passed** on the grouped-kernel revision.

This is a plumbing smoke, not production veRL. The repo's veRL entrypoint
remains explicitly marked as a placeholder. The smoke used prompt `37 * 19`,
G=4, cap 16, temperature 0.8, and a preregistered numeric-distance reward plus
0.001 lexical-diversity smoke shaping.

- initial rollout: four samples and 3/3 Triton receipts;
- old/reference log probabilities: exact match;
- advantages: nonzero variance;
- policy loss 3.94548e-6; initial-policy KL 0;
- actor gradient norm 0.50077; update 0.359 seconds;
- 26 parameters changed; max update 7.6294e-6;
- checkpoint resume maximum parameter difference: 0;
- updated export 3.509 seconds;
- sync method: full export plus engine rebuild;
- rebuild plus post-sync generation 21.689 seconds;
- post-sync dispatch: 3/3 Triton;
- total shell wall: 55 seconds.

No 20-, 50-, or full production wave was launched.

## Decision And Remaining Blockers

The architecture now has a real export, explicit HF classes, vLLM
Transformers-backend loading, Paged KV execution, heterogeneous continuous
batching, Triton receipts, a reference training step, a tiny GRPO update,
checkpoint resume, and rebuild-based weight synchronization.

## Track A1: Deep-Logit Drift Localization

Run/screen: `shs_triton_drift_localize_20260712_v1`.

Result: **localized; fast Triton variants failed the unchanged 0.9999 logits-
cosine gate, while the strict reference-equivalent backend passed exactly**.

The matched run captured gate/up/down projection outputs, the Layer-10 SwiGLU
product, the Layer-10 residual, every later block residual, and final logits.
It compared the current grouped BF16 Triton delta, a grouped Triton delta with
FP32 accumulation through the final store, and the canonical 32-sliced-GEMM
reference reduction order. All projection dispatches matched their requested
backend; no fallback occurred. Resolved topology was one visible RTX PRO 6000,
world size 1, TP=1, and one replica.

For the original 18-token full-context parity prompt, grouped BF16 began with
small projection differences and a Layer-10 residual relative L2 of
`3.5820e-5`. The first greater-than-2x amplification occurred at Layer 11.
Relative L2 reached `0.0026048` at Layer 20, `0.0096999` at Layer 27, and
`0.0046073` at final logits; logits cosine was `0.9997475`. On the exact
32-flattened-token panel, final relative L2/cosine were `0.0043974` and
`0.9993597`.

FP32 delta accumulation did not remove the drift. Its full-context final
relative L2/cosine were `0.0048959` and `0.9997367`; the 32-token values were
`0.0049047` and `0.9993603`. This rules out BF16 accumulation of the 32 group
partials as the primary cause. The remaining difference begins in the grouped
inner reductions, is mostly hidden by the Layer-10 residual, and is amplified
by later transformer blocks. The reference-equivalent backend was exactly
equal at every capture point and remains the strict-parity backend. The grouped
Triton implementations remain separately labelled fast candidates and are not
strict-built.

A preliminary one-token-only development diagnostic quantized the Layer-10
differences away at the residual and therefore did not reproduce deep drift.
Its evidence is retained under a clearly labelled development directory and
was not used for the formal conclusion.

## Track A2: Triton-Forward Reference-Recompute Autograd

Run/screen: `shs_triton_autograd_parity_20260712_v1`.

Result: **training mechanics operational; strict end-to-end gradient parity
failed because the known fast-forward drift changes the upstream gradient**.

The custom autograd integration executes the grouped Triton multiplicative
forward and recomputes the canonical PyTorch grouped algebra in backward. It is
explicitly labelled `triton_forward_reference_recompute_backward` and
`custom_backward=false`. Base gate/up/down GEMMs and the additive low-rank path
remain normal PyTorch autograd. Focused CUDA tests established bit-exact local
VJP gradients for input, weight, and multiplicative grid when supplied the same
upstream gradient. A full SHS-wrapper test covered base projection weights,
the grid generator, multiplicative scales, additive scales, and additive
left/right factors; all nine focused kernel/module tests passed.

The formal real-checkpoint run used validation-cache item 0, tail 256 tokens,
and deterministic math SDPA. Reference and Triton model losses were
`1.9963317` and `1.9974571`, an absolute difference of `0.0011255`, below the
preregistered `0.01` loss limit. FP32/BF16 cross-entropies were recorded for
both paths. Reference and Triton forward/backward took `0.6198` and `0.5689`
seconds in this diagnostic; peak allocated memory was 9,736,268,288 bytes.

All requested gradients were present and finite, and all three projections
reported Triton-forward/reference-recompute dispatch with no fallback. Under
the full-model loss, however, 12 tensors crossed the preregistered combined
gradient gate of cosine below `0.999` and relative L2 above `0.05`. The worst
cosine was `0.9982286` for down-path `add_right`; the worst relative L2 was
`0.0833333` for down `mul_scale`. This is consistent with Track A1: the local
backward is exact for a fixed upstream gradient, while later-block forward
amplification changes the loss gradient arriving at Layer 10. The gradient
threshold was not relaxed.

The Triton path completed an optimizer update with 26 changed parameters.
Sparse checkpoint reload had zero parameter difference, and original versus
resumed second-step loss and final parameters matched exactly. The path is an
honest training-capable fast candidate, but it does not satisfy strict
production-candidate parity until the fast forward meets the A1 gate. Matched
SFT timing may proceed with this limitation explicit and with reference cells
as the production-correct anchor.

## Track B: Matched 50-Step SFT Runtime Matrix

Run/screen: `shs_sft_runtime_matrix_20260712_v1`.

Result: **passed as a matched timing matrix; reference eager remains the
production-correct and fastest path**.

All four cells loaded the same completed SHS checkpoint, used the deterministic
production train-item order, sequence length 2048, microbatch 1, gradient
accumulation 8, AdamW, BF16 model contract, and one resolved GPU process. Each
cell ran one correctness/cold step, five warmup steps, and 50 measured optimizer
steps. The preregistered topology preflight resolved one visible GPU and
`nproc_per_node=1`. Every cell reported 3/3 requested projection backends with
no fallback. Triton cells explicitly recorded `custom_backward=false`.

| Cell | Cold s | Median s | p10 s | p90 s | Assistant tok/s | Peak GB | Projected 3,916-step loop |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reference eager | 2.329 | 1.4526 | 1.4447 | 1.4571 | 7,730.0 | 18.44 | 1.58 h |
| Reference compile | 105.714 | 1.4698 | 1.4686 | 1.4722 | 7,638.3 | 11.38 | 1.60 h |
| Triton forward / reference-recompute backward eager | 4.901 | 3.5948 | 3.5897 | 3.6094 | 3,122.5 | 14.85 | 3.91 h |
| Triton forward / reference-recompute backward compile | 49.253 | 3.3034 | 3.2983 | 3.3081 | 3,399.1 | 9.78 | 3.59 h |

The 1.58-hour reference-eager loop projection closely reproduces the existing
approximately 1 hour 37 minute production anchor. It excludes validation,
checkpoint, and outer orchestration overhead and therefore remains a train-loop
estimate rather than a shell-wall promise. Reference `torch.compile` was a
slight steady-state regression and incurred about 106 seconds of cold compile,
so it has no measured break-even under this contract.

Reference recompute makes the Triton training path 2.47x slower than reference
eager. Compilation recovers about 8.1% of that path and reduces allocated
memory, but it remains 2.27x slower than reference eager. Dynamo recorded 17
unique graphs and 12 `Tensor.item()` graph breaks at static group-offset
extraction in recompute backward. This is a concrete composition optimization,
but eliminating it cannot plausibly close the full gap alone. Actor backward is
therefore a material bottleneck, and a custom multiplicative backward is now
profile-justified if a faster kernel-enabled actor path is required. Until then,
reference eager is the reliable SFT runtime and the Triton cells are non-strict
engineering candidates, not speedups.

## Track C1: Matched Long-Response vLLM Matrix

Run/screen: `shs_vllm_matched_longdecode_20260712_v1`.

Result: **passed; SHS reference is both strict-correct and faster than grouped
Triton at production-relevant high pressure**.

The fixed panel used the first 64 deterministic Numina validation problems,
temperature 0.8, top-p 0.95, minimum 800 generated tokens, cap 1024, and
pressures 1/8/16/32/64. All cells used TP=1, max 64 sequences, Paged KV cache,
chunked prefill, no prefix caching, and the same per-request seeds. The cells
were naive Qwen vLLM, SHS reference vLLM, and SHS grouped-Triton-fast vLLM. A
separate strict-Triton cell was honestly unavailable because Track A1 failed
the fixed parity gate. SHS cells emitted exactly 3/3 requested dispatch
receipts with no fallback.

| Cell | P1 tok/s | P8 tok/s | P16 tok/s | P32 tok/s | P64 tok/s |
|---|---:|---:|---:|---:|---:|
| Naive Qwen | 143.5 | 952.1 | 1,878.8 | 3,373.1 | 5,329.7 |
| SHS reference | 40.2 | 312.1 | 615.6 | 1,181.0 | 2,262.8 |
| SHS grouped Triton fast | 48.5 | 350.8 | 686.5 | 1,181.7 | 1,725.1 |

At pressure 64, the three cells generated 62,890, 63,386, and 62,714 tokens,
with 47, 47, and 44 cap hits; the length mix therefore does not explain the
Triton regression. SHS reference completed in 28.013 seconds versus 36.354
seconds for grouped Triton. The grouped kernel helps at low pressure, ties at
pressure 32, and loses at pressure 64 because its per-token/output-tile launch
shape scales poorly while the 32 sliced reference GEMMs become efficient at
large flattened-token batches. Isolated operator or short cap-16 results must
not be extrapolated to production long decode.

The reference-vLLM cell initially exposed a CPU map-buffer defect because only
the Triton branch migrated runtime buffers. Buffer migration is now a common
backend invariant, after which reference engine warmup and all pressures
completed. The synchronous vLLM API again returned null per-request timestamps,
so scheduler-native latency p50/p90/p99 remain unavailable rather than inferred.

For provisional pure decode only, four independent SHS-reference replicas at
the measured pressure-64 rate provide about 9,051 generated tokens/s. A
512-prompt, group-size-4 global batch with 800-1,000 tokens per response would
therefore spend roughly 181-226 seconds in pure decode under a matched length
and pressure distribution. Reward, log probabilities, actor work, weight sync,
checkpointing, and distributed imbalance are not included; Track C2 must
measure them before this becomes an RL wall-time estimate.

### C1 Budget Projection And Runtime Decision

The matched C1 result supersedes the earlier short pressure-32 proxy used in
planning. The current runtime decision is to retain the strict-correct SHS
reference PyTorch/cuBLAS projection inside vLLM continuous batching. This does
not mean returning to serial `transformers.generate()`: Paged KV cache, vLLM
scheduling, and one TP=1 replica per GPU remain part of the selected path.

| Budget | SHS-reference pure rollout | Provisional rollout wall |
|---:|---:|---:|
| 20 batches | 1.0-1.3 h | 1.2-1.6 h |
| 50 batches | 2.5-3.1 h | 3-4 h |
| 98 batches | 4.9-6.2 h | 6-8 h |
| 391 batches | 19.7-24.6 h | 23-30 h |

The rightmost column adds only a scheduling/prefill/imbalance margin. It does
not include reward, old/reference log probabilities, actor training,
checkpointing, or weight synchronization. Before the C2 production-shaped
shard, the complete 391-batch GRPO estimate remains approximately 40-70 hours
and must be labelled provisional.

Naive Qwen reached 5,329.7 tokens/s per GPU at the same pressure, implying
77-96 seconds per global-batch rollout and 8.4-10.4 pure rollout hours over 391
batches on four GPUs. The measured high-pressure SHS-reference rollout is about
2.35x slower. Further kernel work is therefore justified, but the grouped
kernel's small-M benefit cannot support production selection. Future work
should target large flattened-token batches or test an explicitly dispatched
hybrid that uses Triton only below a measured M threshold and cuBLAS/reference
at high pressure.

## Track C2: Production-Shaped One-GPU Shard Readiness

Run ID: `shs_grpo_replica_shard_20260712_v1`.

Result: **blocked before launch by real trainer semantics; no misleading shard
was executed**.

An executable preflight confirmed that `verl.trainer.main_ppo`, both production
parquet files, and the completed SHS trainable checkpoint are present. The
remaining blockers are integration code rather than missing compute or data:

- the configured reward is still `string_exact_placeholder`, and the runtime
  does not contain the production `math_verify`/LaTeX verifier;
- veRL `external_lib` imports `verl_model_hook.py`, but that module has no
  import-time registration/call site and therefore does not apply SHS surgery
  or the Layer-10 freeze policy to the real actor;
- the hook does not overlay the completed SHS `trainable_state.pt`;
- passing the completed SHS deployment export directly would load the SHS
  architecture but bypass the intended freeze/overlay contract and risks
  double injection if the hook is later activated;
- the active GRPO config still selects synchronous HF rollout rather than the
  C1-selected SHS-reference vLLM path;
- actor-to-rollout SHS weight synchronization and the explicit one-GPU
  128-prompt, group-size-4 shard config are not implemented.

The readiness manifest therefore records `launch_attempted=false`,
`production_candidate=false`, and `production_ready=false`. Running veRL now
would train the wrong actor state with placeholder rewards, so additional GPU
time would not produce decision-grade evidence. The next gate is to implement
the actor construction/checkpoint overlay and production verifier first, then
wire SHS-reference vLLM synchronization and execute the one-GPU shard. The
four-GPU global batch, two-batch canary, and 20-batch pilot remain later
production-readiness gates under the updated plan.

It is still **not called built under the declared definition** because Gate 1
failed the fixed logits-cosine threshold. Next work should reduce deep-logit
drift without post-hoc tolerance changes. Separately, production GRPO still
needs real veRL integration, the production verifier/reward, a complete
configured global batch, and component timing before any budget ladder.

## Production Evaluator Native-vLLM Vertical Slice

Run/screen: `vllm_eval_single_gpu_parity_20260712_v1` /
`qwen_vllm_eval_parity_20260712_v1`.

Source commit: `90dc4435e5bf2e9a8cc14cddc75664a8d84089a0`.

The first attempt stopped before model load because the pinned vLLM/veRL
environment did not contain EvalScope. No GPU memory was allocated. Attempt 2
installed only the exact `evalscope==1.8.1` wheel without dependency resolution
and reused missing pure-Python dependencies from the existing Python 3.12
EvalScope environment through a trailing `.pth` entry. Import receipts confirmed
EvalScope 1.8.1, vLLM 0.10.2, torch 2.8.0+cu128, and that torch resolved from the
vLLM environment rather than the fallback environment.

The fixed tiny gate used the first two paper-pinned GSM8K items, one greedy
sample per item, cap 256, batch size 2, seed 20260707, and the same existing
EvalScope benchmark adapter, prompt template, extractor, numeric scorer, report
writer, and prediction/review JSONL pipeline for both backends.

| Contract | Result |
|---|---|
| Canonical item IDs | exact: `0, 1` on both paths |
| Sample IDs | exact: `0, 1` on both paths |
| Prompt text and prompt hashes | exact |
| Duplicates or omissions | none |
| Extracted answers | exact: `18`, `3` |
| Per-item scores | exact: `1.0`, `1.0` |
| Aggregate score | exact: `1.0` |
| Raw greedy text | not exact |

This is an extracted-answer and score parity pass, not token/text parity. The
raw generations differed in wording despite identical prompts and greedy
decoding, so the stricter raw-text result remains recorded as a failure rather
than being hidden by the scoring match.

HF used 10 seconds of shell phase time and 4.055 seconds inside generation.
The enforce-eager vLLM path used 81 seconds of shell phase time, including
13.269 seconds of engine load and 60.118 seconds inside the pressure-2
generation call. This tiny low-pressure result is a performance regression and
is not speed evidence. A non-eager sanity run and the approved realistic speed
panel remain required before any evaluator acceleration claim.

Both paths wrote append-only, fsynced generation receipts plus EvalScope
predictions, reviews, reports, and phase completion receipts. Requested and
actual backend matched with no fallback. A follow-up patch propagates the task
seed into `GenerateConfig`, reads EvalScope's actual `stop_seqs` field, excludes
padding from HF generated-token receipts, and labels zero-repeat AMC phases as
skipped. AMC Average@32 is not yet parity-ready: stable per-request seeds must
be derived from canonical `(benchmark, item_id, sample_id)` identities before
sampled resume or sharding can pass.

### Evaluator Identity, Seed, And Resume Follow-Up

The non-eager native-vLLM sanity cell used the same two GSM8K items and cap 256.
Engine load took 22.583 seconds, while 463 generated tokens completed in 1.339
seconds, or about 345.8 tokens/s at pressure 2. The 31-second shell phase was
therefore cold-load dominated. This identifies `enforce_eager` as the cause of
the earlier 60.118-second tiny generation call; non-eager remains the selected
path for the realistic speed gate.

EvalScope assigns stable `sample.id` and `group_id` values before prompt
formatting. The paper adapters now attach a content-neutral message metadata
identity with benchmark, item ID, repeat-local sample ID, and EvalScope sample
ID. The generator derives a per-request seed by hashing this identity together
with the preregistered base seed. This identity does not depend on submission or
completion order. vLLM receives one `SamplingParams` object per request; the HF
sampled parity path requires batch size 1 so each request can reset its own RNG.

A real AMC smoke used one item, two repeats, temperature 1.0, top-p 1.0, and
cap 64. Receipts contained identities `(paper_amc23, 0, 0)` and
`(paper_amc23, 0, 1)` with distinct derived seeds `1631926959` and
`1478039851`. EvalScope reviews contained group IDs `[0, 0]` and sample IDs
`[0, 1]`, with no duplicates or omissions. The cap-64 score was zero and is not
a quality result.

Complete-cache resume reported zero items to process and two fully cached,
with no generation-completed receipt. A copied partial cache retained one row;
resume reported one cached and one missing row. Completion order meant the
retained row was sample 1, so sample 0 was regenerated with its original seed
`1631926959`. The merged cache contained exactly two unique rows. Regenerated
raw text, extracted answer, and score were bitwise/equal to the original run;
the cached row was unchanged. This tiny sampled-resume gate passed.

EvalScope writes resumed predictions, reviews, and reports back into the
supplied cache directory; the new work directory receives the manifest,
generation receipts, and phase receipt. Replica ownership and merge tooling
must account for this behavior explicitly rather than assuming resumed rows are
materialized under the new work directory. Cold engine load still occurs even
for a fully complete cache and costs about 21 seconds in this environment.

### Topology-Neutral Replica Dry-Run

Run/screen: `vllm_eval_multireplica_dryrun_20260712_v1` /
`qwen_vllm_eval_multireplica_dryrun_20260712_v1`.

The physical preflight observed one visible GPU, requested one TP=1 replica,
and assigned all 51 synthetic expected identities to the single physical output
owner. A separate CPU simulation used seven logical ranks to verify that the
same code has no hidden four-GPU limit. Stable SHA-256 identity sharding produced
rank counts `7, 6, 4, 11, 8, 8, 7` after shuffled completion order.

All seven rank receipts were complete. The first atomic merge added 51 rows
with zero missing or unexpected identities. Repeating the merge added zero rows
and recorded 51 byte-equivalent idempotent skips. Focused tests also require a
fatal error for duplicate rows within a rank, wrong-rank ownership, unexpected
identities, and conflicting rows. The dry-run took two seconds and allocated no
GPU memory. It is sharding/merge evidence only and is not a multi-GPU throughput
measurement.

### Baseline Evaluator Speed Evidence And Expanded Parity

Run/screen: `vllm_eval_single_gpu_speed_20260712_v1` /
`qwen_vllm_eval_speed_20260712_v1`.

All six cells completed without traceback. The live dashboard reported partial
review accuracy, completed rows, generated tokens, actual backend, and engine
load every 15 seconds. Matched main cells used eight items from each of the
three main benchmarks at cap 1024. Matched AMC cells used one item with 32
samples at temperature 1.0 and top-p 1.0. Production-pressure cells used 32
items from each main benchmark and four AMC items times 32 samples.

| Cell | Shell s | Engine load s | Phase s | Rows | Tokens | Phase tok/s |
|---|---:|---:|---:|---:|---:|---:|
| Matched HF main | 43 | instrumentation defect | 37.70 | 24 | 11,345 | 300.9 |
| Matched vLLM main | 41 | 21.15 | 13.61 | 24 | 11,051 | 812.1 |
| Matched HF AMC@32 | 176 | instrumentation defect | 169.97 | 32 | 13,472 | 79.3 |
| Matched vLLM AMC@32, pressure 1 | 99 | 21.22 | 72.16 | 32 | 16,177 | 224.2 |
| Production vLLM main | 57 | 22.84 | 27.50 | 96 | 49,479 | 1,799.2 |
| Production vLLM AMC@32 | 54 | 22.16 | 25.45 | 128 | 74,488 | 2,926.7 |

HF engine-load receipts incorrectly reported zero because timing began after
the HF model loader returned. Source instrumentation now measures that load;
the table does not relabel the observed zero as a direct measurement.

Scaling the two production-pressure vLLM phases linearly to 2,494 main rows and
1,280 AMC sampled rows gives 714.4 and 254.5 seconds. A 40-row AMC greedy proxy
adds 8.0 seconds and one engine load adds 22.8 seconds, for a provisional
baseline full-evaluation projection of 999.7 seconds, or 16.66 minutes. This
includes EvalScope extraction/report phase time but remains a projection.

The expanded matched panel failed semantic parity. Identity sets, prompt hashes,
review keys, and counts matched with zero duplicates. Only 6/24 raw greedy texts,
16/24 extracted answers, and 21/24 scores matched. MATH-500 accuracy was 7/8 on
both paths. GSM8K was 5/8 for HF versus 6/8 for vLLM. OlympiadBench was 1/8 on
both paths but two item outcomes swapped. Overall score was therefore 13/24 for
HF and 14/24 for vLLM. The differences are generation-trajectory differences,
not extraction or accounting errors. The fixed tiny two-item gate passed, but
the broader evidence blocks baseline full evaluation and any production label.

### SHS-Reference Evaluator Gate

The completed SHS SFT deployment export was traced to checkpoint
`step_00003916` and its existing manifest/hash contract. The first evaluator
attempt used non-eager compilation and failed before generation because
TorchDynamo cannot trace the dispatch receipt's Python `open` call. The retained
eager path matches the previously validated SHS runtime configuration.

The eager SHS-reference vLLM cell completed in 27 seconds: 14.11 seconds engine
load and 5.91 seconds generation for 386 tokens. It emitted exactly three
reference dispatch receipts, all buffers were on `cuda:0`, and no fallback was
reported. Runtime dispatch correctness therefore passed.

Semantic parity failed on the two GSM8K prompts. The serial HF checkpoint path
extracted `2` and `3`; SHS-reference vLLM extracted `18` and `3`. Scores differed
on the first item. A separate HF checkpoint-versus-HF-export audit was bitwise
token/text exact on both prompts, including extracted answers. The deployment
export is therefore faithful; the remaining mismatch is the vLLM runtime
trajectory, likely amplified by SHS dynamics. SHS-reference fast evaluation is
not labelled supported and no SHS full evaluation was launched.

### Real veRL Construction And Reward Gate

The next production-candidate gate now uses veRL's actual pre-FSDP hook callsite
in `verl/workers/fsdp_workers.py`. The project hook distinguishes a base Qwen
model that needs SHS injection from a preconstructed SHS deployment export,
rejects partial or double construction, overlays the completed
`trainable_state.pt` exactly once, validates its complete key set against the
Layer-10 freeze policy, and writes checkpoint hash/key/trainable audits. Actor
parameters remain trainable according to that policy; reference parameters use
the same overlaid state and are then fully frozen.

The generated veRL command requires the completed checkpoint for every
non-identity architecture. SHS vLLM rollout is pinned to the validated
`transformers` model implementation, eager execution, and the reference
PyTorch/cuBLAS projection backend. The readiness audit verifies both the actual
veRL hook callsite and veRL's actor-to-vLLM `update_weights` path from source;
neither is represented by a hard-coded pass.

The reward path now calls veRL's production `math_verify` adapter and fails
closed when that dependency is unavailable; there is no string-exact fallback.
`math-verify==0.9.0` was installed in `envs/vllm0102_verl061` after a dry run
showed it would add only that wheel. A real verifier smoke scored equivalent
LaTeX `\\frac{1}{2}` as 1 and `2` versus `3` as 0. Torch remained
`2.8.0+cu128` and vLLM remained `0.10.2`.

Focused remote tests passed 15 command/reward/freeze tests and two hook tests,
including preconstructed-export overlay, checkpoint key equality, exactly-once
guarding, audit contents, missing-checkpoint failure, reference-vLLM command
arguments, production reward fail-closed behavior, and partial SHS construction
failure. This is source-level and CPU module evidence; the real 128-by-4 GPU
shard has not yet been launched and no production-candidate claim is made.

### Real One-GPU 128-by-4 veRL Shard

Run/screen: `shs_grpo_replica_shard_20260712_v2_realverl` /
`qwen_shs_grpo_replica_v2_20260712`. The physical topology was one RTX PRO
6000, one TP=1 vLLM engine, 128 prompts, and group size four, for 512 rollout
responses. The actor and reference were both constructed from the SHS
deployment export. Their audits recorded `preconstructed` mode, one completed
checkpoint overlay with all 29 expected keys, and 12 deterministic SHS block-ID
buffers excluded from FSDP weight synchronization. The actor retained 29
trainable tensors; the reference retained zero.

The first real attempt failed during actor-to-rollout synchronization because
veRL included deterministic block-ID buffers that the vLLM deployment contract
reconstructs rather than loads. Marking exactly the expected 3 projections by 4
buffers non-persistent fixed the sync without removing runtime buffers or any
trainable parameter. The second attempt completed rollout, production reward,
old/reference log probabilities, and reached actor backward, but the incorrectly
single-GPU-expanded PPO micro-batch of eight exhausted 94.83 GiB. The paper
topology uses eight across four replicas, or two per GPU. Setting the one-GPU
micro-batch to two preserved the global 128-prompt optimizer batch through
gradient accumulation and reduced the observed actor-phase allocation to about
47.4 GiB.

The third attempt completed step 1 and checkpointed successfully. Shell wall
was 826 seconds and measured trainer step time was 745.60 seconds. The phase
decomposition was 266.54 seconds generation, 9.84 reward/extraction, 90.67 old
log probabilities, 93.71 reference log probabilities, 284.67 actor update, and
10.40 checkpoint save. It processed 516,253 tokens at 692.40 tokens/s. Mean
response length was 897.60 tokens, 8.79% reached the 3,072-token cap, mean
binary reward was 0.13867, gradient norm was 0.10968, and policy-gradient loss
was `7.87e-7`.

The successful process emitted nine SHS dispatch receipts across actor,
reference, and rollout model instances. Every receipt selected the reference
PyTorch/cuBLAS backend on `cuda:0`; none reported fallback. The saved veRL model
contained every one of the 29 completed SFT trainable keys. All 29 tensors and
72,889,627 elements changed after the optimizer step, with maximum absolute
delta `7.63e-6`; no initial key was missing.

Raw veRL `resume_mode=auto` loaded model, optimizer, scheduler, and RNG from
`global_step_1` and displayed progress 1/1, but nevertheless entered rollout
generation again. That check was intentionally interrupted before a second
optimizer step; no `global_step_2` was written. The tracked launcher now runs a
pre-Ray completion gate that validates the tracker plus model/optimizer/extra
actor files. The actual repeated check returned immediately with
`completed_step=target_steps=1` and `new_optimizer_steps=0`.

This run is **not** labelled production candidate. The full execution plumbing,
checkpoint change, and project-level resume gates passed, but SHS-reference
vLLM still failed the earlier fixed semantic parity panel. An earlier summary
described the shard as containing extensive malformed generations, but the
committed compact evidence does not contain a malformed-output definition,
count, category ledger, or examples. What is established is 71/512 reward-one
responses, 441/512 reward-zero responses, and 45/512 cap hits; reward zero and
cap hit do not by themselves imply malformed output. The response-quality state
therefore remains unclassified pending a trace audit. Four-GPU global-batch,
two-batch canary, and 20-batch pilot gates therefore remain blocked. Compact
metrics are recorded in
`docs/experiment_records/compact_metrics/2026-07-12_shs_grpo_replica_shard_v2_realverl.json`.

### Full SHS vLLM Shadow Evaluation

Run/screen: `shs_vllm_full_eval_shadow_20260712_v1` /
`qwen_shs_vllm_full_eval_shadow_20260712_v1`. This was deliberately executed
as **SHADOW/NEW BACKEND** measurement after expanded semantic parity failed. It
is not a strict-compatible result and does not replace the serial HF scores.

The run used the completed `step_00003916` SHS checkpoint and its existing
hash-bound deployment export. The checkpoint `trainable_state.pt` SHA-256 was
`ebb4cd92f6e890c17cf0e14a883557358dff927e901b69588f1867e6dd016712` and the
export `model.safetensors` SHA-256 was
`6266ca7fdba2f8e117d0560897562468163e9ddf997a4a15ec10944b3d22c583`.
The RTX PRO 6000 environment used PyTorch `2.8.0+cu128`, vLLM `0.10.2`, and
EvalScope `1.8.1`. Every dispatch receipt selected the reference
PyTorch/cuBLAS SHS backend on `cuda:0`, with no fallback. TP remained one,
`model_impl=transformers`, and eager execution remained enabled throughout.

The initial 64-sequence run exposed 43/21 microbatch fragmentation and only
about 51-55% GPU utilization. After 149 MATH rows, it was resumed without
repeating verified identities using a 250 ms gather window,
`max_num_seqs=128`, `max_num_batched_tokens=131072`, and evaluation batch size
128. Prompts, caps, stops, seeds, item/sample identities, sampling parameters,
and grading semantics were unchanged. The retuned incremental panel reached
about 1,080 generated tokens/s, versus roughly 240 whole-run tokens/s before
the first retuned batch landed.

Final verified scores and paired transitions relative to serial HF were:

| Benchmark | HF % | vLLM % | Delta pt | Correct->wrong | Wrong->correct | Extracted-answer agreement |
|---|---:|---:|---:|---:|---:|---:|
| MATH-500 | 59.0000 | 58.2000 | -0.8000 | 29 | 25 | 67.40% |
| GSM8K | 76.9522 | 78.0895 | +1.1372 | 36 | 51 | 86.73% |
| OlympiadBench | 22.9630 | 22.2222 | -0.7407 | 33 | 28 | 39.70% |
| AMC Average@32 | 13.4375 | 8.3594 | -5.0781 | 137 | 72 | 6.41% |
| AMC greedy pass@1 | 27.5000 | 30.0000 | +2.5000 | 2 | 3 | 40.00% |

The four-task average moved from `43.0882%` to `41.7178%`, a `-1.3704` point
delta. Multiple components exceed the fixed 0.5-point boundary, so strict
compatibility failed. AMC Average@32 had 82 versus 194 missing extractions and
56 versus 40 tokenized 3,072-cap hits. Its mean/p50/p90 response lengths moved
from `802.1/629/1556` to `866.9/597/2293` tokens. Olympiad cap hits increased
from 81 to 100; MATH increased from 34 to 36. Per-item AMC correct counts out
of 32 are retained in the compact metrics.

The first complete generation/evaluation attempt took 3,202 seconds
(`53m22s`). Its phase walls were 1,576.35 seconds main, 973.78 seconds AMC
Average@32, and 78.80 seconds AMC greedy. Strict JSONL verification then found
three AMC review identities missing because EvalScope concurrent writers had
interleaved three records. The malformed originals and incomplete receipts
were preserved. Clean-cache resume regenerated `(11,13)`, `(25,2)`, and
`(32,28)` with their original derived seeds; a final single-writer repair for
`(25,2)` produced exact verified counts of 500, 1,319, 675, 1,280, and 40.

Artifact-complete wall, including diagnosis, engine reloads, and repairs, was
4,497.73 seconds (`74m57.7s`). This is a measured `6.537x` speedup over the
8h10m serial HF wall despite missing the 20-35 minute target and exceeding the
60-minute secondary gate. Generation-call wall was 3,057.78 seconds:
1,934.88 main, 1,044.86 AMC Average@32, and 78.04 AMC greedy. Including repair
work, 2,572,071 tokens were generated at 571.9 tokens/s over artifact-complete
wall, only 25.27% of the 2,262.8 tokens/s long-decode anchor. The bottlenecks
were eager reference-SHS execution, initial request fragmentation, long-output
tails, CPU extraction/reporting, repeated cold engine loads, and the JSONL
serialization repair. Compact evidence is recorded in
`docs/experiment_records/compact_metrics/2026-07-12_shs_vllm_full_eval_shadow_20260712_v1.json`.
