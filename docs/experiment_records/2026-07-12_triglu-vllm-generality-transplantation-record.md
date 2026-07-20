# 2026-07-12 TriGLU vLLM Generality Transplantation Record

Date: 2026-07-12

Status: **PASS for bounded TriGLU vLLM onboarding and matched throughput.**
This record does not authorize 50/98-batch GRPO or the deferred Eval Parity
Matrix.

## Fixed Identity And Scope

- Methodology: `custom_ffn_vllm_onboarding_methodology_20260712_v1`.
- Run: `triglu_vllm_onboarding_smoke_20260712_v1`.
- Screen: `qwen_triglu_vllm_smoke_20260712_v1`.
- Hardware: one host with two independent TP=1 RTX 5090 replicas.
- Runtime: PyTorch 2.8.0+cu128, vLLM 0.10.2 V1 enforce-eager,
  Transformers 4.57.1, Triton 3.4.0.
- Base: Qwen3-1.7B-Base.
- TriGLU checkpoint SHA-256:
  `8a463168b4dce0f698357a821dfaca2d7b7fa90032841adb717251e323c48ab8`.
- TriGLU: Layer 10, `side_dim=512`, `side_hidden=2048`,
  `side_scale=0.1`, exact-zero initial side return.
- Scheduler: `max_num_batched_tokens=32768`,
  `gpu_memory_utilization=0.85`, max 64 sequences, max model length 2,048,
  chunked prefill enabled, prefix caching disabled.
- Excluded: Eval Parity Matrix, production GRPO, actor synchronization,
  unrelated Triton work, and shutdown.

## Implementation Result

The implementation now provides:

- an architecture-neutral export metadata and dispatch-receipt helper;
- explicit `Qwen3TriGLUConfig`, base-model, and causal-LM classes;
- a unique `qwen3_triglu` model type and `Qwen3TriGLUForCausalLM`
  architecture;
- an out-of-tree `vllm.general_plugins` entry point;
- a registered custom wrapper inheriting vLLM `TransformersForCausalLM`;
- post-weight-load FP32/device finalization for the TriGLU side branch;
- one semantic dispatch receipt proving Layer 10, dimensions, backend, device,
  dtype, and no fallback;
- resumable cell execution that reused the passed HF export and vanilla
  control across repairs;
- focused no-op, export-key, dispatch, forward/backward, plugin, and generation-
  interface regressions.

The production inference backend is ordinary PyTorch/cuBLAS inside vLLM. No
TriGLU Triton kernel was added.

## Failure-Driven Revisions

Four failed attempt artifacts were preserved before the final pass:

1. Forcing `model_impl=transformers` selected generic
   `TransformersForCausalLM`; vLLM converted side linears to BF16 and the FP32
   side input failed dtype matching.
2. A post-load finalizer did not help because the generic wrapper still
   bypassed the registered custom class.
3. Switching to `model_impl=auto` reached the registry, but the custom wrapper
   inherited hidden-state-only `TransformersModel` and was rejected for the
   generation runner.
4. Inheriting `TransformersForCausalLM` made the custom math execute, but the
   semantic receipt assumed `nn.Linear.out_features`; converted
   `ReplicatedLinear` exposes the weight tensor rather than that convenience
   attribute.

The final path uses `model_impl=auto`, resolves
`Qwen3TriGLUForCausalLM`, reapplies FP32 after `load_weights`, and derives
receipt dimensions from backend-neutral weight shapes.

The retrospective also exposed that historical SHS runs resolved generic
`TransformersForCausalLM`, so the registered SHS wrapper had not actually been
exercised. This is reported as a separate SHS issue only. Per owner direction,
no SHS source, test contract, or runtime was changed and no SHS GPU benchmark
was run in this wave.

## HF And vLLM Parity

The runtime-surgery TriGLU model and explicit HF export had:

- zero missing, unexpected, mismatched, or loading-error keys;
- FP32 side parameters after reload;
- last-token logits max-absolute difference `0.0` and relative L2 `0.0`;
- cosine `0.9999942183` from the finite-precision cosine calculation despite
  elementwise equality;
- equal top-1 and equal eight-token greedy continuation.

The explicit HF export and vLLM custom path produced the same eight greedy
token IDs at pressures 1, 8, and 16. Direct vLLM full-logit extraction was not
part of this bounded harness, so the honest claim is **exact greedy token
parity**, not direct HF-versus-vLLM logit parity.

The vLLM receipt proved:

- variant `qwen_swiglu_triglu_side`;
- backend `reference_pytorch_cublas`;
- Layer 10, `side_dim=512`, `side_hidden=2048`;
- side parameters on `cuda:0` in FP32;
- `fallback=false`.

Engine profile completed with 20.40 GiB KV cache and 93.23x maximum full-2,048-
token concurrency. TriGLU engine load was 15.83 seconds; vanilla load was 15.07
seconds. Parent-process PyTorch peak-memory counters do not observe vLLM's
engine subprocess and are therefore treated as unavailable rather than zero.

## Matched Long-Decode Throughput

Both cells used the same 64 prompt identities, seeds, scheduler configuration,
temperature 0.8, top-p 0.95, minimum 800 tokens, and maximum 1,024 tokens.

| Pressure | TriGLU tok/s/GPU | Vanilla tok/s/GPU | TriGLU mean length | Vanilla mean length |
|---:|---:|---:|---:|---:|
| 16 | 809.9 | 1,690.4 | 1,002.3 | 995.8 |
| 32 | 1,480.6 | 3,097.4 | 971.3 | 980.1 |
| 64 | **2,872.1** | **5,442.1** | 972.1 | 982.7 |

At pressure 64, TriGLU reached **52.8% of vanilla**. It was **151.8% of the
historical SHS 5090 matched anchor** of 1,892.1 tok/s/GPU. The final TriGLU and
reused vanilla cells were measured sequentially on the same host/profile after
resumption, not as a simultaneous pair. They are valid per-GPU operational
comparisons without cross-GPU contention.

Sampled token traces differed across architectures. Lengths were bounded and
close, so the table is throughput evidence; it is not semantic parity or a
pure operator-level hardware comparison.

## Variant Decision

TriGLU did not meet the provisional 70-80% of vanilla threshold for being
called near-baseline speed. It is nevertheless 51.8% faster than the current
SHS reference path, so **baseline plus TriGLU remain the lower-cost first pair
for one/two/20-batch RL canaries** when the research goal is to compare a naive
whole-layer update with one added-expressivity architecture. This result alone
does not authorize a 50- or 98-batch run.

## Reusable Versus Architecture-Owned Work

Architecture-independent pieces now include export metadata, explicit HF
construction order, plugin registration, generation-interface inspection,
post-load finalization timing, semantic receipts, pressure/long-decode harness
shape, failure classification, and resumable cell reuse.

TriGLU-specific pieces remain its six side projections, multiplicative
combination, exact-zero `up` initialization, FP32 side policy, target-layer
count, dimensions, checkpoint-key prefixes, and receipt identity.

## Remaining Gates

- direct HF-versus-vLLM logit/ranking comparison if strict logit parity is
  required for policy use;
- separately approved investigation of the registered SHS wrapper; this
  TriGLU wave deliberately leaves SHS unchanged;
- one/two-batch actor-versus-vLLM log-probability, KL, ratio, and clipping
  receipts;
- multi-GPU weight synchronization, receiver hashes, and fresh-reload oracle;
- real-length memory and full-component timing on the selected 8-GPU topology;
- 20-batch trend/go-no-go review;
- the separately deferred Eval Parity Matrix and parallel evaluator.

Production authorization remains **false** for 50/98-batch GRPO.

## Evidence And Git

- Local compact run:
  `runs/runtime_smokes/triglu_vllm_onboarding_smoke_20260712_v1/`.
- Remote compact archive:
  `/root/autodl-tmp/triglu_vllm_onboarding_smoke_20260712_v1_compact.tar.gz`.
- Archive SHA-256:
  `f4a523f76b7607628894d76398f5cde80c217e924efeadc67b2a59ffe7742267`.
- Deployment export remains remote and uncommitted; compact archive size is
  52 KiB.
- TriGLU implementation commits: `4068415`, `23b1871`, `e4d324e`, `6d044c9`,
  `be085ea`, and `aa7b4a2`. The temporary SHS edit in `1b5451d` was explicitly
  reverted at the owner's direction.
- Remote ended with no screen/process and both GPUs at 0 MiB/0%; it was not
  shut down.
