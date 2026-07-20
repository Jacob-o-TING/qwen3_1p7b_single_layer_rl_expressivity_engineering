# 2026-07-12 SHS 2xRTX 5090 KV-Balanced Throughput Record

Date: 2026-07-12

Run ID: `shs_2x5090_kv_balanced_throughput_20260712_v1`

Status: **degraded, hypothesis confirmed**. Reducing the oversized prefill
token budget released activation memory to KV cache and substantially improved
continuous-batching decode throughput. The staged profile missed the existing
production-length acceptance boundary, so it is not relabelled as a pass and
does not authorize production GRPO.

## Immutable Control And Preregistration

The earlier `shs_2x5090_production_length_throughput_20260712_v1` run remained
immutable. Its manifest SHA-256 stayed
`929b3d56790438255d410c695f0a0459d1cbd1fb1bccaab04a78441ef3b23db8`.
It is the activation-heavy control with:

- `max_model_len=4096`;
- `max_num_seqs=64`;
- `max_num_batched_tokens=131072`;
- `gpu_memory_utilization=0.80`.

The approved primary profile used `max_num_batched_tokens=32768` and memory
utilization 0.85. A 16384 profile was preregistered but could run only if either
GPU reported fewer than 32,768 KV tokens, less than 8x full-4096 concurrency,
or an OOM. Missing engine-profile receipts were a fail-stop condition. No
explicit KV-cache allocation, 0.88/0.95 utilization, checkpoint mutation,
actor update, GRPO, or dependency change was allowed.

All profiles retained two independent TP=1 RTX 5090 replicas, reference
PyTorch/cuBLAS SHS inside vLLM V1, the same real Numina prompts, group size
four, staged pressure 16/32/64, and the same matched and production decode
cells. The prompt manifest SHA remained
`512f2e2c2720aa48dc7678d597e5954c19a7a53e05257f7b622f3d066820e278`.

## Engine Profile

Both primary replicas reported identical memory profiles:

| Metric | Control 131072/.80 | Balanced 32768/.85 | Change |
|---|---:|---:|---:|
| Peak activation | 20.60 GiB | 5.16 GiB | -75.0% |
| Available KV cache | 1.22 GiB | 18.24 GiB | 14.95x |
| KV cache tokens | 11,440 | 170,720 | 14.92x |
| Full-4096 concurrency | 2.79x | 41.68x | 14.94x |

The 32768 profile therefore exceeded both conditional thresholds by a wide
margin. The decision was `accept_primary`; the 16384 profile was not launched.

Engine load took 16.84-16.85 seconds. Cold eight-token probes took
0.346-0.349 seconds, and warm probes took 0.238-0.242 seconds. Each GPU emitted
exactly three `reference` dispatch receipts. There was no fallback, OOM, CUDA
error, checkpoint mismatch, screen leak, or running process at closeout.

## Throughput

Matched 800-1024 cells provide the cleanest throughput comparison because
their generated-length ranges are controlled. Production cells are retained as
the actual operational stochastic distribution under each scheduler config.

| Cell | Pressure/GPU | Control pair tok/s | Balanced pair tok/s | Balanced/control |
|---|---:|---:|---:|---:|
| matched 800-1024 | 16 | 704.7 | 1003.9 | 1.42x |
| production cap3072 | 16 | 246.0 | 253.0 | 1.03x |
| matched 800-1024 | 32 | 868.7 | 1902.7 | 2.19x |
| production cap3072 | 32 | 463.5 | 470.2 | 1.01x |
| matched 800-1024 | 64 | 959.0 | **3742.1** | **3.90x** |
| production cap3072 | 64 | 446.8 | **899.1** | **2.01x** |

Balanced matched pressure-64 throughput averaged 1,892.1 tokens/s/GPU, which
is 83.62% of the RTX PRO 6000 C1 anchor of 2,262.8 tokens/s/GPU. The earlier
21.84% ratio was therefore predominantly a memory/scheduler-configuration
artifact rather than the RTX 5090 compute ceiling.

Matched pair throughput scaled 3.73x from pressure 16 to 64, and production
pair throughput scaled 3.55x. The ideal pressure ratio is 4x. The immutable
control's matched scaling over the same range was only 1.36x. This directly
supports the stated hypothesis: releasing activation budget allowed many more
long sequences to remain active and restored continuous-batching scaling.

The balanced pressure-64 production cell generated 85,599 tokens across 128
responses at 899.07 pair tokens/s. Mean length was 668.74, p50 was 391, p90 was
1,710, and 8/128 responses hit the 3,072 cap. Pair start skew was 0.55 seconds,
and execution overlap was 97.59%. Steady sampled VRAM peaked at 23,873 and
24,701 MiB, leaving bounded device headroom.

## Sampling Stability And Degraded Status

The production profile missed the existing `897.6 +/-25%` mean-length boundary:
the lower boundary was 673.2, while the observed mean was 668.74. The staged
runner therefore correctly returned `degraded`; no tolerance was changed.

The control and balanced runs used the same prompt manifest and all per-request
seeds matched. Nevertheless, PyTorch-native top-p sampling was not
trajectory-stable after scheduler pressure changed:

| Pressure | Requests | Equal seeds | Equal token traces | Equal lengths | Control tokens | Balanced tokens |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 32 | 32 | 32 | 32 | 23,692 | 23,692 |
| 32 | 64 | 64 | 10 | 14 | 44,913 | 45,524 |
| 64 | 128 | 128 | 34 | 41 | 111,428 | 85,599 |

At pressure 64 the mean absolute per-request length delta was 607.2 tokens and
the maximum was 2,856. FlashInfer was unavailable, so vLLM used the
PyTorch-native top-p/top-k sampler. The matched cell remains a valid controlled
throughput comparison, but the production 2.01x result must be labelled an
operational-distribution comparison rather than paired semantic evidence.

vLLM 0.10.2 synchronous V1 again returned null request timestamps. Request
latency p50/p90 and scheduler-time p50 remain unavailable rather than being
approximated from batch wall. A separately named async instrumentation gate is
needed if per-request latency becomes decision-critical.

## Economic Projection

Scaling the measured production pair rate by four gives an ideal eight-GPU
aggregate of 3,596.3 tokens/s. Applying it to the recorded four-replica C2
token work of 2,065,012 tokens gives 574.2 seconds, or **9.57 minutes of pure
rollout per global batch**.

| Budget | Balanced current-distribution pure rollout on 8x5090 |
|---:|---:|
| 20 batches | 3.19 h |
| 50 batches | 7.98 h |
| 98 batches | 15.63 h |
| 196 batches | 31.26 h |
| 391 batches | 62.36 h |

This is about 2.01x faster than the activation-heavy production control and
roughly halves its 125.5-hour pure-rollout projection. It excludes reward,
old/reference log probabilities, actor forward/backward, synchronization,
checkpointing, orchestration, and evaluation. Because the observed production
length distribution changed, the table is an operational estimate, not a
fixed-trajectory hardware comparison or production authorization.

## Verification And Artifacts

Ten focused local tests and four balanced remote tests passed. They cover
engine-profile parsing, conditional triggering, fail-stop behavior, sampling
stability, grouping/seeds, atomic completion, decision gates, asymmetric-rank
exit, and concurrent-start validity.

The compact evidence archive contains 43 files and no model or checkpoint
tensors. Local root:

`eval_artifacts/shs_2x5090_kv_balanced_throughput_20260712_v1/`

Archive SHA-256:

`b4ac5da78eb41658de403dfbdaa53bd2ae752cbe0cedf81dbda4f16eb79ae902`

The remote host was idle at closeout with both GPUs at zero MiB and no screen.
It was not shut down. Production GRPO remains unauthorized.

## Owner Decision: End Throughput Testing And Enter Production Staging

After reviewing the bounded result, the owner accepted it as sufficient for the
intended concept-level estimate and ended the standalone throughput program.
The `degraded` status, changed sampled trajectories, unavailable synchronous
request timestamps, and wide timing uncertainty remain part of the immutable
record, but they do not require another performance-only gate before a staged
production canary.

The selected starting profile is `max_num_batched_tokens=32768` with
`gpu_memory_utilization=0.85`. The preregistered conditional 16,384 profile will
not be run. No separate queue-depth benchmark, FlashInfer benchmark, or async
latency-instrumentation run is required before production. The 131,072/.80
control and this 32,768/.85 result remain distinct historical artifacts.

For rental planning, retain the measured range instead of choosing a false
point estimate: matched sustained capability gives an optimistic approximately
15 pure-rollout hours for 391 batches on eight RTX 5090s, while the observed
finite stochastic workload gives a conservative 62.36 pure-rollout hours.
Reward, old/reference log probabilities, actor update, synchronization,
checkpointing, and evaluation remain additional. Production should replace
this broad bracket with live component timing after its first completed
batches, rather than delay launch for more synthetic throughput refinement.

This owner decision retires throughput uncertainty as a launch blocker; it does
not erase numerical, actor-versus-rollout, weight-sync receipt, cleanup, OOM,
or resume requirements. Production still uses a separately approved run name,
starts with a bounded canary, preserves sparse exact checkpoints and readable
logs, and stops only at the documented economic or correctness gates.
