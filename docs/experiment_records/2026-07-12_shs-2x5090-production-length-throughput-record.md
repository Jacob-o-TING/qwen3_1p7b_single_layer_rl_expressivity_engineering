# 2026-07-12 SHS 2xRTX 5090 Production-Length Throughput Record

Date: 2026-07-12

Run ID: `shs_2x5090_production_length_throughput_20260712_v1`

Screen: `qwen_shs_2x5090_longdecode_20260712_v1`

Final status: **throughput gate passed; current engine configuration rejected
for an economical production GRPO launch**. No actor update, GRPO quality wave,
checkpoint mutation, dependency change, or shutdown occurred.

## Identity And Topology

The run used two independent TP=1 vLLM V1 replicas on one host with two RTX
5090 GPUs. Each GPU had 32,607 MiB and used the reference PyTorch/cuBLAS SHS
projection inside vLLM. The grouped Triton path was excluded because it had
already failed the unchanged 0.9999 full-model parity gate.

The hash-bound inputs were:

- base source commit: `623bbdfe3d8ff7fbec344db923a99d327b150e70`;
- SFT trainable state: `ebb4cd92f6e890c17cf0e14a883557358dff927e901b69588f1867e6dd016712`;
- exported `model.safetensors`: `6266ca7fdba2f8e117d0560897562468163e9ddf997a4a15ec10944b3d22c583`;
- export `config.json`: `6fb2be47e145d65f40c30f4dcd113a43c79d7359decbbb56f4cf683bc2021c7a`;
- final deployed runner: `59ebe9739e556c293a6b637680018387eee7f3e77d0a9a89babaa6405a50c7df`;
- prompt manifest: `512f2e2c2720aa48dc7678d597e5954c19a7a53e05257f7b622f3d066820e278`.

Thirty-two real prompts were selected without replacement from the production
50k veRL training parquet using seed 20260712. Each GPU received 16 disjoint
prompts. Group size four produced pressure cells of 16, 32, and 64 requests per
GPU from 4, 8, and 16 unique prompts respectively. Structured outputs were
written atomically under independent GPU/cell/pressure paths and then merged by
the parent process; no concurrent JSONL writer was used.

## Workloads And Startup

Both replicas loaded once and ran two cells at each staged pressure:

1. `matched_800_1024`: temperature 0.8, top-p 0.95, minimum 800, cap 1024;
2. `production_cap3072`: temperature 1.0, top-p 1.0, no minimum, cap 3072.

The engine used `max_model_len=4096`, `max_num_seqs=64`,
`max_num_batched_tokens=131072`, memory utilization 0.80, chunked prefill,
Paged KV cache, eager execution, and no prefix cache. Engine load took 19.26
and 20.79 seconds. The per-GPU cold eight-token probes took 0.355 seconds and
the warm probes took 0.237-0.240 seconds.

Each worker emitted exactly three `reference` SHS dispatch receipts. No
fallback, OOM, CUDA error, or model/checkpoint mismatch occurred.

## Throughput Results

Pair throughput is total tokens divided by the wall from the earlier worker
start to the later worker finish. It therefore includes real straggler idle
time. The per-GPU mean is retained separately and must not replace the pair
metric for rental projections.

| Cell | Pressure/GPU | Pair tok/s | Mean tok/s/GPU | Mean length | P50 | P90 | Cap hit |
|---|---:|---:|---:|---:|---:|---:|---:|
| matched 800-1024 | 16 | 704.7 | 353.9 | 1001.5 | 1024 | 1024 | 87.50% |
| production cap3072 | 16 | 246.0 | 124.8 | 740.4 | 455 | 2090 | 6.25% |
| matched 800-1024 | 32 | 868.7 | 444.3 | 1004.4 | 1024 | 1024 | 79.69% |
| production cap3072 | 32 | 463.5 | 236.7 | 701.8 | 461 | 1401 | 6.25% |
| matched 800-1024 | 64 | 959.0 | 494.3 | 998.5 | 1024 | 1024 | 80.47% |
| production cap3072 | 64 | **446.8** | 259.9 | **870.5** | **467** | **3038** | **10.16%** |

The final production pair started only 1.55 seconds apart. The faster rank
finished after 182.87 seconds and the slower rank after 247.86 seconds, giving
a 73.15% interval overlap. This is a valid concurrent measurement: the
non-overlap is measured tail imbalance, not sequential execution.

The production distribution matched the earlier real C2 anchor closely. Mean
length was 870.5 versus 897.6 (-3.0%), and the cap-hit rate was 10.16% versus
8.79%. The 128-response cell therefore passes the preregistered +/-25% length
boundary and is materially more representative than the earlier cap-16 smoke.

The matched pressure-64 cell reached only 494.3 tokens/s/GPU, or 21.84% of the
RTX PRO 6000 C1 reference-SHS anchor of 2,262.8 tokens/s/GPU. This is not a
clean GPU compute ratio. The 5090 engine profile reserved approximately 20.6
GiB for peak activations under `max_num_batched_tokens=131072`, leaving only
1.22 GiB / 11,440 tokens for KV cache and reporting 2.79x full-4096-token
concurrency. Increasing pressure 16 to 64 consequently raised matched
throughput by only 40%. The current configuration is scheduler/KV constrained.

The external one-second sampler observed steady-cell peaks of 9,451 MiB for
matched decode and 7,537/7,383 MiB for the final production cell. These do not
include the short engine-profile activation peak reported internally by vLLM.

vLLM 0.10.2's synchronous V1 `LLM.generate` returned null request timestamps,
as in the earlier C1 run. Per-request latency p50/p90 and scheduler-time p50 are
therefore recorded as unavailable rather than approximated from batch wall.
Configured sequence occupancy was 25%, 50%, and 100% at pressure 16, 32, and
64. The worker logs retain vLLM scheduler/KV receipts. FlashInfer sampling was
unavailable and vLLM used its PyTorch-native top-p/top-k sampler.

## Recovery Audit

Two orchestration defects were encountered and preserved as separate attempts:

1. attempt 1 failed before GPU load because the parent opened `gpu0/worker.log`
   before creating its directory; both GPUs remained at zero MiB;
2. attempt 2 completed one rank's pressure-64 production cell, then the parent
   incorrectly treated that normally exited fast rank as a pair failure and
   terminated the still-running slow rank.

The fixes added directory creation, failed-manifest archival, per-rank stage
ownership, atomic resume skips, and a tested asymmetric-rank wait contract.
The resumed run skipped every completed cell and filled only the missing rank.
A final concurrent pressure-64 production repeat was then required because
cells completed in separate attempts cannot establish pair throughput. The
non-concurrent receipts remain under `attempts/nonconcurrent_p64_*`. Six focused
CPU tests now cover grouping/seeds, atomic completion, decision gates,
asymmetric rank exit, and concurrent-start validity.

## Economic Decision

The measured production pair rate is 446.75 tokens/s. Scaling four such pairs
ideally to eight RTX 5090s gives 1,787.0 tokens/s. Applying that rate to the
recorded four-replica C2 token work of 2,065,012 tokens gives 1,155.6 seconds,
or **19.26 minutes of pure rollout per global batch**.

| Budget | Current-config pure rollout projection on 8x5090 |
|---:|---:|
| 20 batches | 6.42 h |
| 50 batches | 16.05 h |
| 98 batches | 31.46 h |
| 196 batches | 62.91 h |
| 391 batches | 125.51 h |

These values exclude reward, old/reference log probabilities, actor
forward/backward, weight synchronization, checkpoints, orchestration, and final
evaluation. They supersede the optimistic short-cap 5090 extrapolation for this
exact engine configuration, but they are not a final 5090 hardware verdict.

The next justified gate is a separately named 5090-balanced engine profile,
starting with a smaller `max_num_batched_tokens` such as 32,768 and an explicit
KV-cache allocation target. It must rerun the matched and production
pressure-64 cells before any GRPO launch. No production GRPO is authorized by
this record.

## Artifacts

The compact local archive contains 49 files and no model/checkpoint tensors:

`eval_artifacts/shs_2x5090_production_length_throughput_20260712_v1/`

Archive SHA-256:

`071e33ddbe9f83c108e62300d41d3cc7377684692f798334a62194c1579dfae5`

The remote host was idle at closeout with no screen, no GPU process, and zero
MiB used on both GPUs. It was not shut down.
