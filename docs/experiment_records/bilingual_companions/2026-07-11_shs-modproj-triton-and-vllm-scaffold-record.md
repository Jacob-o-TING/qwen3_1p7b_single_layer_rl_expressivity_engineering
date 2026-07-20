# 2026-07-11 SHS Modulated Projection Triton And vLLM Scaffold Record / 中英混搭版

<!-- bilingual-companion-v1 -->

> **English original / 英文原件：** [2026-07-11_shs-modproj-triton-and-vllm-scaffold-record.md](../2026-07-11_shs-modproj-triton-and-vllm-scaffold-record.md)
> **中文导读 / Bilingual guide：** 这份 companion 的主题是 **2026-07-11 SHS Modulated Projection Triton And vLLM Scaffold Record**。下面完整保留 original report 的 metrics、tables、commands、paths、hashes 与 scientific claims，同时加入中文 narrative layer 和 bilingual section handles，方便项目负责人阅读与后续 Agent 检索。
> **Evidence contract / 证据契约：** English original 保持 immutable；如果 companion 与原件出现 wording tension，以原件中的 exact numbers、machine-readable artifacts 和 source paths 为准，companion 不得 retroactively 改写实验事实。


Date: 2026-07-11

Run ID: `shs_modproj_triton_decode_bench_20260711_v1`

Status: multiplicative projection implementation, parity matrix, controlled
microbenchmark, module opt-in integration, and vLLM static scaffold complete.
No end-to-end vLLM generation or GRPO claim is made.

## Environment And Scope

- Host: AutoDL kernel-development clone only, SSH port 36348.
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB.
- Driver: 580.82.09.
- Python: 3.12.3.
- PyTorch: 2.8.0+cu128; CUDA 12.8.
- Triton: 3.4.0.
- vLLM: 0.10.2; Transformers: 4.57.1.
- Screen: `qwen_shs_kernel_bench_20260711_v1`.
- Remote result root:
  `runs/kernel_benchmarks/shs_modproj_triton_decode_bench_20260711_v1`.
- Random seed: 20260711 for generated benchmark tensors. SHS maps use explicit
  deterministic seed offsets; no DataLoader, dropout, augmentation, resume RNG,
  or stochastic model execution participates in this operator benchmark.

The implemented equation is:

```text
y[m,n] = sum_k x[m,k] * W[n,k]
         * (1 + mul_scale * tanh(grid[m,row_id[n],col_id[k]]))
```

`src/qwen_single_layer_rl/kernels/shs_modulated_projection.py` retains a
PyTorch reference and implements a Triton inference path with FP32 accumulation.
The kernel removes the base projection plus 32 sliced multiplicative GEMMs and
their `base_col` materializations. It does not couple or optimize `add_grid`,
`add_left`, or `add_right`. It has no backward implementation.

`ShuffledHyperGridDeltaLinear.set_inference_mul_backend("triton")` is explicit
opt-in and is active only in eval mode under `torch.no_grad()`. Training and the
default inference path remain the original reference algebra. Explicit Triton
failure raises; the benchmark manifest records `triton`, `reference`, or
`failed` for every case.

## Correctness Gates

The controlled run passed 32/32 Triton parity cases:

- shapes: gate 2048->6144, up 2048->6144, down 6144->2048, and odd-tail
  2053->2111;
- token batches: 1, 8, 16, and 32;
- dtypes: FP32 and BF16;
- FP32 tolerance: rtol 2e-4, atol 2e-4;
- BF16 tolerance: rtol 3e-2, atol 8e-2.

Maximum observed FP32 absolute error was `5.960464477539062e-07`; maximum
observed BF16 absolute error was `0.00390625`. The reference path was bit-exact
to `F.linear` for both zero grids with nonzero scale and nonzero grids with zero
scale. The production default remains the exact-noop reference path.

The post-integration remote test set passed 10/10 tests. It verifies explicit
module opt-in, complete output parity including the untouched additive path,
separate Mul/Add row and column maps, deterministic map persistence, exact
initial no-op, no silent explicit-Triton fallback, odd tails, and HF checkpoint
key layout.

## Controlled Warm Timings

All values are median CUDA-event milliseconds over 15 repeats after five
warmups. Cold compilation was separate: first-shape FP32 compile plus execution
was 502.34 ms; first BF16 compile plus execution was 217.80 ms. Subsequent
already-specialized invocations were approximately 0.20-0.30 ms cold-call wall
time in the correctness sequence.

| Projection | Tokens | Reference ms | Triton ms | Speedup |
|---|---:|---:|---:|---:|
| gate | 1 | 2.336 | 0.111 | 21.01x |
| gate | 8 | 2.342 | 0.216 | 10.82x |
| gate | 16 | 2.337 | 0.360 | 6.48x |
| gate | 32 | 2.316 | 0.640 | 3.62x |
| up | 1 | 2.259 | 0.111 | 20.36x |
| up | 8 | 2.306 | 0.216 | 10.66x |
| up | 16 | 2.303 | 0.358 | 6.43x |
| up | 32 | 2.616 | 0.641 | 4.08x |
| down | 1 | 2.713 | 0.230 | 11.77x |
| down | 8 | 2.744 | 0.266 | 10.30x |
| down | 16 | 2.927 | 0.409 | 7.16x |
| down | 32 | 2.910 | 0.711 | 4.09x |

Peak allocated memory across timing cases was 36,770,816 bytes for the
reference and 34,341,376 bytes for Triton. The authoritative screen rerun took eight
shell-wall seconds after the earlier direct smoke populated part of the Triton
cache.

`torch.compile` was meaningful only as a diagnostic on the PyTorch reference.
The up/M=8/BF16 case compiled in 347.32 ms and measured 2.524 ms warm, slower
than eager reference at 2.306 ms and Triton at 0.216 ms. Dynamo reported a graph
break at validation-side `Tensor.item()`; this is not a full-graph result.

The first direct smoke failed honestly with
`AttributeError: module 'triton.language' has no attribute 'libdevice'`.
Triton 3.4 requires `tl.extra.cuda.libdevice.tanh`; after that compatibility
fix, the repeated smoke passed with odd-tail FP32 max error 1.79e-7 and a
gate/M=1/BF16 speedup of 7.44x over two timing repeats.

## Decode Economics

At M=1, the three multiplicative projections fall from approximately 7.31 ms
total in the isolated 32-loop reference to 0.45 ms in Triton, a provisional
6.86 ms operator saving per Layer-10 MLP invocation. At M=32, the corresponding
isolated saving is approximately 5.85 ms. These numbers indicate that replacing
the sliced multiplicative path is economically material.

They are not an end-to-end decode estimate. The grid generator, additive
low-rank 32-loop path, activation, attention, KV cache, scheduler, sampling,
other 27 layers, host overhead, and vLLM continuous batching are excluded. No
rental or GRPO budget should be extrapolated until a cached decode step and a
fixed-token vLLM generation panel are measured.

## vLLM 0.10.2 Audit And Scaffold

Pinned source inspection established these requirements:

- `ModelRegistry.register_model` supports lazy `module:class` strings;
- `vllm.general_plugins` entry points are discovered by the installed runtime;
- the Transformers backend creates `AutoModel.from_config` on meta, replaces
  linear modules, creates vLLM attention instances, executes flattened tokens
  through a temporary batch dimension, and loads with `AutoWeightsLoader`;
- TP=1 avoids requiring a custom tensor-parallel plan, but does not remove
  custom config/model, buffer placement, attention, or weight-loading gates.

The scaffold adds `Qwen3SHSConfig`, `Qwen3SHSForCausalLM`, metadata-only export
config construction, and a lazy plugin registration targeting vLLM's generic
`TransformersModel`. Editable installation exposed the entry point and registry
registration resolved to the expected lazy target. A tiny explicit HF model
preserved the existing runtime-surgery checkpoint key prefixes, including
`model.layers.10.mlp.base_mlp.*` and `model.layers.10.mlp.shs.*`.

Full onboarding was deliberately not claimed. A real exported SHS checkpoint
directory was not available in this task. Before engine load, the scaffold must
also ensure persistent shuffle buffers created by the custom HF model move off
CPU/meta correctly under vLLM initialization. Then validate exact loaded/missing
key sets, flattened token counts, paged attention, eager generation, compile,
CUDA graphs, chunked prefill, continuous batching, and update-boundary weight
synchronization.

Recommended next command after creating a disposable exported model directory:

```bash
VLLM_USE_V1=1 vllm serve <exported-shs-model-dir> \
  --model-impl transformers \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --enforce-eager \
  --max-num-seqs 1 \
  --max-num-batched-tokens 256
```

Run one deterministic prompt and inspect loaded/missing/unexpected keys before
removing `--enforce-eager` or raising batching pressure. Only after eager
parity should CUDA graphs and continuous batching be benchmarked.

## Artifacts And Decision

Compact `benchmark.log` and `manifest.json` were pulled locally under the same
run root. The run directory remains ignored and is not committed. Source,
tests, scripts, this record, and the plan update are commit candidates; no
weights, datasets, checkpoints, caches, or large artifacts are included.

Decision: retain the inference-only multiplicative kernel and proceed to a
cached Layer-10 MLP/decode benchmark after the exported-model buffer and weight
loading gates. The separate additive path is now the obvious remaining SHS MLP
bottleneck. Do not start a full GRPO wave from this microbenchmark alone.

## Completion Boundary And Continuing Objective

This record is **Milestone 1 complete**, not architecture-runtime completion.
The kernel-development instance must remain available for the following gated
work. A fast isolated multiplicative projection does not establish that the
kernel participates in full-model generation, vLLM continuous batching, SFT,
or GRPO.

Completed now:

- inference-only SHS base-plus-multiplicative Triton projection;
- explicit opt-in integration with no silent explicit-Triton fallback;
- full module forward parity while the additive path remains reference;
- operator microbenchmarks and compact artifact archival;
- vLLM plugin/config/model and checkpoint-key static scaffold.

Not completed now:

- export and engine-load of a real SHS checkpoint directory;
- full-model logits and generation parity with Triton enabled;
- vLLM flattened-token execution, PagedAttention, CUDA graphs, or continuous
  batching;
- additive-path Triton optimization;
- any Triton backward implementation or training speedup;
- SFT optimizer-step parity with a kernel-enabled forward path;
- rollout-only GRPO plumbing smoke or complete one-batch GRPO update;
- actor-to-rollout custom-weight synchronization;
- end-to-end quality, throughput, or rental-cost validation.

The next system result may be called "SHS runtime built" only after a real
exported checkpoint loads in vLLM, matched full-model generation passes, and a
continuous-batched rollout smoke records that the Triton backend actually ran.
The broader RL path may be called built only after one complete GRPO batch also
passes reward, log-probability, backward/update, weight synchronization,
checkpoint, and resume gates.

### SFT Smoke Meaning

The current inference-only kernel intentionally rejects autograd. Therefore an
ordinary SFT smoke must continue to use the reference path and can prove only
that integration did not regress training. A genuinely kernel-enabled SFT smoke
requires one of these explicitly labelled contracts:

1. Triton forward plus a correct reference-recompute backward, with complete
   gradient parity but no backward speed claim; or
2. Triton forward and Triton/custom backward, with gradient parity and measured
   training performance.

Do not label a reference-backend optimizer step as "kerneled SFT". Gradient
parity must cover inputs, base weights, grid-generator outputs and parameters,
SHS scales, `add_left`, and `add_right` before an optimized SFT claim.

### GRPO Smoke Meaning

Use two successive gates:

1. a rollout-only smoke that loads the exported SHS checkpoint, executes the
   Triton inference path under the selected runtime, generates all four samples
   for a tiny prompt group, and records backend dispatch, tokens, lengths,
   rewards, memory, and wall time;
2. one complete GRPO global-batch smoke that additionally computes old/reference
   log probabilities, advantages/loss, actor backward and optimizer update,
   synchronizes updated custom weights to rollout replicas, checkpoints state,
   and verifies deterministic resume.

Neither gate authorizes a 20-, 50-, or full-budget RL wave until its numerical
and economic report is reviewed.
