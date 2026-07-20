# Rollout Concurrency Notes

This note captures the side-conversation decision context for the first
Qwen3-1.7B single-layer RL reproduction runs.

## Current Scaffold State

- Runtime defaults to `nproc_per_node: 4`, `backend: fsdp`, and
  `inference_backend: hf` for the first architecture-variant wave.
- GRPO uses `group_size: 4`; the veRL template maps this to `rollout.n: 4`.
- vLLM-specific concurrency knobs such as `tensor_parallel_size`,
  `max_num_seqs`, `gpu_memory_utilization`, and `max_num_batched_tokens` are
  kept in config for a later plain-baseline speed run.

## Recommended Concurrency Interpretation

For a plain Qwen3-1.7B baseline using vLLM, prefer replica/data parallel
rollout rather than tensor parallel rollout:

```text
tensor_parallel_size = 1
one full model replica per GPU
vLLM continuous batching inside each GPU replica
```

On a 4-GPU node, this means four rollout replicas, not one model split across
four GPUs. Start with `max_num_seqs` around `16` or `32` per GPU, then tune
upward only if memory headroom and throughput improve.

For SHS, TriGLU, and OFT, use patched synchronous HF rollout first. That path
is slower than vLLM but ensures rollout generation uses the same modified model
graph as training.

## Hardware Cost Context

Quoted rental prices:

| Hardware | Price |
|---|---:|
| 4x RTX 5090 | RMB 11.13 / hour |
| 4x RTX PRO 6000 | RMB 23.91 / hour |
| 2x H800 | RMB 17.77 / hour |

Using the previous rough 4x5090 wall-time estimate of `3-8h`, approximate
costs are:

| Hardware | Rough wall time | Rough cost |
|---|---:|---:|
| 4x RTX 5090 | `3.0-8.0h` | RMB `33-89` |
| 4x RTX PRO 6000 | `2.2-7.6h` | RMB `53-182` |
| 2x H800 | `2.4-9.4h` | RMB `43-167` |

These early estimates assumed an optimized continuous-batching rollout path.
They are not the current budget anchor for SHS/TriGLU/OFT correctness runs,
which must initially use the slower patched synchronous Hugging Face path. See
`docs/experiment_records/2026-07-11_sft-loss-eval-misalignment-and-budgeted-rl-record.md`
for the provisional 64-hour full-run anchor and staged 20/50/98-batch timing.

The RTX PRO 6000 and H800 options may improve stability and memory headroom,
but they need large speedups to beat 4x5090 on cost. For this 1.7B run, 4x5090
is likely the cost-efficient baseline unless 32GB VRAM limits rollout
concurrency.

## Follow-Up

Before launching later vLLM speed runs, make the intended rollout topology
explicit in the config or veRL mapping layer:

- `tensor_parallel_size: 1`
- rollout replicas equal to available GPUs for 1.7B
- initial `max_num_seqs: 16` or `32`
- explicit `gpu_memory_utilization`
- optional `max_num_batched_tokens`
