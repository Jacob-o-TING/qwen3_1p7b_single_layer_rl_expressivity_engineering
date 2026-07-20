# 2026-07-12 Baseline And TriGLU 2x5090 GRPO Prelaunch Gate Record / 中英混搭版

<!-- bilingual-companion-v1 -->

> **English original / 英文原件：** [2026-07-12_baseline-triglu-2x5090-grpo-prelaunch-gate-record.md](../2026-07-12_baseline-triglu-2x5090-grpo-prelaunch-gate-record.md)
> **中文导读 / Bilingual guide：** 这份 companion 的主题是 **2026-07-12 Baseline And TriGLU 2x5090 GRPO Prelaunch Gate Record**。下面完整保留 original report 的 metrics、tables、commands、paths、hashes 与 scientific claims，同时加入中文 narrative layer 和 bilingual section handles，方便项目负责人阅读与后续 Agent 检索。
> **Evidence contract / 证据契约：** English original 保持 immutable；如果 companion 与原件出现 wording tension，以原件中的 exact numbers、machine-readable artifacts 和 source paths 为准，companion 不得 retroactively 改写实验事实。


Status: **PASS.** Both variants are ready for a later eight-GPU
production-shaped canary. This record does not authorize 20, 50, or 98 global
batches.

## 固定身份与范围 / Fixed Identity And Scope
- Run: `baseline_triglu_2x5090_grpo_prelaunch_gate_20260712_v1`
- Screen: `qwen_baseline_triglu_2x5090_prelaunch_20260712_v1`
- Hardware: one host with two RTX 5090 GPUs, one actor GPU and one TP=1 vLLM
  rollout GPU.
- Initialization: the same untuned Qwen3-1.7B-Base revision for both variants.
- Baseline: naive Layer-10 whole-layer training.
- TriGLU: the same backbone plus Layer-10 `side_dim=512`,
  `side_hidden=2048`, exact-no-op initialization.
- Rollout contract: eight fixed prompts, group size four, temperature 1.0,
  top-p 1.0, 128-token bounded response cap, two optimizer updates.
- Reward/KL/clip: production `verl_math_verify`, `KL=0.001`, `clip=0.2`.

The base `config.json` SHA-256 was
`1bb33a92c3548fbc68b889b490e810440435253598835bd71dff0396060c12db`.
The full data-selection ledger SHA-256 remained
`1097fdde429daf60eca6cbfb9b4e9f2f49ca5386133bd6618155087069a3968d`.
The bounded prompt/group ledger SHA-256 was
`e687e18037f654b5c16cfa243f0d572753d4a7a6c3380a0778d9fbeb1f748ff2`.

## Result Summary

| Receipt | Baseline | TriGLU |
|---|---:|---:|
| Status | PASS | PASS |
| Rollout-vs-actor logprob mean absolute delta | 0.019896 | 0.018569 |
| Rollout-vs-actor logprob max absolute delta | 0.311789 | 0.476549 |
| Finite logprob tokens | 3,999 / 3,999 | 4,029 / 4,029 |
| Post-update approximate KL | 0.000904 | 0.000888 |
| Post-update clip fraction | 0.009752 | 0.010176 |
| Trainable tensors | 11 | 23 |
| TriGLU side trainable tensors | 0 | 12 |
| Exact-resume max parameter delta | 0 | 0 |
| Resume repeat / skip | 0 / 0 | 0 / 0 |
| Peak actor allocated GiB | 20.33 | 21.36 |
| Peak actor reserved GiB | 30.37 | 26.16 |
| Bounded generation seconds | 2.61 | 4.46 |

The actor and frozen reference had exactly equal initial token logprobs for
both variants. The first four prompt groups happened to contain no within-group
reward variation, so the first update correctly had zero gradient. The second
update had gradient norms 0.293894 for baseline and 0.267013 for TriGLU. The
gate therefore requires at least one effective update across the two bounded
steps, rather than incorrectly requiring the first step to update.

## Reward Sanity

Both variants produced 32 responses under the intentionally short 128-token
cap. Each had one reward-one response and 31 reward-zero responses. Baseline
classified its responses as 28 token-cap, two valid-but-wrong, one
extraction/format failure, and one correct. TriGLU classified 28 token-cap,
three valid-but-wrong, and one correct.

All 64 rewards were binary, used `verl_math_verify`, and had zero reward-wiring
failures. This rules out broken verifier plumbing in the bounded path. The high
cap-hit rate is expected under this diagnostic cap and is not a quality result
or a production response-length recommendation.

## TriGLU Live Weight Synchronization

The final actor contract contained all 23 selected Layer-10 tensors: 11
backbone tensors and 12 TriGLU side tensors. vLLM loaded all 23 without an
engine rebuild. Sender and receiver version were both two, all receiver hashes
matched the sender, and 21 of 23 receiver tensors changed observably after the
bounded update. The live-sync wall time was 0.783 seconds; worker load time was
0.0286 seconds.

The receiver reported one CUDA TriGLU wrapper, FP32 side parameters, the
`reference_pytorch_cublas` backend, and no fallback. A fresh vLLM engine loaded
from the updated export produced exactly the same 32 greedy tokens and exactly
the same sampled-token logprobs as the live-synchronized engine.

## Checkpoint And Exact Resume

Each variant saved after step one, restored trainable weights, optimizer,
scheduler, all RNG state, and the exact prompt/group cursor, then replayed only
step two. Resumed and uninterrupted step-two parameters were bit-exact. The
eight prompt IDs had complete coverage with zero repeats and zero skips.
Optimizer state contained 11 entries for baseline and 23 for TriGLU.

## Failure-Driven Repairs

Three bounded attempts were retained as evidence:

1. Variant injection created new side modules on CPU after the backbone was
   moved to CUDA. The harness now rebinds the full model to CUDA before the
   exact-no-op parity pass.
2. vLLM 0.10.2 rejected a trusted same-host callable RPC until
   `VLLM_ALLOW_INSECURE_SERIALIZATION=1` was explicitly selected. The opt-in is
   limited to this local worker control plane.
3. A green systems run exposed that `.triglu_side.` had been classified as an
   adapter while `train_adapter_modules=false`. That run was invalidated. The
   final config trains the side branch together with Layer 10 and fails fast if
   TriGLU side trainables are absent.

## Decision And Remaining Boundary

Baseline and TriGLU pass the two-GPU prelaunch gate and may proceed to the
separately approved eight-GPU production-shaped canary. The eight-rank
topology, global 512-prompt batch, and distributed checkpoint receipts remain
outside this run. Production 98-batch GRPO remains unauthorized.

The mandatory pending registry remains unchanged and deferred here:

- PENDING-01 Eval Parity Matrix
- PENDING-02 Pure-BF16 SHS and TriGLU paths
- PENDING-03 Registered SHS CausalLM route

## 证据与 Git / Evidence And Git
- Compact archive:
  `runs/runtime_smokes/baseline_triglu_2x5090_grpo_prelaunch_gate_20260712_v1/baseline_triglu_2x5090_grpo_prelaunch_gate_20260712_v1_compact.tar.gz`
- Archive SHA-256:
  `f5b3148c4e8389d2f352c80a9d5e54cc66e80fb733890a51c9a4f10aa3c79810`
- Initial implementation: `eb74ed2`
- Device-placement repair: `921da22`
- Receipt/RPC repair: `ec27c3d`
- Side-trainable contract: `903f35c`

At closeout there was no screen or related process, both GPUs reported zero
MiB and zero utilization, and the instance was not shut down.
