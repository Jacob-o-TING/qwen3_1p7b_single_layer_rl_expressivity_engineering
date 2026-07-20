# 2026-07-12 RTX 5090 Pair Bring-Up And Weight-Sync Record / 中英混搭版

<!-- bilingual-companion-v1 -->

> **English original / 英文原件：** [2026-07-12_rtx5090-pair-bringup-and-weight-sync-record.md](../2026-07-12_rtx5090-pair-bringup-and-weight-sync-record.md)
> **中文导读 / Bilingual guide：** 这份 companion 的主题是 **2026-07-12 RTX 5090 Pair Bring-Up And Weight-Sync Record**。下面完整保留 original report 的 metrics、tables、commands、paths、hashes 与 scientific claims，同时加入中文 narrative layer 和 bilingual section handles，方便项目负责人阅读与后续 Agent 检索。
> **Evidence contract / 证据契约：** English original 保持 immutable；如果 companion 与原件出现 wording tension，以原件中的 exact numbers、machine-readable artifacts 和 source paths 为准，companion 不得 retroactively 改写实验事实。


Date: 2026-07-12

Run IDs:

- `rtx5090_pair_bringup_20260712_v1`
- `shs_2x5090_actor_rollout_weight_sync_20260712_v1`

Status: environment and bounded runtime bring-up passed. The complete Gate C
claim did not pass because only 23 of 29 trainable tensors changed after
normalizing the initial FP32 SHS tensors to the actor's BF16 runtime dtype, and
the run did not capture direct-sync versus fresh-reload full-logit parity or
per-version receiver timing. No production GRPO wave is authorized.

## Topology

The endpoint exposed one host with two NVIDIA GeForce RTX 5090 GPUs, not two
independent instances. Each GPU reported 32,607 MiB, compute capability 12.0,
and driver 580.76.05. `nvidia-smi topo -m` reported `NODE` connectivity within
one NUMA node and no NVLink. The host had 754 GiB RAM and 213 GiB free on the
250 GiB data volume at audit time.

The preferred production schedule remains colocated and phase-separated: both
GPUs are TP=1 rollout replicas during generation and both are FSDP actor ranks
during the update. This run does not support a default static one-actor plus
one-rollout split, and an eight-GPU system should not be assumed to mean a
static four-plus-four split without an overlap and staleness profile.

## Gate A: Environment And Pair Transport

Gate A passed after installing only the two missing pinned packages,
EvalScope 1.8.1 and math-verify 0.9.0. `pip check` reported no broken
requirements. The selected project-local runtime was:

- PyTorch 2.8.0+cu128;
- vLLM 0.10.2;
- Triton 3.4.0;
- Transformers 4.57.1;
- EvalScope 1.8.1;
- math-verify 0.9.0;
- veRL 0.6.1.

Both GPUs exposed `sm_120`. BF16 matmul, loss, backward, AdamW, and CUDA
synchronization passed on each GPU. The completed SHS checkpoint hash matched
`ebb4cd92f6e890c17cf0e14a883557358dff927e901b69588f1867e6dd016712`.

The two-rank NCCL test reduced the expected value on both ranks. A 64 MiB
all-reduce had a median wall of approximately 2.33 ms on each rank after three
warmups. This is a same-host systems receipt, not a cross-node network claim.

## Gate B: Architecture And Runtime Smoke

The reference training integration passed after the launcher supplied
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, required by deterministic cuBLAS. The first
measured optimizer step took 0.940 seconds; the two steady steps took 0.258 and
0.244 seconds. Checkpoint reload and the resumed second update matched exactly.
Peak allocated memory was 10.28 GB.

The fresh-cache isolated Triton projection matrix passed FP32 and BF16
correctness for gate, up, down, and odd-tail shapes at token batches 1, 8, 16,
and 32. Required BF16 pressure-1/8 speedups over the sliced reference were:

| Projection | Tokens 1 | Tokens 8 |
|---|---:|---:|
| gate | 28.83x | 15.47x |
| up | 29.15x | 15.18x |
| down | 15.68x | 13.13x |

These are isolated operator results, not end-to-end speedups.

The unchanged full-model Triton gate failed as expected. All three projections
dispatched Triton and greedy tokens matched, but full-logit cosine was
0.9997553, below the preregistered 0.9999 boundary. Relative L2 was 0.0046466.
The production backend therefore remains reference PyTorch/cuBLAS inside the
vLLM Transformers integration.

The initial V1 TP=1 vLLM smoke passed pressure 1, 8, and 16 with HF greedy
token parity and three Triton dispatch receipts. Peak observed memory was
23,142 MiB. The first fresh-cache request took 43.07 seconds, while warm
pressure 8 and 16 reached 147.4 and 293.7 generated tokens/s. The cold request
must not be extrapolated to a full wave.

## Gate C: Repeated Live Sync And Restart

The real veRL path ran ten continuous two-GPU steps, followed by a process
restart from `global_step_10` and one additional step. The first process wrote
a complete step-10 checkpoint and exited zero after 152 seconds shell wall.
The training loop itself completed in 75 seconds. Early steps took 12.03 and
8.87 seconds; warm short-response steps stabilized near 4.57-4.78 seconds.

The restart loaded both ranks' model, optimizer, RNG, and scheduler states,
then executed and saved step 11. Its update had nonzero gradient norm and the
tracker advanced exactly from 10 to 11. The first resume attempt deliberately
recorded a failure after changing dataloader workers from 8 to 0: torchdata
cannot load a multi-process StatefulDataLoader state into a single-process
iterator. Restoring the checkpoint-compatible worker count to 8 repaired the
resume without repeating the first ten steps.

Peak actor memory reached 29.73 GB allocated and 31.96 GB reserved per reported
rank. This leaves little headroom on a 32,607 MiB card. Larger response caps or
microbatches require a separate pressure gate rather than extrapolation.

Both successful processes emitted a DataLoader-worker-killed traceback during
Ray teardown after final checkpoint and metrics had completed. The shell exit,
checkpoint tracker, rank files, and independent resume gate all passed. Treat
this as a reproducible cleanup defect, not data corruption, but do not call the
closeout clean until the Ray/torchdata shutdown path is fixed.

## Tensor And Reload Audit

The step-11 FSDP merge initially exposed a packaging defect: veRL saved
`shs_hf_model.py` with a relative import that was not self-contained. The new
metadata repair utility preserves the original config, writes hash receipts,
switches to thin project-backed AutoClass wrappers, and does not modify tensor
files. The repaired checkpoint merged into a disposable 4.11 GB HF export.

The initial trainable contract contained exactly 29 tensors. No deterministic
SHS block-ID buffer leaked into the merged export. Raw hashes differed for 27
tensors, but this included 18 FP32-to-BF16 dtype transitions. After casting the
initial tensors to the updated runtime dtype, only 23 of 29 tensors visibly
changed. The unchanged tensors were:

- `model.layers.10.input_layernorm.weight`;
- `model.layers.10.self_attn.q_norm.weight`;
- the gate, up, and down `add_scale` tensors;
- the up-projection `mul_scale` tensor.

The owner's interpretation is adopted for the scientific record: this result
does **not** indicate that the SHS hypernetwork failed to learn. The learned
HyperGrid generator, low-rank additive factors, and the remaining observable
SHS tensors changed. Gate and down multiplicative scales also changed. The
unchanged SHS set consists only of the three global additive amplitude scalars
and the up multiplicative amplitude scalar; the other two unchanged tensors are
native Qwen RMSNorm scales. A tiny real-GRPO update not moving these six values
across a BF16 representable boundary is unsurprising and is not an architecture
failure. `No observable BF16 change` must not be rewritten as `no gradient` or
`no FP32 optimizer-state change` without those additional receipts.

This fails the preregistered 29-of-29 observable-update gate. It is consistent
with tiny real-GRPO gradients not crossing the BF16 ULP for every tensor; it is
not evidence that the completed 23-tensor updates were stale. A separately
labelled synthetic observable-update test is still required and must not be
mixed with quality evidence.

The repaired step-11 export passed a fresh V1 TP=1 vLLM reload at pressure 1,
8, and 16. HF and vLLM greedy tokens matched for the fixed prompt, and all
three Triton dispatch receipts were present. Warm pressure 16 reached 297.1
generated tokens/s with 21,742 MiB observed memory. Full direct-sync versus
fresh-reload logits and receiver-side parameter hashes were not captured, so
Gate C remains partial even though veRL's actual direct `update_weights` path,
ten cycles, and restart/resync all executed.

## Throughput Interpretation And Eight-GPU Projection

The 293.7-297.1 generated tokens/s values are **aggregate per-GPU rates across
all 16 concurrent requests**, not a per-request rate. The implementation sums
the generated-token counts from every returned request and divides that sum by
the wall time of the single batched `generate` call. Dividing by 16 gives a
descriptive average near 18.6 tokens/s/request, but continuous batching does not
allocate a fixed independent throughput lane to each request.

This metric used response caps 4, 8, 12, and 16. It is startup-heavy and is not
a production-length 800-1,000-token rollout measurement. Eight TP=1 replicas
would provide a literal aggregate of approximately 2,377 generated tokens/s at
the measured short-response rate. For the full global batch of 512 prompts,
group size four, and 800-1,000 response tokens, blindly applying that rate
would imply 11.5-14.4 minutes of pure rollout per batch and 74.9-93.6 pure-
rollout hours over 391 batches. That number is deliberately rejected as a
final estimate because the workload does not match the measurement.

The better provisional end-to-end anchor remains the production-shaped C2
component record: 745.60 seconds per global batch in the four-replica planning
topology. Ideal fixed-global-batch scaling from four to eight colocated,
phase-separated ranks gives 372.8 seconds per batch and 40.5 hours for the
complete four-epoch, 391-batch wave. Allowing for PCIe/NCCL communication,
stragglers, checkpointing, weight synchronization, and non-ideal actor/log-
probability scaling gives a current **45-60 hour end-to-end RL planning range**
on eight RTX 5090s. Final evaluation is excluded from that number.

This range is conditional, not an authorization. The short Gate C actor already
reserved 31.96 GB on a 32,607 MiB GPU. A production-length 5090 gate must prove
that 3,072-token responses fit and measure steady long-response throughput
before the estimate can be narrowed. If the current authoritative serial-HF
evaluation is retained, its roughly 8h10m SHS wall is additional; a sharded or
vLLM evaluator needs its own semantic and aggregation gate before claiming a
shorter final-evaluation wall.

## Artifacts And Decision

Compact records were pulled locally as 48 files. Archive SHA-256:

`e855573565bcd6bcc9b704706f94286fc679517c7d3469b2a04caf557d7988b6`

Local root:

`eval_artifacts/rtx5090_pair_bringup_20260712_v1`

The remote machine was idle at closeout: no screen, no GPU process, and zero
MiB reported GPU memory. The owner may shut it down after this record and its
source commits are pushed. Large FSDP checkpoints and disposable exports remain
remote-only and are reproducible from the archived SFT checkpoint plus source.

Next requirements before production GRPO are:

1. add receiver-side version, timing, and parameter-hash receipts around
   veRL/vLLM `update_weights`;
2. run a separate deterministic all-29 observable-update gate;
3. compare direct-sync and fresh-reload fixed logits, rankings, and tokens;
4. fix or explicitly suppress the post-checkpoint Ray/torchdata teardown race;
5. re-run a production-length 5090 pressure gate because this test used a
   response cap of 16 and reached nearly all reserved VRAM during actor work.
