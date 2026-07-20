# Custom Architecture vLLM Onboarding

Date established: 2026-07-12

Status: durable runtime-engineering workflow. Experiment-specific measurements
and decisions remain in `docs/experiment_records/` and
`docs/experiment_plans/`.

## Purpose

Onboard a custom causal language-model architecture to vLLM without modifying
the installed vLLM source tree, while retaining Paged KV cache, continuous
batching, chunked prefill, deterministic decoding, and explicit backend
receipts.

Continuous batching itself is not registered per architecture. vLLM provides
the scheduler and cache manager. Registration maps the architecture string in
an exported model config to code that vLLM can instantiate, load, and execute
over flattened dynamic token batches.

## Preferred Path For Local MLP Or Adapter Changes

Architectures that preserve the base model's causal attention and KV-cache
semantics should first use vLLM's Transformers backend:

1. Define an explicit Hugging Face `PretrainedConfig` subclass with a unique
   `model_type` and serialized architecture parameters.
2. Define explicit base-model and causal-LM classes. Construct the custom
   modules during `__init__`; do not depend on unrecorded runtime surgery.
3. Export a self-describing `config.json` containing `model_type`,
   `architectures`, `auto_map`, dimensions, target layers, seeds/maps, and
   variant metadata.
4. Preserve a stable checkpoint-key contract. Record missing, unexpected,
   duplicated, and intentionally reconstructible keys.
5. Register the Hugging Face classes with `AutoConfig`, `AutoModel`, and
   `AutoModelForCausalLM`.
6. Add a Python package entry point under `vllm.general_plugins`.
7. In the plugin callback, map the exported architecture string through
   `vllm.ModelRegistry.register_model` to a lazy runtime implementation.
8. Initially subclass vLLM's `TransformersModel` so the native supported base
   attention path supplies PagedAttention and KV-cache integration.
9. In runtime finalization, move persistent custom buffers to the resolved
   device, rebind modules replaced by vLLM, select the requested custom or
   reference backend, and emit dispatch/fallback receipts.
10. Install the project into the exact pinned vLLM environment so Python entry-
    point discovery runs at engine startup.

The SHS implementation is the reference example:

- `src/qwen_single_layer_rl/vllm/shs_hf_model.py`
- `src/qwen_single_layer_rl/vllm/shs_plugin.py`
- `src/qwen_single_layer_rl/vllm/shs_vllm_model.py`
- `[project.entry-points."vllm.general_plugins"]` in `pyproject.toml`

The first generalization is TriGLU:

- `src/qwen_single_layer_rl/vllm/custom_ffn_contract.py`
- `src/qwen_single_layer_rl/vllm/triglu_hf_model.py`
- `src/qwen_single_layer_rl/vllm/triglu_plugin.py`
- `src/qwen_single_layer_rl/vllm/triglu_vllm_model.py`

The shared code owns lifecycle metadata and semantic dispatch receipts. It does
not own SHS HyperGrid or TriGLU side-FFN mathematics.

## Fixed Custom-FFN Lifecycle Contract

Every local-FFN architecture follows this order:

1. **Declare identity.** Allocate a unique `model_type`, architecture string,
   variant-config key, target-layer list, dimensions, precision policy, and
   initialization contract.
2. **Construct explicitly.** Replace the target module during HF model
   `__init__`. Runtime-only surgery is permitted for training experiments but
   is not a deployment format.
3. **Audit state ownership.** Base weights have one registered owner. Mark
   deterministic reconstructible buffers separately from synchronized learned
   tensors. Reject missing, unexpected, mismatched, and duplicate keys.
4. **Export self-describing metadata.** Write `config.json`, local remote-code
   shims, tokenizer files, and architecture parameters without mutating the
   canonical checkpoint.
5. **Reload an oracle.** Compare the runtime-surgery model with the explicit HF
   export at logits, top-1, greedy tokens, seeded samples, dtypes, and devices.
6. **Register lazily.** Install an out-of-tree `vllm.general_plugins` entry
   point and map the architecture to a `TransformersModel` wrapper.
7. **Finalize runtime state.** Move architecture-owned state to the resolved
   device, preserve its precision policy, and reject a missing or duplicated
   custom module.
8. **Prove dispatch.** An opt-in semantic receipt names the variant, actual
   backend, fallback status, target layer, dimensions, devices, and dtypes.
9. **Exercise the scheduler.** Pass single-request greedy parity, heterogeneous
   pressure, long decode, cap/stop, memory, and no-OOM gates before making a
   throughput claim.
10. **Separate claims.** Different sampled traces are valid operational
    throughput evidence only when lengths are bounded/matched. They are not
    semantic parity or hardware-isolation evidence.
11. **Protect training.** Run a reference forward/backward or trainable-state
    regression after adding deployment hooks. Receipt code must be inert when
    its environment variable is absent.
12. **Defer production authorization.** Actor synchronization, on-policy
    log-probability receipts, resume, and full evaluation are separate gates.

## SHS Failure Ledger And Reusable Invariants

| Observed failure | Diagnosis | Durable invariant |
|---|---|---|
| Safetensors export found ambiguous projection keys | Base linears were registered through both the MLP and SHS delta modules | Every tensor has one registered owner; auxiliary references are non-registered and rebound explicitly |
| SHS generator/adapters silently became BF16 | Generic loading inherited the base-model dtype | Serialize and assert per-component precision; TriGLU side modules remain FP32 because their forward consumes `x.float()` |
| TriGLU vLLM profile saw FP32 input and BF16 side weights | vLLM replaced/loaded Linear modules after wrapper `__init__`, undoing early dtype finalization | Reapply architecture-owned device/dtype finalization after `TransformersModel.load_weights`, before profile or generation |
| The post-load finalizer still never ran | `model_impl=transformers` forced generic `TransformersForCausalLM` and bypassed the registered architecture wrapper | Use `model_impl=auto` for registered custom architectures; verify the resolved architecture in engine logs and never equate a module-level receipt with plugin-wrapper dispatch |
| Registered wrapper was rejected for `runner generate` | It inherited hidden-state-only `TransformersModel`, which lacks the generation protocol's `compute_logits` | Generative custom wrappers inherit `TransformersForCausalLM`; verify `ModelRegistry` reports `is_text_generation_model=True` before engine launch |
| First custom-wrapper profile completed math but receipt formatting crashed | vLLM converted `nn.Linear` to `ReplicatedLinear`, which has weights but no `out_features` attribute | Semantic receipts use backend-neutral tensor shapes and declared metadata, never framework-specific convenience attributes |
| AutoModel or remote-code load failed | The export lacked complete config/model shims and a unique architecture identity | Export `model_type`, `architectures`, `auto_map`, config class, base model class, and causal-LM class together |
| Deterministic maps remained on CPU | Only one custom backend migrated buffers | Device finalization is architecture-wide and happens before local state capture |
| Weight sync attempted deterministic block-ID buffers | Learned tensors and reproducible runtime state shared one synchronization ledger | Synchronize learned parameters only; reconstruct deterministic buffers and verify their hashes separately |
| Short cap-16 smoke overpredicted production speed | It did not exercise long decode, KV pressure, or response-length tails | Use pressure 64 and matched 800-1,024-token responses before runtime projection |
| RTX 5090 throughput collapsed with `max_num_batched_tokens=131072` | Activation profiling left too little KV cache | Profile activation/KV allocation together; the accepted 5090 starting point is 32,768 batched tokens and 0.85 memory utilization |
| Fast Triton passed local tolerance but missed the fixed full-logit cosine gate | Small Layer-10 drift amplified downstream | Production remains reference PyTorch/cuBLAS until unchanged end-to-end gates pass |
| Seeded vLLM traces changed with pressure | Scheduler and native top-p fallback changed sampled trajectories | Greedy parity is the strict token gate; sampled throughput comparisons disclose trace/length drift |
| Concurrent evaluation writers corrupted resumability | Multiple ranks wrote one stream without ownership | One writer per shard, stable identities, atomic completion, deterministic merge, and duplicate/gap rejection |

The failure ledger is a checklist, not a promise that every future architecture
needs the same patch. New failures are first classified as architecture math,
lifecycle-contract defects, or backend limitations.

## Architecture Decision Tree

### MLP, adapter, normalization, or residual changes only

Reuse the Transformers backend and the base model's supported attention path.
Replace only the changed modules in the explicit Hugging Face model. This is
the lowest-risk route for SHS, TriGLU, OFT, and related Layer-10 variants.

### Causal attention changes with conventional KV semantics

Start with the Transformers backend if its attention bridge supports the
operator. Otherwise implement a native out-of-tree vLLM model that uses vLLM
attention modules and exposes the expected forward, load-weights, and cache
contracts.

### New recurrent state, nonstandard cache, dynamic routing, or multimodality

Registration alone is insufficient. Implement the corresponding state/cache
manager or multimodal processor/input mapper. Explicitly test request
preemption, resume, heterogeneous lengths, and state ownership.

## Execution Contract

A continuously batched model must:

- accept flattened dynamic token batches and explicit positions;
- avoid assumptions that every invocation is a rectangular
  `[batch, sequence]` tensor;
- use vLLM-managed KV cache instead of a private generation loop;
- tolerate changing token counts across prefill and decode scheduler steps;
- maintain correct stop, EOS, cap, and seeded-sampling behavior;
- keep persistent maps and buffers on the correct rank/device;
- load every required weight exactly once and reject unknown required keys;
- expose the actual backend used, including any fallback, per projection or
  custom operator;
- keep scientific decoding and item/sample identities independent of request
  completion order.

## Required Validation Ladder

1. HF reference initial no-op and checkpoint-key audit.
2. HF export/reload forward, logit, greedy, and seeded-sampling parity.
3. vLLM enforce-eager single-request parity with explicit dispatch receipts.
4. Heterogeneous prompt and response lengths with Paged KV and chunked prefill.
5. Pressure sweep at 1/8/16/32/64 concurrent requests.
6. Long-response panel matching the production token-length distribution.
7. Reference-versus-custom backend timing at identical generated-token counts.
8. Compile/CUDA-graph testing only after eager correctness.
9. Actor-to-rollout weight synchronization and post-sync parity.
10. Interruption, partial-output, checkpoint, and deterministic-resume tests.

Never infer production speed from an isolated operator or short cap-16 smoke.
The production selection is the fastest end-to-end path that passes the fixed
numerical and semantic gates; it may legitimately retain cuBLAS/reference
projections inside vLLM continuous batching.

## Evaluation Reuse

The same runtime can serve evaluation after the training path becomes
production ready. Use one TP=1 replica per GPU, shard stable benchmark
item/sample IDs across replicas, continuously batch requests locally, and merge
by identity rather than completion order. Preserve serial evaluator parity,
partial-result visibility, idempotent resume, and complete extraction/grading
receipts.

Every model summary must report both paper-primary AMC Average@32 and the
separate AMC greedy pass@1 diagnostic when available. Greedy results are never
included in the four-benchmark arithmetic mean unless a future preregistered
protocol explicitly changes that statistic.
