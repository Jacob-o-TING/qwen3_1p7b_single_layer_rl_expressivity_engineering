# Qwen3-1.7B Expressivity Engineering

Research code for Qwen3-1.7B-Base single-layer reinforcement learning and
feed-forward-network expressivity experiments. The repository contains GRPO /
veRL-style training infrastructure, SHS, TriGLU and OFT variants, registered
vLLM integration points, reproducible evaluation utilities, and compact
experiment records.

The lightweight local test path does not load Qwen weights or require veRL.
Full training requires a separate GPU environment, Qwen3 weights, veRL, and
the selected datasets. Model weights, datasets, checkpoints, uncurated
generations, private operational logs, and chat transcripts are intentionally
excluded.

## Repository status

This is a research release, not a turnkey claim of paper-exact reproduction.
Historical records preserve negative results, protocol changes, and known
evaluation caveats. The canonical unresolved gates are tracked in
`docs/experiment_plans/PENDING_2026-07-12_mandatory-experiment-gates.md`.

The publication fork was derived from source snapshot
`c464e3a2ef5892c9bf33e9eb06a2f7f6d958e8f5`. Curated tracked audit inputs,
model traces, verdicts, and compact runtime evidence are preserved for
reproducibility. Untracked training artifacts and large generated outputs are
not included.

## What is included

- Paper-aligned configs for Qwen3-1.7B-Base, Layer 10, middle layers 11-15,
  B5/B10-style layer sets, and adapter variants.
- Dependency-light Python package under `src/qwen_single_layer_rl`.
- Dataset prep and decontamination placeholders for NuminaMath-CoT style data.
- Layer freezing utilities for Hugging Face Qwen-style models.
- Adapter / architecture variant registry for OFT-like or custom modules.
- Explicit training modes for adapter-only, backbone-only, joint, and baseline
  runs.
- veRL launch glue for materialized RLHF parquet, custom reward scoring, and
  patched synchronous HF rollout for architecture-variant runs.
- Smoke tests that validate config, registry, layer selection, and dry-run
  artifact layout locally.
- Curated manual-audit packages under `audit_inputs/` and compact runtime
  receipts under `runs/runtime_smokes/`.
- AutoDL / 4x5090 / H800 notes for later controlled runs.

## Reproduction target

Primary target:

- Base model: `Qwen/Qwen3-1.7B-Base`
- Architecture facts: 28 decoder layers, GQA 16 Q heads / 8 KV heads, 32,768
  context length.
- Algorithm: GRPO in veRL.
- Dataset: NuminaMath-CoT, downsampled to 50K problems after decontamination.
- Single-layer protocol: freeze embeddings, LM head, and every decoder layer
  except the selected layer.
- Paper anchors for Qwen3-1.7B: Layer 10 and Layer 12 are high-contribution;
  the heuristic middle-5 selection is layers 11-15.

Paper hyperparameters captured in `configs/base_qwen3_1p7b.yaml`:

| Field | Value |
| --- | --- |
| Learning rate | `5e-6` |
| LR sweep for full baseline | `[1e-6, 3e-6, 5e-6, 1e-5]` |
| Train batch size | `512` |
| PPO mini batch size | `128` |
| PPO micro batch size | `8` |
| Group size | `4` |
| Max response length | `3072` |
| KL coefficient | `0.001` |
| Clip range | `0.2` |
| Epochs | `4` |
| Adaptive LR boosted/base | `1e-5` / `5e-6` |

## Layout

```text
qwen3_1p7b_single_layer_rl_expressivity_engineering/
  configs/
    base_qwen3_1p7b.yaml
    layer10_grpo.yaml
    middle_11_15_grpo.yaml
    b5_b10_hooks.yaml
    smoke_tiny.yaml
    adapters/
      oft_like_layer10.yaml
    training_modes/
      adapter_only_oft_like.yaml
      selected_layer_no_adapter.yaml
      selected_layer_plus_adapter.yaml
      full_backbone_plus_adapter.yaml
      full_backbone_no_adapter_baseline.yaml
    verl/
      qwen3_1p7b_layer10_grpo.yaml
  docs/
    autodl_hardware_notes.md
    autodl_sync_and_remote.md
    extension_points.md
    qwen3_layer10_variant_dimensions.md
    repro_notes.md
  scripts/
    launch_single_node_4gpu.sh
    launch_verl_grpo.sh
    prepare_numina_math.sh
    remote/
    run_smoke.ps1
    run_smoke.sh
  src/qwen_single_layer_rl/
    data/
    eval/
    model_surgery/
    rewards/
    training/
  tests/
```

## Local smoke path

From this folder:

```powershell
python -m pip install -e .
python -m qwen_single_layer_rl.training.dry_run --config configs/smoke_tiny.yaml --out runs/smoke
python -m unittest discover -s tests
```

or:

```powershell
.\scripts\run_smoke.ps1
# If local PowerShell script execution is disabled:
powershell -ExecutionPolicy Bypass -File .\scripts\run_smoke.ps1
```

On Linux:

```bash
bash scripts/run_smoke.sh
```

The smoke run creates:

```text
runs/smoke/
  dry_run_manifest.json
  planned_trainables.txt
  resolved_config.json
```

## Training modes

The framework does not assume the Qwen3-1.7B backbone is always frozen. The
explicit contract is:

```yaml
freeze_policy:
  backbone_train_mode: frozen | selected | full
  train_adapter_modules: false | true
```

Supported configs:

| Mode | Config | Backbone | Adapter / architecture module |
| --- | --- | --- | --- |
| Adapter-only | `configs/training_modes/adapter_only_oft_like.yaml` | frozen | trained |
| Selected backbone only | `configs/training_modes/selected_layer_no_adapter.yaml` | Layer 10 | none |
| Selected backbone plus adapter | `configs/training_modes/selected_layer_plus_adapter.yaml` | Layer 10 | trained |
| Full backbone plus adapter | `configs/training_modes/full_backbone_plus_adapter.yaml` | full | trained |
| No-adapter baseline | `configs/training_modes/full_backbone_no_adapter_baseline.yaml` | full | none |

Use `freeze_embeddings` and `freeze_lm_head` to decide whether embedding and
LM-head parameters are included in full-backbone baselines.

## Full training sketch

Install full dependencies in the GPU environment, then:

```bash
bash scripts/prepare_numina_math.sh \
  --source AI-MO/NuminaMath-CoT \
  --out data/numina_math_cot_50k
```

That command uses Hugging Face `datasets` to stream `AI-MO/NuminaMath-CoT`,
deterministically reservoir-sample 50K train records after normalized exact and
token 8-gram near-duplicate decontamination, and write:

```text
data/numina_math_cot_50k/
  train.jsonl
  train.parquet
  val.jsonl
  val.parquet
  prep_manifest.json
```

For a quick pipeline check without downloading the full dataset:

```bash
bash scripts/prepare_numina_math.sh \
  --source AI-MO/NuminaMath-CoT \
  --source-hub modelscope \
  --out data/numina_math_cot_smoke \
  --target-size 100 \
  --max-source-records 1000
```

If you already downloaded or mirrored the data as JSONL:

```bash
bash scripts/prepare_numina_math.sh \
  --input-jsonl /path/to/numina_math_cot.jsonl \
  --out data/numina_math_cot_50k \
  --benchmark-problems data/decontam/qwen_math_eval_<revision>/benchmark_problems.jsonl
```

Prepare the pinned paper-aligned Qwen math benchmark snapshot and its auditable
decontamination contract first:

```bash
bash scripts/prepare_paper_benchmarks.sh
```

The snapshot pins HuggingFaceH4/MATH-500 revision
`6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be` and QwenLM/Qwen2.5-Math revision
`a45202bd16f1ec06f433442dc1152d0074773465`: MATH500 (500), GSM8K (1319),
text-only English OlympiadBench (675), and AMC23 (40). The generated manifest
records every source URL, revision, SHA-256, normalization version, row count,
and output hash.

Then materialize veRL-format parquet, render the training plan, and launch:

```bash
bash scripts/launch_verl_grpo.sh configs/layer10_grpo.yaml
```

For single-node 4-GPU:

```bash
bash scripts/launch_single_node_4gpu.sh configs/layer10_grpo.yaml
```

If `python` points to the Microsoft Store shim on Windows, activate the intended
environment or use its absolute interpreter path.

## AutoDL sync sketch

Package and sync without storing secrets in files:

```powershell
.\scripts\remote\package_for_autodl.ps1
.\scripts\remote\sync_to_autodl.ps1 -HostName <host> -Port <port> -KeyPath "$env:USERPROFILE\.ssh\id_ed25519_autodl_codex"
```

On the remote machine, prefer absolute Python paths:

```bash
cd /root/autodl-tmp/qwen3_1p7b_single_layer_rl
PYTHON_BIN=/root/miniconda3/bin/python \
bash scripts/remote/autodl_remote_launch.sh configs/training_modes/selected_layer_no_adapter.yaml layer10_no_adapter_pilot
```

See `docs/autodl_sync_and_remote.md` for `screen`, logging, and artifact
pullback conventions.

The remote runner for architecture variants uses a patched veRL `v0.6.1`
checkout with synchronous HF rollout, so generation executes the same model
surgery as actor training. vLLM remains useful for a later plain-baseline speed
run, but it should not be used for SHS/TriGLU/OFT unless the custom
architecture is explicitly supported by the inference engine.

### Ordered production SFT

The production SFT launcher auto-detects the visible GPU count and preserves an
effective packed batch size of 8. With micro-batch size 1, this selects gradient
accumulation 8 on one GPU and 2 on four GPUs:

```bash
cd /root/autodl-tmp/qwen3_1p7b_single_layer_rl
bash scripts/start_sft_ordered_variants.sh
bash scripts/monitor_sft.sh
```

The order is SHS, whole-layer baseline, TriGLU, and OFT. Each variant must have
an exact completed final checkpoint before its paper-pinned evaluation starts.
Evaluation covers MATH-500, GSM8K, OlympiadBench, and sampled AMC Average@32;
the next variant starts only after all four report hashes are stored in
`evaluation_complete.json`. Reusing the same `RUN_STAMP` resumes the latest
checkpoint and skips only evaluations with a valid, untampered receipt.

The custom EvalScope model uses an eight-request static micro-batcher by
default. EvalScope worker threads rendezvous for 10 ms, then one padded HF
`generate()` call handles requests with the same decoding configuration. Every
unlimited final evaluation first runs a bounded 64-token batched preflight.
Override the production width with `SFT_EVAL_BATCH_SIZE` only as a separately
recorded throughput choice.

Training writes one metric record per optimizer step and validates/checkpoints
at 10%, 25%, 50%, 75%, and 100%. The monitor displays those milestones, the
current evaluation phase, rolling first/recent loss means, median step time,
training ETA, GPU/disk state, and recent errors without opening a new training
process.

## Layer-10 Architecture Variants

The ordered whole-layer run set for Qwen3-1.7B layer 10 is:

```text
1. configs/layer10_whole_layer_shs.yaml
2. configs/layer10_whole_layer_baseline.yaml
3. configs/layer10_whole_layer_triglu_side_ffn.yaml
4. configs/layer10_whole_layer_oft.yaml
```

Launch/render in that order:

```bash
bash scripts/launch_layer10_ordered_variants.sh
```

Implementation dimensions and trainability rules are recorded in
`docs/qwen3_layer10_variant_dimensions.md`.

## Extension pattern

Model variants are defined by a registry:

- `identity`: no architecture surgery.
- `oft_like`: placeholder OFT-style trainable transform injection.
- `qwen_swiglu_shs`: SHS HyperGrid modulation over Qwen SwiGLU.
- `qwen_swiglu_triglu_side`: residual-delta TriGLU side FFN over Qwen SwiGLU.
- `qwen_swiglu_oft`: OFT input rotations on Qwen SwiGLU projections.
- Add your own by subclassing `ArchitectureVariant` and registering it in
  `model_surgery/registry.py`.

The training flow should call:

1. Load config.
2. Load model.
3. Apply `architecture_variant`.
4. Apply `freeze_policy`.
5. Hand the model and rendered config to veRL.

This order keeps layer freezing and model surgery explicit and auditable.

## References

- Qwen3-1.7B-Base model card: https://huggingface.co/Qwen/Qwen3-1.7B-Base
- Qwen veRL docs: https://qwen.readthedocs.io/en/latest/training/verl.html
- Single-layer RL paper: https://arxiv.org/abs/2607.01232
