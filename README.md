# Qwen3-1.7B Expressivity Engineering

Expressivity Engineering (EE) is a design perspective for studying neural
computation whose effective transform depends more richly on its current input
than ordinary linear channel mixing. This repository tests that perspective by
adding an exact-no-op, multiplicative TriGLU side network to one feed-forward
layer of Qwen3-1.7B-Base and comparing it with a matched whole-layer GRPO
baseline.

The experiment uses the layer-localization and single-layer reinforcement
learning workflow of *Is One Layer Enough?* as practical guide rails: Layer 10,
the freezing boundary, GRPO data flow, and the Math/OOD evaluation structure.
The EE hypothesis and TriGLU intervention are separate contributions. This
bounded setup makes a higher-capacity architecture test possible without
pretraining a new model from scratch.

**Follow-up, 2026-07-31; updated 2026-08-27:** [Expressivity Engineering
Follow-up](docs/followups/2026-07-31_expressivity-engineering-follow-up.md)
records a separate, post-report retrospective and general research agenda. It
connects the empirically selected TriGLU bottleneck to LatentMoE, then develops
additive and multiplicative expert composition, structured multi-LoRA SHS
HyperNetworks, full-rank dynamic SwiGLU components, corrected complexity
accounting, attention-replacement directions including selective state-space
models, frequency-shaped EE schedules as alternatives to tied recurrence, and
phase-locked Full/Linear-Attention-to-FFN expressivity schedules. The dated
2026-08-27 extension adds asymmetric exact-no-op initialization, cascaded
PolyNorm correction gates, SwiGLU-internal gate placement, progressive
Best-Layer-to-B5/B10 EE curricula, and downstream-evaluation requirements that
do not treat training reward or pretraining loss as final quality selectors.
It also proposes a gated positive-polynomial PolyNorm family in which every
gate multiplies a polynomial with a positive constant term: a hard ReLU member
with an exactly zero negative half-axis, plus smooth Swish/SiLU members that
trade the strict threshold for improved gradient flow while preserving a
first-order response.

The repository contains GRPO / veRL-style training infrastructure, SHS,
TriGLU and OFT variants, registered vLLM integration points, reproducible
evaluation utilities, and compact experiment records.

The lightweight local test path does not load Qwen weights or require veRL.
Full training requires a separate GPU environment, Qwen3 weights, veRL, and
the selected datasets. Model weights, datasets, checkpoints, uncurated
generations, private operational logs, and chat transcripts are intentionally
excluded.

## Technical report

The canonical technical report is maintained under `docs/technical_report/`,
with the publication-ready PDF at
`docs/technical_report/expressivity_engineering_qwen3_1p7b.pdf`.

- [Technical report (PDF)](docs/technical_report/expressivity_engineering_qwen3_1p7b.pdf)
- [LaTeX source](docs/technical_report/expressivity_engineering_qwen3_1p7b.tex)
- [Source layout and build notes](docs/technical_report/README.md)
- [Corrected consolidated result table](docs/experiment_records/2026-07-18_qwen3-1p7b-math-ood-corrected-consolidated-master-table.md)

The report defines the EE taxonomy, documents the exact TriGLU/ToTGLU
implementation and six-GPU training contract, presents corrected matched-step
Math and OOD results, records protocol and response-cap boundaries, and
separates supported findings from proposed scaling directions.

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

The main completed research wave is a matched six-GPU GRPO comparison between
the naive Layer-10 whole-layer update and Layer-10 TriGLU. Both start from the
same untuned Qwen3-1.7B-Base revision, use the same shuffled data ledger and
seeds, and have complete Math and out-of-distribution evaluations at matched
steps 158, 196, 226, 256, and 294. The consolidated result table is in
`docs/experiment_records/2026-07-18_qwen3-1p7b-math-ood-corrected-consolidated-master-table.md`.

### Current implementation and evidence status

| Variant | Inference integration | Training evidence | Current boundary |
| --- | --- | --- | --- |
| Whole-layer baseline | Standard Hugging Face and vLLM Qwen route | Matched six-GPU GRPO through step 294 | Primary control |
| TriGLU | Registered `Qwen3TriGLUForCausalLM`, out-of-tree vLLM plugin, exact greedy-token parity in the bounded gate, and validated live weight synchronization | Matched six-GPU GRPO through step 294 | Primary added-expressivity result |
| SHS | Reference PyTorch/cuBLAS custom projection inside vLLM V1; long-decode continuous batching measured | SFT and bounded runtime/weight-sync evidence | Historical runs resolved the generic wrapper; the registered SHS CausalLM route remains pending |
| OFT | Hugging Face/vLLM scaffold and completed SFT checkpoint path | SFT only; no GRPO update was launched | Deferred until a matched adaptation-policy comparison is defined |

TriGLU's bounded registered-vLLM gate reached 2,872.1 generated tokens/s/GPU
at pressure 64, or 52.8% of the matched vanilla Qwen rate and 151.8% of the
historical SHS reference-path rate. These are operational throughput results,
not a claim of full-logit backend parity. SHS continuous batching was also
demonstrated, but its historical runs selected the generic
`TransformersForCausalLM` route; the separately registered SHS generation
route is therefore still an explicit pending obligation.

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
  architecture-aware vLLM rollout, export, and live weight synchronization.
- Six-rank matched GRPO orchestration and six-way TP=1 parallel evaluation.
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

Paper-aligned hyperparameters captured in `configs/base_qwen3_1p7b.yaml`:

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

### Six-GPU matched-run batch contract

The completed RTX 5090 comparison did not use the paper-aligned
`512/128/8` train/mini/micro batch tuple. To use all six ranks without leaving
two GPUs idle, it used the explicitly approved `504/126/6` tuple:

| Field | Six-rank value |
| --- | ---: |
| Train batch size | `504` prompts |
| PPO mini batch size | `126` prompts |
| PPO micro batch size | `6` prompts |
| Group size | `4` responses per prompt |
| Responses per update | `2,016` |
| Prompts through step 98 | `49,392` |
| Deterministically omitted rows at step 98 | `608 / 50,000` |

Both variants use the same deterministic shuffled permutation, omitted-row
set, initialization seed, rollout seed, response cap, and veRL parquet hashes.
The loader uses `drop_last=True`; consequently, step 98 is labelled a
**near-one-pass** milestone rather than an exact full epoch. No irregular tail
batch was introduced. This is a matched architecture comparison with a 1.56%
exposure deviation from `512 x 98`, not an exact reproduction of the paper's
batch arithmetic.

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
cd /root/autodl-tmp/qwen3_1p7b_single_layer_rl_expressivity_engineering
PYTHON_BIN=/root/miniconda3/bin/python \
bash scripts/remote/autodl_remote_launch.sh configs/training_modes/selected_layer_no_adapter.yaml layer10_no_adapter_pilot
```

See `docs/autodl_sync_and_remote.md` for `screen`, logging, and artifact
pullback conventions.

The repository preserves both the earlier synchronous-HF fallback and the
later architecture-aware vLLM path. Baseline uses the native Qwen vLLM route.
TriGLU uses its registered causal-LM class and plugin, with semantic dispatch
receipts and actor-to-vLLM weight synchronization. SHS has measured vLLM V1
continuous batching through its historical generic-wrapper path, but its
registered causal-LM route is not yet closed. OFT has integration scaffolding
but no matched GRPO result. Backend-specific scores should not be treated as
strictly interchangeable until the pending Eval Parity Matrix is complete.

### Historical SFT pilot

Before the GRPO wave, a two-epoch, 50K-example SFT pilot was completed for SHS,
the whole-layer baseline, TriGLU, and OFT. It was operationally successful but
not a successful model-selection proxy: SHS, baseline, and TriGLU differed by
only 0.32 points in the four-task average, while OFT obtained the lowest
teacher-forced validation loss and simultaneously collapsed on autoregressive
math evaluation. The project therefore treats this SFT wave as negative and
diagnostic evidence about objective mismatch, not as its main architecture
result. The primary comparison is the later matched GRPO wave from the untuned
base model.

The preserved SFT launcher auto-detects the visible GPU count and maintains an
effective packed batch size of 8. With micro-batch size 1, this selects gradient
accumulation 8 on one GPU and 2 on four GPUs:

```bash
cd /root/autodl-tmp/qwen3_1p7b_single_layer_rl_expressivity_engineering
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

OFT was not advanced into the matched GRPO comparison. The planned OFT policy
froze the original Layer-10 SwiGLU projections and trained orthogonal rotations,
whereas TriGLU trained the original SwiGLU jointly with its side branch. A
direct result would therefore conflate added expressivity with different
backbone-SwiGLU adaptation constraints. The OFT tracker remained at step zero;
no OFT GRPO update ran. A future OFT experiment must first define a matched
adaptation policy rather than reuse the deferred configuration as if it were a
fair architecture control.

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
