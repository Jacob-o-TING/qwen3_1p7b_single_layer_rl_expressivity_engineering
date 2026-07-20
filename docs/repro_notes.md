# Reproduction notes

## Paper alignment

Target paper: "Is One Layer Enough? Training A Single Transformer Layer Can
Match Full-Parameter RL Training" (arXiv 2607.01232).

Qwen3 experiment setup:

- Qwen3-1.7B-Base, 28 layers.
- GRPO through veRL.
- AdamW optimizer.
- NuminaMath-CoT, randomly downsampled to 50K problems after decontamination.
- Single-layer backbone-only training freezes all layers except one decoder
  layer, plus embeddings and LM head.
- Adapter-only training freezes the original Qwen backbone and trains only the
  injected architecture/adapters.
- Joint training trains selected or full backbone weights plus injected modules.
- Full no-adapter baseline uses the same hyperparameters but all backbone
  parameters are unfrozen; configure embeddings and LM head explicitly.

Key Qwen3-1.7B layer anchors:

- Layer 10: strongest reported math contribution in the summary table.
- Layer 12: also high contribution and best overall in the shown table.
- Layers 11-15: middle-5 heuristic for Qwen3-1.7B.
- B5/B10: contribution-guided sets should be replaced by exact local scan
  rankings if you rerun the full layer sweep.

## Reproducibility checklist

- Record exact Git commit of this scaffold and exact veRL commit.
- Record model revision hash from Hugging Face or ModelScope.
- Record dataset source, revision, row count before and after decontamination.
- Record whether data came from HF streaming, cached HF parquet, or a local
  mirror; save `data/numina_math_cot_50k/prep_manifest.json`.
- Save sampling seed and benchmark decontamination hash files.
- Save resolved config JSON, trainable parameter audit, and launch command.
- Save per-run logs, metrics, `best.pt`, and `latest.pt`.
- Avoid treating remote GPU disks as archival storage.

## Randomness sources to audit before full runs

- Python `random`.
- NumPy RNG.
- Torch CPU and CUDA RNG.
- Model initialization for newly added adapters/transforms.
- Dataset shuffling and sampling.
- Dataloader worker seeds.
- Rollout sampling temperature/top-p/top-k.
- vLLM/SGLang generation seeds if exposed by the installed revision.
- Resume checkpoint RNG state.

## Naming scheme proposal

Use names that encode architecture, layer policy, seed, dataset, and algorithm:

```text
qwen3_1p7b_<layer-policy>_<variant>_numina50k_grpo_seed<seed>
```

Examples:

- `qwen3_1p7b_layer10_identity_numina50k_grpo_seed20260707`
- `qwen3_1p7b_middle11_15_identity_numina50k_grpo_seed20260707`
- `qwen3_1p7b_layer10_oftlike_r16_numina50k_grpo_seed20260707`
- `qwen3_1p7b_adapteronly_oftlike_r16_numina50k_grpo_seed20260707`
- `qwen3_1p7b_full_noadapter_numina50k_grpo_seed20260707`

Ask for explicit approval before launching a new controlled benchmark wave.
