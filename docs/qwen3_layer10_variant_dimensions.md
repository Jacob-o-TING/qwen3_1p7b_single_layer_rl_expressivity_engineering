# Qwen3-1.7B Layer-10 SwiGLU Variant Dimensions

Qwen3-1.7B-Base dimensions from the model config:

```text
hidden_size = 2048
intermediate_size = 6144
num_hidden_layers = 28
num_attention_heads = 16
num_key_value_heads = 8
head_dim = 128
```

All four configs train the whole selected decoder layer 10 unless noted:
attention projections, RMSNorm parameters, and non-MLP layer parameters remain
trainable. Only embedding and LM head parameters stay frozen by default.

Hard initialization invariant:

- SHS and TriGLU must initially have zero effect on the base Qwen SwiGLU path.
- SHS enforces this by zero-initializing the grid generator output projection,
  so all HyperGrid mul/add grids are exactly zero at step 0.
- SHS multiply and additive deltas use separate deterministic shuffled row/col
  block maps, so the same projection no longer couples multiply and additive
  inductive biases through one shared partition.
- TriGLU enforces this by zero-initializing the side return projection, so the
  residual multiplier is exactly `1 + side_scale * tanh(0) = 1` at step 0.
- Do not replace these with merely "small random" initializations unless the run
  name and config explicitly label that as a different ablation.

## Ordered run list

```text
1. configs/layer10_whole_layer_shs.yaml
2. configs/layer10_whole_layer_baseline.yaml
3. configs/layer10_whole_layer_triglu_side_ffn.yaml
4. configs/layer10_whole_layer_oft.yaml
```

Use:

```bash
bash scripts/launch_layer10_ordered_variants.sh
```

## Deferred 20% Budget Option B

This is recorded for a later lower-capacity ablation, but it is not the active
configuration right now.

- SHS option B: `grid=16 x 32`, `hypergrid_generator_hidden=1024`,
  `hypergrid_add_rank=64`; exact added capacity `8,918,022` parameters, or
  `17.72%` of the selected Qwen decoder layer.
- TriGLU option B: `side_dim=512`, `side_hidden=1024`; exact added capacity
  `9,972,224` parameters, or `19.81%` of the selected Qwen decoder layer.
- OFT remains unchanged: `block_size=64`, implemented raw block-matrix
  capacity `655,360` parameters, or `1.30%` of the selected Qwen decoder layer.

## SHS over SwiGLU

Config: `configs/layer10_whole_layer_shs.yaml`

- Scope: only layer-10 `mlp`.
- Base SwiGLU: trainable jointly with the rest of layer 10.
- HyperGrid target projections: `gate_proj`, `up_proj`, `down_proj`.
- Grid: `32 x 32`.
- Grid generator hidden: `2048`.
- Additive basis rank: `64`.
- Projection seed offsets from `hypergrid_shuffle_seed`: gate `+101`, up
  `+202`, down `+303`.
- Per-projection Add/Multiply shuffle seed offsets: `mul_row=+0`,
  `mul_col=+7919`, `add_row=+15485863`, `add_col=+15493782`.
- Each wrapped Linear now stores four persistent deterministic maps:
  `mul_row_block_ids`, `mul_col_block_ids`, `add_row_block_ids`, and
  `add_col_block_ids`.
- Init scales: `mul=0.001`, `add=0.001`.
- Initial effect: exact zero delta through zero-initialized
  `shs.grid_generator.out`.

Exact added trainable capacity for the SHS machinery is `22,554,630`
parameters, or `44.81%` of the selected Qwen decoder layer. Base Qwen MLP
weights are not copied; the wrapper modulates the original projections. This is
the active high-capacity setting for the first comparison wave.

## TriGLU Side FFN over SwiGLU

Config: `configs/layer10_whole_layer_triglu_side_ffn.yaml`

- Scope: only layer-10 `mlp`.
- Base SwiGLU: trainable jointly with the rest of layer 10.
- Side bottleneck: `2048 -> 512`.
- Side TriGLU hidden: `2048`.
- Side return: `2048 -> 6144`.
- Semantics: residual delta multiplier,
  `hidden = base_swiglu_hidden * (1 + 0.1 * tanh(side))`.
- Side return projection is zero-initialized, so the initial function matches
  the base SwiGLU path.

Exact added trainable capacity is `20,986,368` parameters, or `41.69%` of the
selected Qwen decoder layer. This is the active high-capacity setting for the
first comparison wave.

## OFT on SwiGLU

Config: `configs/layer10_whole_layer_oft.yaml`

- Scope: only layer-10 `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`.
- Base SwiGLU projection weights: frozen.
- OFT block size: `64`.
- OFT blocks: `2048 / 64 = 32` for gate/up input rotations, `6144 / 64 = 96`
  for the down projection input rotation.
- OFT trainables are initialized to identity rotations through zero skew
  matrices.
- Attention, RMSNorm, and other non-SwiGLU layer-10 parameters remain
  trainable.

Implemented OFT trainable capacity is `655,360` raw block-matrix parameters, or
`1.30%` of the selected Qwen decoder layer, before optimizer states. A
triangular-compressed skew parameterization would have `322,560` independent
degrees of freedom, but the current implementation stores full `64 x 64` raw
blocks.

## Single-Layer Baseline

Config: `configs/layer10_whole_layer_baseline.yaml`

- Scope: whole decoder layer 10.
- Architecture variant: identity.
- All layer-10 parameters, including attention, RMSNorm, and SwiGLU, train.
- Embeddings and LM head stay frozen.
