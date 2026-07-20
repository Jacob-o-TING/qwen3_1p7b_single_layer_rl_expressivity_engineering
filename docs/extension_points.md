# Extension points

This scaffold separates backbone trainability from architecture surgery.

## Layer policy

`freeze_policy` decides which original Qwen decoder layers can update and
whether architecture/adapters can update.

Explicit fields:

- `backbone_train_mode: frozen`: no original Qwen backbone weights train.
- `backbone_train_mode: selected`: only `train_layers` train.
- `backbone_train_mode: full`: all backbone weights train, subject to
  `freeze_embeddings` and `freeze_lm_head`.
- `train_adapter_modules: true`: adapter / architecture module params train.
- `train_adapter_modules: false`: adapter / architecture module params remain
  frozen or absent.

Required training modes:

| Case | Backbone | Adapter / architecture module |
| --- | --- | --- |
| Frozen backbone + adapter | `frozen` | `train_adapter_modules: true` |
| Selected backbone, no adapter | `selected` | `false` |
| Selected/full backbone + adapter | `selected` or `full` | `true` |
| No-adapter baseline | `selected` or `full` | `false`, `identity` variant |

The utility entry point is:

```python
from qwen_single_layer_rl.layers import apply_freeze_policy
report = apply_freeze_policy(model, cfg)
```

## Architecture variants

`architecture_variant` decides whether to alter the model before training.

Built-ins:

- `identity`: no surgery.
- `oft_like`: records an OFT-like plan but does not inject torch modules yet.

To add a variant:

1. Create `src/qwen_single_layer_rl/model_surgery/my_variant.py`.
2. Subclass `ArchitectureVariant`.
3. Implement `apply(model, config)`.
4. Register it in `registry.py`.
5. Add a config under `configs/adapters/`.

Apply surgery before freezing:

```python
variant = build_variant(cfg)
model = variant.apply(model, cfg)
report = apply_freeze_policy(model, cfg)
```

This ensures newly injected module names are included in the trainable audit.

## Recommended surgery contract

Each variant should:

- Leave a manifest on the model, such as `_qwen_single_layer_rl_variant_manifest`.
- Use deterministic initialization controlled by the global seed.
- Emit stable parameter name hints for trainable audits.
- Avoid hidden changes to tokenizer, reward, or dataset behavior.
- Be compatible with FSDP wrapping in the final veRL integration.
- Ensure injected parameter names include a marker from
  `freeze_policy.adapter_name_markers`, such as `.adapters.` or `.oft_like.`.

## OFT-like placeholder notes

The `oft_like` config intentionally names target modules but does not implement
the transform. Before a real run, decide:

- Whether the transform is applied to attention only, MLP only, or both.
- Whether the transform updates base weights, wraps linear layers, or adds a
  separate trainable branch.
- How orthogonality is parameterized and constrained.
- Whether only transform params train or both transform and selected base layer
  params train.
