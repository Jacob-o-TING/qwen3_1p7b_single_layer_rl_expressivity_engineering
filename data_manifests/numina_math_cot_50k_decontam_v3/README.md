# NuminaMath 50K Selection Provenance

`selected_rows.tsv` is the complete row-level map from the canonical
materialized train/validation data to the online NuminaMath-CoT source. Source
and materialized indices are zero-based. `manifest.json` pins file hashes,
source split content/order digests, the upstream data revision, and the
ModelScope mirror route used for recovery.

The ledger is intentionally metadata-only: it covers all 50,100 selected rows
without duplicating prompts or solutions in Git.

Regenerate on the configured AutoDL environment from the project root:

```bash
PYTHONPATH=src envs/evalscope181/bin/python \
  -m qwen_single_layer_rl.data.build_selection_provenance \
  --train-jsonl data/numina_math_cot_50k_decontam_v3/train.jsonl \
  --validation-jsonl data/numina_math_cot_50k_decontam_v3/val.jsonl \
  --source-contract configs/data/numina_math_cot_e8b6aad.json \
  --source-hub modelscope \
  --validation-selection-manifest \
    data/numina_math_cot_50k_decontam_v3/validation_manifest.json \
  --candidate-source-split data/numina_math_cot_50k/val.jsonl=test \
  --candidate-source-split data/numina_math_cot_50k/train.jsonl=train \
  --out-dir data_manifests/numina_math_cot_50k_decontam_v3
```

Expected ledger SHA-256:
`1097fdde429daf60eca6cbfb9b4e9f2f49ca5386133bd6618155087069a3968d`.
