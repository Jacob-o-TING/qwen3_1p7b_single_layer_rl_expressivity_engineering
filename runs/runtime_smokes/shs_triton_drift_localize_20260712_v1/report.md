# shs_triton_drift_localize_20260712_v1

Status: **failed**

Fixed logits cosine gate: `0.9999` (not relaxed).

## Panel full_context_reproduction

- `reference_equivalent`: logits cosine 1.00000000, relative L2 0.00000000; first amplification `None`; dispatch `['reference', 'reference', 'reference']`.
- `grouped_bf16`: logits cosine 0.99974746, relative L2 0.00460731; first amplification `layer11_residual`; dispatch `['triton', 'triton', 'triton']`.
- `grouped_fp32_accumulate`: logits cosine 0.99973667, relative L2 0.00489588; first amplification `layer11_residual`; dispatch `['triton_fp32', 'triton_fp32', 'triton_fp32']`.

## Panel flattened_pressure_1

- `reference_equivalent`: logits cosine 1.00000000, relative L2 0.00000000; first amplification `None`; dispatch `['reference', 'reference', 'reference']`.
- `grouped_bf16`: logits cosine 1.00000000, relative L2 0.00000000; first amplification `None`; dispatch `['triton', 'triton', 'triton']`.
- `grouped_fp32_accumulate`: logits cosine 1.00000000, relative L2 0.00000000; first amplification `None`; dispatch `['triton_fp32', 'triton_fp32', 'triton_fp32']`.

## Panel flattened_pressure_32

- `reference_equivalent`: logits cosine 1.00000000, relative L2 0.00000000; first amplification `None`; dispatch `['reference', 'reference', 'reference']`.
- `grouped_bf16`: logits cosine 0.99935973, relative L2 0.00439741; first amplification `layer11_residual`; dispatch `['triton', 'triton', 'triton']`.
- `grouped_fp32_accumulate`: logits cosine 0.99936032, relative L2 0.00490469; first amplification `layer11_residual`; dispatch `['triton_fp32', 'triton_fp32', 'triton_fp32']`.
