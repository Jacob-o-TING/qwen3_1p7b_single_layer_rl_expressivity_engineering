# shs_triton_autograd_parity_20260712_v1

Status: **failed**

Backend: Triton multiplicative forward plus PyTorch reference-recompute backward.
This is explicitly not a custom backward kernel.

Loss absolute difference: `0.0011254549026489258`.
Worst gradient cosine: `0.9982286095619202` (`model.layers.10.mlp.shs.down.add_right`).
Worst gradient relative L2: `0.0833333358168602` (`model.layers.10.mlp.shs.down.mul_scale`).
Dispatch: `['triton_forward_reference_recompute_backward', 'triton_forward_reference_recompute_backward', 'triton_forward_reference_recompute_backward']`; fallback: `False`.
Checkpoint reload max difference: `0.0`.
Resumed second-update max difference: `0.0`.
