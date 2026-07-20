"""SHS multiplicative projection kernels and reference-recompute autograd.

The Triton path fuses the base projection and multiplicative HyperGrid term.
It intentionally does not implement the separate additive low-rank path.  The
first training-capable path uses Triton only for forward and recomputes the
reference PyTorch algebra during backward; it is not a custom backward kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on CPU-only installs
    triton = None
    tl = None


Backend = Literal["auto", "reference", "triton"]


@dataclass(frozen=True)
class SHSProjectionResult:
    output: torch.Tensor
    backend: Literal["reference", "triton"]
    fallback_reason: str | None = None


def _validate_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    mul_grid: torch.Tensor,
    row_ids: torch.Tensor,
    col_ids: torch.Tensor,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    if x.ndim < 2:
        raise ValueError(f"x must have at least 2 dimensions, got {x.shape}")
    if weight.ndim != 2 or weight.shape[1] != x.shape[-1]:
        raise ValueError(f"weight {weight.shape} is incompatible with x {x.shape}")
    leading_shape = tuple(x.shape[:-1])
    if tuple(mul_grid.shape[:-2]) != leading_shape:
        raise ValueError(f"mul_grid leading shape {mul_grid.shape[:-2]} != x {leading_shape}")
    if row_ids.shape != (weight.shape[0],) or col_ids.shape != (weight.shape[1],):
        raise ValueError("row_ids/col_ids must map every output/input feature")
    if row_ids.dtype not in (torch.int32, torch.int64) or col_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("row_ids and col_ids must be int32 or int64")
    # CUDA maps are deterministic persistent model buffers. Avoid a scalar
    # reduction and host synchronization on every decode invocation.
    if not row_ids.is_cuda and row_ids.numel() and int(row_ids.max()) >= mul_grid.shape[-2]:
        raise ValueError("row_ids exceed mul_grid rows")
    if not col_ids.is_cuda and col_ids.numel() and int(col_ids.max()) >= mul_grid.shape[-1]:
        raise ValueError("col_ids exceed mul_grid columns")
    return x.reshape(-1, x.shape[-1]), leading_shape


def shs_modulated_projection_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    mul_grid: torch.Tensor,
    row_ids: torch.Tensor,
    col_ids: torch.Tensor,
    mul_scale: float | torch.Tensor,
) -> torch.Tensor:
    """Reference algebra matching the original per-column-block SHS loop."""
    x_flat, leading_shape = _validate_inputs(x, weight, mul_grid, row_ids, col_ids)
    grid_flat = torch.tanh(mul_grid.reshape(-1, *mul_grid.shape[-2:])).to(x.dtype)
    result = F.linear(x_flat, weight.to(x.dtype))
    delta = torch.zeros_like(result)
    for col_id in range(mul_grid.shape[-1]):
        indices = torch.nonzero(col_ids == col_id, as_tuple=True)[0]
        if indices.numel() == 0:
            continue
        base_col = F.linear(x_flat.index_select(1, indices), weight.to(x.dtype).index_select(1, indices))
        values = grid_flat[:, :, col_id].index_select(1, row_ids)
        delta = delta + base_col * values
    scale = torch.as_tensor(mul_scale, dtype=x.dtype, device=x.device)
    return (result + scale * delta).reshape(*leading_shape, weight.shape[0])


if triton is not None:

    @triton.jit
    def _shs_modulated_projection_kernel(
        x_ptr,
        weight_ptr,
        grid_ptr,
        row_ids_ptr,
        col_ids_ptr,
        out_ptr,
        scale_ptr,
        n_size: tl.constexpr,
        k_size: tl.constexpr,
        grid_rows: tl.constexpr,
        grid_cols: tl.constexpr,
        grid_token_stride: tl.constexpr,
        grid_row_stride: tl.constexpr,
        grid_col_stride: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
        fused_base: tl.constexpr,
    ):
        token = tl.program_id(0)
        n_offsets = tl.program_id(1) * block_n + tl.arange(0, block_n)
        n_mask = n_offsets < n_size
        row_ids = tl.load(row_ids_ptr + n_offsets, mask=n_mask, other=0).to(tl.int32)
        scale = tl.load(scale_ptr).to(tl.float32)
        accumulator = tl.zeros((block_n,), dtype=tl.float32)

        for k_start in range(0, k_size, block_k):
            k_offsets = k_start + tl.arange(0, block_k)
            k_mask = k_offsets < k_size
            x_values = tl.load(x_ptr + token * k_size + k_offsets, mask=k_mask, other=0.0)
            col_ids = tl.load(col_ids_ptr + k_offsets, mask=k_mask, other=0).to(tl.int32)
            weights = tl.load(
                weight_ptr + n_offsets[:, None] * k_size + k_offsets[None, :],
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            grid_offsets = (
                token * grid_token_stride
                + row_ids[:, None] * grid_row_stride
                + col_ids[None, :] * grid_col_stride
            )
            grid_values = tl.load(
                grid_ptr + grid_offsets,
                mask=n_mask[:, None] & k_mask[None, :],
                other=0.0,
            )
            grid_tanh = tl.extra.cuda.libdevice.tanh(grid_values.to(tl.float32))
            modulation = 1.0 + scale * grid_tanh if fused_base else grid_tanh
            accumulator += tl.sum(weights * x_values[None, :] * modulation, axis=1)

        tl.store(out_ptr + token * n_size + n_offsets, accumulator, mask=n_mask)

    @triton.jit
    def _shs_grouped_delta_kernel(
        x_ptr,
        weight_ptr,
        grid_ptr,
        row_ids_ptr,
        permutation_ptr,
        offsets_ptr,
        out_ptr,
        n_size: tl.constexpr,
        k_size: tl.constexpr,
        grid_cols: tl.constexpr,
        grid_token_stride: tl.constexpr,
        grid_row_stride: tl.constexpr,
        grid_col_stride: tl.constexpr,
        block_n: tl.constexpr,
        block_k_col: tl.constexpr,
        accumulate_fp32: tl.constexpr,
    ):
        token = tl.program_id(0)
        n_offsets = tl.program_id(1) * block_n + tl.arange(0, block_n)
        n_mask = n_offsets < n_size
        row_ids = tl.load(row_ids_ptr + n_offsets, mask=n_mask, other=0).to(tl.int32)
        delta = tl.zeros((block_n,), dtype=tl.float32 if accumulate_fp32 else tl.bfloat16)
        col_offsets = tl.arange(0, block_k_col)
        for col_id in range(0, grid_cols):
            start = tl.load(offsets_ptr + col_id).to(tl.int32)
            end = tl.load(offsets_ptr + col_id + 1).to(tl.int32)
            col_mask = col_offsets < (end - start)
            k_indices = tl.load(permutation_ptr + start + col_offsets, mask=col_mask, other=0).to(tl.int32)
            x_values = tl.load(x_ptr + token * k_size + k_indices, mask=col_mask, other=0.0)
            weights = tl.load(
                weight_ptr + n_offsets[:, None] * k_size + k_indices[None, :],
                mask=n_mask[:, None] & col_mask[None, :],
                other=0.0,
            )
            base_col = tl.sum(weights * x_values[None, :], axis=1)
            if not accumulate_fp32:
                base_col = base_col.to(tl.bfloat16)
            grid_offsets = (
                token * grid_token_stride
                + row_ids * grid_row_stride
                + col_id * grid_col_stride
            )
            grid_values = tl.load(grid_ptr + grid_offsets, mask=n_mask, other=0.0)
            grid_tanh = tl.extra.cuda.libdevice.tanh(grid_values.to(tl.float32))
            if accumulate_fp32:
                delta += base_col * grid_tanh
            else:
                delta = (delta + base_col * grid_tanh.to(tl.bfloat16)).to(tl.bfloat16)
        tl.store(out_ptr + token * n_size + n_offsets, delta, mask=n_mask)


def _triton_projection(
    x: torch.Tensor,
    weight: torch.Tensor,
    mul_grid: torch.Tensor,
    row_ids: torch.Tensor,
    col_ids: torch.Tensor,
    mul_scale: float | torch.Tensor,
    *,
    fused_base: bool = True,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not x.is_cuda:
        raise RuntimeError("Triton backend requires CUDA tensors")
    if torch.is_grad_enabled() and any(t.requires_grad for t in (x, weight, mul_grid)):
        raise RuntimeError("SHS Triton projection is inference-only and has no backward")
    x_flat, leading_shape = _validate_inputs(x, weight, mul_grid, row_ids, col_ids)
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"unsupported x dtype: {x.dtype}")
    tensors = (weight, mul_grid, row_ids, col_ids)
    if any(t.device != x.device for t in tensors):
        raise ValueError("all inputs must be on the same CUDA device")
    x_flat = x_flat.contiguous()
    weight = weight.to(x.dtype).contiguous()
    row_ids = row_ids.contiguous()
    col_ids = col_ids.contiguous()
    out = torch.empty((x_flat.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    scale = (
        torch.as_tensor(mul_scale, dtype=torch.float32, device=x.device).reshape(())
        if fused_base
        else mul_grid
    )
    block_n = 32
    block_k = 64
    _shs_modulated_projection_kernel[(x_flat.shape[0], triton.cdiv(weight.shape[0], block_n))](
        x_flat,
        weight,
        mul_grid,
        row_ids,
        col_ids,
        out,
        scale,
        weight.shape[0],
        weight.shape[1],
        mul_grid.shape[-2],
        mul_grid.shape[-1],
        mul_grid.stride(-3),
        mul_grid.stride(-2),
        mul_grid.stride(-1),
        block_n=block_n,
        block_k=block_k,
        fused_base=fused_base,
        num_warps=4,
    )
    return out.reshape(*leading_shape, weight.shape[0])


def shs_modulated_projection(
    x: torch.Tensor,
    weight: torch.Tensor,
    mul_grid: torch.Tensor,
    row_ids: torch.Tensor,
    col_ids: torch.Tensor,
    mul_scale: float | torch.Tensor,
    *,
    backend: Backend = "auto",
) -> SHSProjectionResult:
    """Execute SHS projection and report the backend actually used."""
    if backend not in ("auto", "reference", "triton"):
        raise ValueError(f"unknown backend: {backend}")
    if backend == "reference":
        return SHSProjectionResult(
            shs_modulated_projection_reference(x, weight, mul_grid, row_ids, col_ids, mul_scale),
            "reference",
        )
    try:
        return SHSProjectionResult(
            _triton_projection(x, weight, mul_grid, row_ids, col_ids, mul_scale),
            "triton",
        )
    except (RuntimeError, TypeError, ValueError) as exc:
        if backend == "triton":
            raise
        return SHSProjectionResult(
            shs_modulated_projection_reference(x, weight, mul_grid, row_ids, col_ids, mul_scale),
            "reference",
            fallback_reason=f"{type(exc).__name__}: {exc}",
        )


def shs_multiplicative_delta_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    mul_grid: torch.Tensor,
    row_ids: torch.Tensor,
    col_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute only the unscaled SHS multiplicative delta with Triton.

    Keeping the base projection on its native GEMM preserves its BF16
    reduction order while still replacing 32 sliced multiplicative GEMMs.
    """
    return _triton_projection(
        x,
        weight,
        mul_grid,
        row_ids,
        col_ids,
        0.0,
        fused_base=False,
    )


def shs_grouped_multiplicative_delta_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    mul_grid: torch.Tensor,
    row_ids: torch.Tensor,
    col_permutation: torch.Tensor,
    col_offsets: torch.Tensor,
    *,
    accumulate_fp32: bool = False,
) -> torch.Tensor:
    """Match the reference column-group reduction order in one Triton launch."""
    if triton is None or not x.is_cuda:
        raise RuntimeError("grouped SHS Triton delta requires CUDA and Triton")
    if x.dtype != torch.bfloat16:
        raise TypeError("grouped SHS Triton delta currently supports BF16 inference only")
    if x.ndim < 2 or weight.ndim != 2 or weight.shape[1] != x.shape[-1]:
        raise ValueError("grouped SHS x/weight shapes are incompatible")
    leading_shape = tuple(x.shape[:-1])
    if tuple(mul_grid.shape[:-2]) != leading_shape or row_ids.shape != (weight.shape[0],):
        raise ValueError("grouped SHS grid or row map shape is incompatible")
    x_flat = x.reshape(-1, x.shape[-1])
    tensors = (weight, mul_grid, row_ids, col_permutation, col_offsets)
    if any(tensor.device != x.device for tensor in tensors):
        raise ValueError("all grouped SHS inputs must be on the same CUDA device")
    x_flat = x_flat.contiguous()
    weight = weight.to(x.dtype).contiguous()
    out = torch.empty((x_flat.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    block_n = 32
    _shs_grouped_delta_kernel[(x_flat.shape[0], triton.cdiv(weight.shape[0], block_n))](
        x_flat,
        weight,
        mul_grid,
        row_ids,
        col_permutation,
        col_offsets,
        out,
        weight.shape[0],
        weight.shape[1],
        mul_grid.shape[-1],
        mul_grid.stride(-3),
        mul_grid.stride(-2),
        mul_grid.stride(-1),
        block_n=block_n,
        block_k_col=256,
        accumulate_fp32=accumulate_fp32,
        num_warps=4,
    )
    return out.reshape(*leading_shape, weight.shape[0])


def shs_grouped_multiplicative_delta_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    mul_grid: torch.Tensor,
    row_ids: torch.Tensor,
    col_permutation: torch.Tensor,
    col_offsets: torch.Tensor,
) -> torch.Tensor:
    """Compute the grouped multiplicative delta in the canonical loop order."""
    if x.ndim < 2 or weight.ndim != 2 or weight.shape[1] != x.shape[-1]:
        raise ValueError("grouped SHS x/weight shapes are incompatible")
    leading_shape = tuple(x.shape[:-1])
    if tuple(mul_grid.shape[:-2]) != leading_shape or row_ids.shape != (weight.shape[0],):
        raise ValueError("grouped SHS grid or row map shape is incompatible")
    if col_offsets.shape != (mul_grid.shape[-1] + 1,):
        raise ValueError("grouped SHS offsets must delimit every grid column")
    if col_permutation.numel() != weight.shape[1]:
        raise ValueError("grouped SHS permutation must cover every input feature")
    tensors = (weight, mul_grid, row_ids, col_permutation, col_offsets)
    if any(tensor.device != x.device for tensor in tensors):
        raise ValueError("all grouped SHS inputs must be on the same device")

    dtype = x.dtype
    grid = torch.tanh(mul_grid).to(dtype)
    delta = torch.zeros((*leading_shape, weight.shape[0]), dtype=dtype, device=x.device)
    for col_id in range(mul_grid.shape[-1]):
        start = int(col_offsets[col_id])
        end = int(col_offsets[col_id + 1])
        indices = col_permutation[start:end]
        base_col = F.linear(
            x.index_select(-1, indices),
            weight.to(dtype).index_select(1, indices),
        )
        values = grid[..., :, col_id].index_select(-1, row_ids)
        delta = delta + base_col * values
    return delta


class _GroupedMultiplicativeDeltaReferenceRecompute(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        mul_grid: torch.Tensor,
        row_ids: torch.Tensor,
        col_permutation: torch.Tensor,
        col_offsets: torch.Tensor,
    ) -> torch.Tensor:
        ctx.save_for_backward(x, weight, mul_grid, row_ids, col_permutation, col_offsets)
        return shs_grouped_multiplicative_delta_triton(
            x,
            weight,
            mul_grid,
            row_ids,
            col_permutation,
            col_offsets,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight, mul_grid, row_ids, col_permutation, col_offsets = ctx.saved_tensors
        with torch.enable_grad():
            x_ref = x.detach().requires_grad_(True)
            weight_ref = weight.detach().requires_grad_(True)
            grid_ref = mul_grid.detach().requires_grad_(True)
            output = shs_grouped_multiplicative_delta_reference(
                x_ref,
                weight_ref,
                grid_ref,
                row_ids,
                col_permutation,
                col_offsets,
            )
            grad_x, grad_weight, grad_grid = torch.autograd.grad(
                output,
                (x_ref, weight_ref, grid_ref),
                grad_output,
                create_graph=torch.is_grad_enabled(),
            )
        return grad_x, grad_weight, grad_grid, None, None, None


def shs_grouped_multiplicative_delta_triton_reference_recompute(
    x: torch.Tensor,
    weight: torch.Tensor,
    mul_grid: torch.Tensor,
    row_ids: torch.Tensor,
    col_permutation: torch.Tensor,
    col_offsets: torch.Tensor,
) -> torch.Tensor:
    """Triton forward with a reference-recompute autograd backward."""
    return _GroupedMultiplicativeDeltaReferenceRecompute.apply(
        x,
        weight,
        mul_grid,
        row_ids,
        col_permutation,
        col_offsets,
    )
