from __future__ import annotations

import math
import json
import os
import itertools
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from qwen_single_layer_rl.kernels.shs_modulated_projection import (
    shs_grouped_multiplicative_delta_triton,
    shs_grouped_multiplicative_delta_triton_reference_recompute,
)
from qwen_single_layer_rl.vllm.custom_ffn_contract import write_dispatch_receipt_once


# Effective seed = hypergrid_shuffle_seed + projection offset + delta-map offset.
SHS_PROJECTION_SEED_OFFSETS = {
    "gate": 101,
    "up": 202,
    "down": 303,
}

SHS_DELTA_SEED_OFFSETS = {
    "mul_row": 0,
    "mul_col": 7919,
    "add_row": 15485863,
    "add_col": 15493782,
}


def _get_layers(model: Any) -> Any:
    candidates = [
        ("model", "layers"),
        ("base_model", "model", "model", "layers"),
        ("module", "model", "layers"),
    ]
    for path in candidates:
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise AttributeError("Could not find Qwen decoder layers on model")


def _model_dims(model: Any, params: dict[str, Any]) -> tuple[int, int]:
    cfg = getattr(model, "config", None)
    d_model = int(params.get("hidden_size") or getattr(cfg, "hidden_size", 2048))
    intermediate = int(params.get("intermediate_size") or getattr(cfg, "intermediate_size", 6144))
    return d_model, intermediate


def _act(base_mlp: Any, x: torch.Tensor) -> torch.Tensor:
    act_fn = getattr(base_mlp, "act_fn", None)
    return act_fn(x) if act_fn is not None else F.silu(x)


def make_shuffled_block_ids(size: int, num_blocks: int, seed: int) -> torch.Tensor:
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    if num_blocks > size:
        raise ValueError("num_blocks cannot exceed axis size")
    base_ids = torch.div(torch.arange(size, device="cpu") * num_blocks, size, rounding_mode="floor")
    base_ids = base_ids.clamp_max(num_blocks - 1).to(torch.long)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    permutation = torch.randperm(size, generator=generator, device="cpu")
    ids = torch.empty(size, dtype=torch.long, device="cpu")
    ids[permutation] = base_ids
    return ids


class DirectTokenSwiGLUGridGenerator(nn.Module):
    def __init__(self, d_model: int, rows: int, cols: int, hidden_dim: int) -> None:
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.up = nn.Linear(d_model, hidden_dim)
        self.gate = nn.Linear(d_model, hidden_dim)
        self.out = nn.Linear(hidden_dim, 3 * 2 * rows * cols)
        # Hard invariant: SHS starts as an exact no-op. Zero grids make every
        # HyperGrid mul/add delta exactly zero, independent of random hidden
        # generator weights or nonzero delta scales.
        self.initial_effect = "exact_zero_delta"
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        generator_input = x.to(self.up.weight.dtype)
        hidden = self.up(generator_input) * F.silu(self.gate(generator_input))
        grids = self.out(hidden).view(*x.shape[:-1], 3, 2, self.rows, self.cols)
        return (
            grids[..., 0, 0, :, :],
            grids[..., 0, 1, :, :],
            grids[..., 1, 0, :, :],
            grids[..., 1, 1, :, :],
            grids[..., 2, 0, :, :],
            grids[..., 2, 1, :, :],
        )


class ShuffledHyperGridDeltaLinear(nn.Module):
    def __init__(
        self,
        base_linear: nn.Linear,
        *,
        rows: int,
        cols: int,
        shuffle_seed: int,
        mul_init_scale: float,
        add_init_scale: float,
        add_rank: int,
        shuffle_seed_offsets: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        # The owning Qwen MLP already registers this projection. Keeping a
        # second registered reference creates duplicate checkpoint keys and
        # makes safetensors export ambiguous.
        object.__setattr__(self, "_base_linear", base_linear)
        self.rows = rows
        self.cols = cols
        self.add_rank = int(add_rank)
        self.inference_mul_backend = "reference"
        self.last_inference_mul_backend = "reference"
        self.shuffle_seed = int(shuffle_seed)
        self.shuffle_seed_offsets = dict(SHS_DELTA_SEED_OFFSETS)
        if shuffle_seed_offsets:
            self.shuffle_seed_offsets.update({name: int(value) for name, value in shuffle_seed_offsets.items()})
        self.mul_scale = nn.Parameter(torch.tensor([float(mul_init_scale)], dtype=torch.float32))
        self.add_scale = nn.Parameter(torch.tensor([float(add_init_scale)], dtype=torch.float32))
        out_features, in_features = base_linear.weight.shape
        self.add_left = nn.Parameter(torch.empty(out_features, self.add_rank))
        self.add_right = nn.Parameter(torch.empty(self.add_rank, in_features))
        self.register_buffer(
            "mul_row_block_ids",
            make_shuffled_block_ids(out_features, rows, self.shuffle_seed + self.shuffle_seed_offsets["mul_row"]),
            persistent=True,
        )
        self.register_buffer(
            "mul_col_block_ids",
            make_shuffled_block_ids(in_features, cols, self.shuffle_seed + self.shuffle_seed_offsets["mul_col"]),
            persistent=True,
        )
        self.register_buffer(
            "add_row_block_ids",
            make_shuffled_block_ids(out_features, rows, self.shuffle_seed + self.shuffle_seed_offsets["add_row"]),
            persistent=True,
        )
        self.register_buffer(
            "add_col_block_ids",
            make_shuffled_block_ids(in_features, cols, self.shuffle_seed + self.shuffle_seed_offsets["add_col"]),
            persistent=True,
        )
        self._mul_col_index_names = tuple(f"_mul_col_indices_{col_id}" for col_id in range(cols))
        self._add_col_index_names = tuple(f"_add_col_indices_{col_id}" for col_id in range(cols))
        for col_id, buffer_name in enumerate(self._mul_col_index_names):
            self.register_buffer(
                buffer_name,
                torch.nonzero(self.mul_col_block_ids == col_id, as_tuple=True)[0],
                persistent=False,
            )
        for col_id, buffer_name in enumerate(self._add_col_index_names):
            self.register_buffer(
                buffer_name,
                torch.nonzero(self.add_col_block_ids == col_id, as_tuple=True)[0],
                persistent=False,
            )
        mul_col_indices = [getattr(self, name) for name in self._mul_col_index_names]
        self.register_buffer("_mul_col_permutation", torch.cat(mul_col_indices), persistent=False)
        self.register_buffer(
            "_mul_col_offsets",
            torch.tensor([0] + list(itertools.accumulate(index.numel() for index in mul_col_indices))),
            persistent=False,
        )
        nn.init.normal_(self.add_left, mean=0.0, std=1.0 / math.sqrt(self.add_rank))
        nn.init.normal_(self.add_right, mean=0.0, std=1.0 / math.sqrt(in_features))

    @property
    def base_linear(self) -> nn.Module:
        return self._base_linear

    def rebind_base_linear(self, base_linear: nn.Module) -> None:
        object.__setattr__(self, "_base_linear", base_linear)

    def set_inference_mul_backend(self, backend: str) -> None:
        if backend not in ("reference", "triton", "triton_fp32", "triton_reference_recompute"):
            raise ValueError(f"unsupported SHS inference backend: {backend}")
        self.inference_mul_backend = backend

    def forward(self, x: torch.Tensor, mul_grid: torch.Tensor, add_grid: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        weight = self.base_linear.weight.to(dtype)
        add_grid = torch.tanh(add_grid).to(dtype)
        use_triton_inference = (
            self.inference_mul_backend in ("triton", "triton_fp32")
            and not self.training
            and not torch.is_grad_enabled()
        )
        use_triton_recompute = self.inference_mul_backend == "triton_reference_recompute"
        use_triton = use_triton_inference or use_triton_recompute
        if self.mul_row_block_ids.device != x.device:
            for name, buffer in tuple(self.named_buffers(recurse=False)):
                if buffer.device != x.device:
                    setattr(self, name, buffer.to(device=x.device, non_blocking=True))
        mul_row_block_ids = self.mul_row_block_ids
        add_row_block_ids = self.add_row_block_ids
        if use_triton:
            projection = (
                shs_grouped_multiplicative_delta_triton_reference_recompute
                if use_triton_recompute and torch.is_grad_enabled()
                else shs_grouped_multiplicative_delta_triton
            )
            kwargs = {"accumulate_fp32": True} if self.inference_mul_backend == "triton_fp32" else {}
            mul_delta = projection(
                x,
                weight,
                mul_grid,
                mul_row_block_ids,
                self._mul_col_permutation,
                self._mul_col_offsets,
                **kwargs,
            )
            base_and_mul = self.base_linear(x) + self.mul_scale.to(dtype) * mul_delta
            self.last_inference_mul_backend = (
                "triton_forward_reference_recompute_backward"
                if use_triton_recompute
                else self.inference_mul_backend
            )
            receipt_path = os.environ.get("SHS_DISPATCH_RECEIPT")
            if receipt_path and not getattr(self, "_dispatch_receipt_written", False):
                with open(receipt_path, "a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "pid": os.getpid(),
                                "backend": self.last_inference_mul_backend,
                                "custom_backward": False,
                                "in_features": int(weight.shape[1]),
                                "out_features": int(weight.shape[0]),
                                "buffer_devices": sorted({str(buffer.device) for buffer in self.buffers()}),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                self._dispatch_receipt_written = True
        else:
            base_out = self.base_linear(x)
            mul_grid = torch.tanh(mul_grid).to(dtype)
            mul_delta = torch.zeros_like(base_out)
            for col_id in range(self.cols):
                in_idx = getattr(self, self._mul_col_index_names[col_id])
                x_sub = x.index_select(-1, in_idx)
                weight_sub = weight.index_select(1, in_idx)
                base_col = F.linear(x_sub, weight_sub)
                mul_values = mul_grid[..., :, col_id].index_select(-1, mul_row_block_ids)
                mul_delta = mul_delta + base_col * mul_values
            base_and_mul = base_out + self.mul_scale.to(dtype) * mul_delta
            self.last_inference_mul_backend = "reference"
            receipt_path = os.environ.get("SHS_DISPATCH_RECEIPT")
            if receipt_path and not getattr(self, "_dispatch_receipt_written", False):
                with open(receipt_path, "a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "pid": os.getpid(),
                                "backend": "reference",
                                "in_features": int(weight.shape[1]),
                                "out_features": int(weight.shape[0]),
                                "buffer_devices": sorted({str(buffer.device) for buffer in self.buffers()}),
                                "fallback": False,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                self._dispatch_receipt_written = True

        add_delta = torch.zeros_like(base_and_mul)
        for col_id in range(self.cols):
            in_idx = getattr(self, self._add_col_index_names[col_id])
            x_sub = x.index_select(-1, in_idx)
            add_right_sub = self.add_right.to(dtype).index_select(1, in_idx)
            add_hidden = F.linear(x_sub, add_right_sub)
            add_col = F.linear(add_hidden, self.add_left.to(dtype))
            add_values = add_grid[..., :, col_id].index_select(-1, add_row_block_ids)
            add_delta = add_delta + add_col * add_values
        return base_and_mul + self.add_scale.to(dtype) * add_delta


class QwenSwiGLUSHSWrapper(nn.Module):
    def __init__(self, base_mlp: nn.Module, d_model: int, params: dict[str, Any]) -> None:
        super().__init__()
        self.base_mlp = base_mlp
        self.initial_effect = "exact_identity_via_zero_shs_grids"
        rows = int(params.get("hypergrid_rows", 32))
        cols = int(params.get("hypergrid_cols", 32))
        hidden = int(params.get("hypergrid_generator_hidden", d_model))
        add_rank = int(params.get("hypergrid_add_rank", 64))
        seed = int(params.get("hypergrid_shuffle_seed", 20260707))
        projection_offsets = dict(SHS_PROJECTION_SEED_OFFSETS)
        projection_offsets.update(
            {name: int(value) for name, value in (params.get("hypergrid_projection_seed_offsets") or {}).items()}
        )
        shuffle_seed_offsets = dict(SHS_DELTA_SEED_OFFSETS)
        shuffle_seed_offsets.update(
            {name: int(value) for name, value in (params.get("hypergrid_shuffle_seed_offsets") or {}).items()}
        )
        self.projection_seed_offsets = projection_offsets
        self.shuffle_seed_offsets = shuffle_seed_offsets
        mul_scale = float(params.get("hypergrid_mul_init_scale", 0.001))
        add_scale = float(params.get("hypergrid_add_init_scale", 0.001))
        self.shs = nn.ModuleDict(
            {
                "grid_generator": DirectTokenSwiGLUGridGenerator(d_model, rows, cols, hidden),
                "gate": ShuffledHyperGridDeltaLinear(
                    base_mlp.gate_proj,
                    rows=rows,
                    cols=cols,
                    shuffle_seed=seed + projection_offsets["gate"],
                    mul_init_scale=mul_scale,
                    add_init_scale=add_scale,
                    add_rank=add_rank,
                    shuffle_seed_offsets=shuffle_seed_offsets,
                ),
                "up": ShuffledHyperGridDeltaLinear(
                    base_mlp.up_proj,
                    rows=rows,
                    cols=cols,
                    shuffle_seed=seed + projection_offsets["up"],
                    mul_init_scale=mul_scale,
                    add_init_scale=add_scale,
                    add_rank=add_rank,
                    shuffle_seed_offsets=shuffle_seed_offsets,
                ),
                "down": ShuffledHyperGridDeltaLinear(
                    base_mlp.down_proj,
                    rows=rows,
                    cols=cols,
                    shuffle_seed=seed + projection_offsets["down"],
                    mul_init_scale=mul_scale,
                    add_init_scale=add_scale,
                    add_rank=add_rank,
                    shuffle_seed_offsets=shuffle_seed_offsets,
                ),
            }
        )
        configured_backend = os.environ.get("SHS_INFERENCE_MUL_BACKEND", params.get("inference_mul_backend"))
        if configured_backend in ("reference", "triton", "triton_fp32", "triton_reference_recompute"):
            for projection in ("gate", "up", "down"):
                self.shs[projection].set_inference_mul_backend(configured_backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.shs["gate"].rebind_base_linear(self.base_mlp.gate_proj)
        self.shs["up"].rebind_base_linear(self.base_mlp.up_proj)
        self.shs["down"].rebind_base_linear(self.base_mlp.down_proj)
        gate_mul, gate_add, up_mul, up_add, down_mul, down_add = self.shs["grid_generator"](x)
        gate = self.shs["gate"](x, gate_mul, gate_add)
        up = self.shs["up"](x, up_mul, up_add)
        hidden = _act(self.base_mlp, gate) * up
        return self.shs["down"](hidden, down_mul, down_add)


class QwenSwiGLUTriGLUSideWrapper(nn.Module):
    def __init__(self, base_mlp: nn.Module, d_model: int, intermediate: int, params: dict[str, Any]) -> None:
        super().__init__()
        self.base_mlp = base_mlp
        self.initial_effect = "exact_identity_via_zero_side_return"
        self.custom_ffn_layer_index: int | None = None
        self.last_inference_backend = "not_run"
        side_dim = int(params.get("side_dim", 512))
        side_hidden = int(params.get("side_hidden", 2048))
        self.side_scale = float(params.get("side_scale", 0.1))
        self.triglu_side = nn.ModuleDict(
            {
                "down": nn.Linear(d_model, side_dim, bias=True),
                "value": nn.Linear(side_dim, side_hidden, bias=True),
                "gate": nn.Linear(side_dim, side_hidden, bias=True),
                "ffn_1": nn.Linear(side_dim, side_hidden, bias=True),
                "ffn_2": nn.Linear(side_hidden, side_hidden, bias=True),
                "up": nn.Linear(side_hidden, intermediate, bias=True),
            }
        )
        # Hard invariant: the side branch starts with zero returned signal, so
        # multiplier = 1 + side_scale * tanh(0) = 1 and the base SwiGLU path is
        # exactly unchanged at step 0.
        nn.init.zeros_(self.triglu_side["up"].weight)
        nn.init.zeros_(self.triglu_side["up"].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trunk = _act(self.base_mlp, self.base_mlp.gate_proj(x)) * self.base_mlp.up_proj(x)
        side_z = self.triglu_side["down"](x.float())
        side_value = self.triglu_side["value"](side_z)
        side_gate = F.silu(self.triglu_side["gate"](side_z))
        side_third = F.silu(self.triglu_side["ffn_2"](F.gelu(self.triglu_side["ffn_1"](side_z))))
        side = self.triglu_side["up"](side_value * side_gate * side_third).to(trunk.dtype)
        multiplier = 1.0 + self.side_scale * torch.tanh(side)
        output = self.base_mlp.down_proj(trunk * multiplier)
        self.last_inference_backend = "reference_pytorch_cublas"
        # Registered vLLM writes this receipt after weight loading, before
        # Dynamo traces forward. Keep the helper entirely outside that graph.
        if not getattr(self, "_custom_ffn_dispatch_receipt_written", False):
            write_dispatch_receipt_once(
                self,
                env_var="TRIGLU_DISPATCH_RECEIPT",
                variant="qwen_swiglu_triglu_side",
                backend=self.last_inference_backend,
                payload={
                    "layer_index": self.custom_ffn_layer_index,
                    "side_dim": int(self.triglu_side["down"].weight.shape[0]),
                    "side_hidden": int(self.triglu_side["value"].weight.shape[0]),
                    "parameter_devices": sorted({str(parameter.device) for parameter in self.triglu_side.parameters()}),
                    "parameter_dtypes": sorted({str(parameter.dtype) for parameter in self.triglu_side.parameters()}),
                },
            )
        return output


class OFTRotation(nn.Module):
    def __init__(self, features: int, block_size: int, *, fp32_compute: bool = False) -> None:
        super().__init__()
        if features % block_size != 0:
            raise ValueError(f"OFT block_size={block_size} must divide features={features}")
        self.features = features
        self.block_size = block_size
        self.num_blocks = features // block_size
        self.fp32_compute = bool(fp32_compute)
        self.oft_like = nn.Parameter(torch.zeros(self.num_blocks, block_size, block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        raw = self.oft_like.float()
        skew = raw - raw.transpose(-1, -2)
        eye = torch.eye(self.block_size, device=x.device, dtype=torch.float32).expand(self.num_blocks, -1, -1)
        rotation = torch.linalg.solve(eye + skew, eye - skew)
        blocks = x.float() if self.fp32_compute else x
        if not self.fp32_compute:
            rotation = rotation.to(dtype)
        blocks = blocks.view(*x.shape[:-1], self.num_blocks, self.block_size)
        rotated = torch.einsum("...bi,bij->...bj", blocks, rotation)
        return rotated.reshape_as(x).to(dtype)


class OFTLinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, block_size: int, *, fp32_compute: bool = False) -> None:
        super().__init__()
        # The owning wrapper already registers the original SwiGLU as
        # `base_mlp`. Keep this as a non-registering reference so exported
        # state dicts contain one canonical copy of each frozen base weight.
        object.__setattr__(self, "base_linear", base_linear)
        self.oft_like = OFTRotation(
            base_linear.in_features,
            block_size,
            fp32_compute=fp32_compute,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_linear(self.oft_like(x))


class QwenSwiGLUOFTWrapper(nn.Module):
    def __init__(self, base_mlp: nn.Module, params: dict[str, Any]) -> None:
        super().__init__()
        self.base_mlp = base_mlp
        block_size = int(params.get("block_size", 64))
        self.fp32_compute = bool(params.get("fp32_compute", False))
        self.custom_ffn_layer_index: int | None = None
        self.last_inference_backend = "uninitialized"
        self.oft_gate = OFTLinear(base_mlp.gate_proj, block_size, fp32_compute=self.fp32_compute)
        self.oft_up = OFTLinear(base_mlp.up_proj, block_size, fp32_compute=self.fp32_compute)
        self.oft_down = OFTLinear(base_mlp.down_proj, block_size, fp32_compute=self.fp32_compute)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = _act(self.base_mlp, self.oft_gate(x)) * self.oft_up(x)
        output = self.oft_down(hidden)
        self.last_inference_backend = "reference_pytorch_cublas"
        write_dispatch_receipt_once(
            self,
            env_var="OFT_DISPATCH_RECEIPT",
            variant="qwen_swiglu_oft",
            backend=self.last_inference_backend,
            payload={
                "layer_index": self.custom_ffn_layer_index,
                "block_size": int(self.oft_gate.oft_like.block_size),
                "fp32_compute": self.fp32_compute,
                "parameter_devices": sorted(
                    {
                        str(parameter.device)
                        for module in (self.oft_gate, self.oft_up, self.oft_down)
                        for parameter in module.oft_like.parameters()
                    }
                ),
                "parameter_dtypes": sorted(
                    {
                        str(parameter.dtype)
                        for module in (self.oft_gate, self.oft_up, self.oft_down)
                        for parameter in module.oft_like.parameters()
                    }
                ),
            },
        )
        return output


def _replace_mlp(model: Any, layer_indices: list[int], build_wrapper: Any, manifest: dict[str, Any]) -> Any:
    layers = _get_layers(model)
    applied = []
    for idx in layer_indices:
        layer = layers[int(idx)]
        layer.mlp = build_wrapper(layer.mlp)
        if hasattr(layer.mlp, "custom_ffn_layer_index"):
            layer.mlp.custom_ffn_layer_index = int(idx)
        applied.append(int(idx))
    setattr(model, "_qwen_single_layer_rl_variant_manifest", {**manifest, "applied_layers": applied})
    return model


def inject_qwen_swiglu_shs(model: Any, layer_indices: list[int], params: dict[str, Any]) -> Any:
    d_model, _ = _model_dims(model, params)
    return _replace_mlp(
        model,
        layer_indices,
        lambda mlp: QwenSwiGLUSHSWrapper(mlp, d_model, params),
        {"variant": "qwen_swiglu_shs", "params": dict(params), "initial_effect": "exact_zero_delta"},
    )


def inject_qwen_swiglu_triglu_side(model: Any, layer_indices: list[int], params: dict[str, Any]) -> Any:
    d_model, intermediate = _model_dims(model, params)
    return _replace_mlp(
        model,
        layer_indices,
        lambda mlp: QwenSwiGLUTriGLUSideWrapper(mlp, d_model, intermediate, params),
        {"variant": "qwen_swiglu_triglu_side", "params": dict(params), "initial_effect": "exact_identity_multiplier"},
    )


def inject_qwen_swiglu_oft(model: Any, layer_indices: list[int], params: dict[str, Any]) -> Any:
    return _replace_mlp(
        model,
        layer_indices,
        lambda mlp: QwenSwiGLUOFTWrapper(mlp, params),
        {"variant": "qwen_swiglu_oft", "params": dict(params)},
    )
