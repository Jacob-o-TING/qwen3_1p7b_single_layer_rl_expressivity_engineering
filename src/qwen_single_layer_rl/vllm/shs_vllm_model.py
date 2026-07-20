"""vLLM Transformers wrapper that finalizes SHS inference state."""

from __future__ import annotations

import os
import torch
from vllm.model_executor.models.transformers import TransformersModel

from qwen_single_layer_rl.model_surgery.qwen_swiglu_variant_modules import ShuffledHyperGridDeltaLinear


class Qwen3SHSTransformersModel(TransformersModel):
    """Move custom buffers and require an explicit SHS projection backend."""

    def __init__(self, *, vllm_config, prefix: str = "", **kwargs) -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix, **kwargs)
        requested_backend = os.environ.get("SHS_INFERENCE_MUL_BACKEND")
        if requested_backend not in ("reference", "triton", "triton_fp32"):
            raise RuntimeError(
                "SHS_INFERENCE_MUL_BACKEND must explicitly select reference, triton, or triton_fp32"
            )
        device = vllm_config.device_config.device
        projection_count = 0
        layers = getattr(self.model, "layers", None)
        if layers is None:
            layers = self.model.model.layers
        for layer in layers:
            mlp = layer.mlp
            if not hasattr(mlp, "shs"):
                continue
            mlp.shs["gate"].rebind_base_linear(mlp.base_mlp.gate_proj)
            mlp.shs["up"].rebind_base_linear(mlp.base_mlp.up_proj)
            mlp.shs["down"].rebind_base_linear(mlp.base_mlp.down_proj)
        for module in self.model.modules():
            if not isinstance(module, ShuffledHyperGridDeltaLinear):
                continue
            for name, buffer in tuple(module.named_buffers(recurse=False)):
                setattr(module, name, buffer.to(device=device, non_blocking=True))
            module.set_inference_mul_backend(requested_backend)
            projection_count += 1
        if projection_count != 3:
            raise RuntimeError(f"expected 3 Layer-10 SHS projections, found {projection_count}")
        self.shs_projection_count = projection_count

    def shs_dispatch_state(self) -> dict[str, object]:
        projections = [
            module
            for module in self.model.modules()
            if isinstance(module, ShuffledHyperGridDeltaLinear)
        ]
        return {
            "projection_count": len(projections),
            "configured_backends": [module.inference_mul_backend for module in projections],
            "last_backends": [module.last_inference_mul_backend for module in projections],
            "buffer_devices": sorted({str(buffer.device) for module in projections for buffer in module.buffers()}),
            "cuda_buffers": all(buffer.is_cuda for module in projections for buffer in module.buffers()),
        }
