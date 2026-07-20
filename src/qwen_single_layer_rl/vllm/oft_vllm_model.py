"""vLLM Transformers wrapper that verifies the OFT custom FFN runtime."""

from __future__ import annotations

import torch
from vllm.model_executor.models.transformers import TransformersForCausalLM

from qwen_single_layer_rl.model_surgery.qwen_swiglu_variant_modules import QwenSwiGLUOFTWrapper


class Qwen3OFTTransformersModel(TransformersForCausalLM):
    def __init__(self, *, vllm_config, prefix: str = "", **kwargs) -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix, **kwargs)
        self._oft_device = vllm_config.device_config.device
        params = dict(vllm_config.model_config.hf_config.oft_variant)
        expected_layers = [int(index) for index in params.get("target_layers", [10])]
        wrappers = [module for module in self.model.modules() if isinstance(module, QwenSwiGLUOFTWrapper)]
        if len(wrappers) != len(expected_layers):
            raise RuntimeError(f"expected {len(expected_layers)} OFT wrappers, found {len(wrappers)}")
        self.oft_wrapper_count = len(wrappers)
        self._finalize_oft_runtime()

    def _finalize_oft_runtime(self) -> None:
        for module in self.model.modules():
            if isinstance(module, QwenSwiGLUOFTWrapper):
                module.oft_gate.oft_like.to(device=self._oft_device, dtype=torch.float32)
                module.oft_up.oft_like.to(device=self._oft_device, dtype=torch.float32)
                module.oft_down.oft_like.to(device=self._oft_device, dtype=torch.float32)

    def load_weights(self, weights) -> set[str]:
        loaded = super().load_weights(weights)
        self._finalize_oft_runtime()
        return loaded

    def oft_dispatch_state(self) -> dict[str, object]:
        wrappers = [module for module in self.model.modules() if isinstance(module, QwenSwiGLUOFTWrapper)]
        parameters = [
            parameter
            for wrapper in wrappers
            for rotation in (wrapper.oft_gate, wrapper.oft_up, wrapper.oft_down)
            for parameter in rotation.oft_like.parameters()
        ]
        return {
            "wrapper_count": len(wrappers),
            "configured_backend": "reference_pytorch_cublas",
            "last_backends": [wrapper.last_inference_backend for wrapper in wrappers],
            "fp32_compute": [wrapper.fp32_compute for wrapper in wrappers],
            "parameter_devices": sorted({str(parameter.device) for parameter in parameters}),
            "parameter_dtypes": sorted({str(parameter.dtype) for parameter in parameters}),
            "cuda_parameters": all(parameter.is_cuda for parameter in parameters),
        }
