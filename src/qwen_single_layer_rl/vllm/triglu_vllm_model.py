"""vLLM Transformers wrapper that verifies the TriGLU custom FFN runtime."""

from __future__ import annotations

import torch
from vllm.model_executor.models.transformers import TransformersForCausalLM

from qwen_single_layer_rl.model_surgery.qwen_swiglu_variant_modules import QwenSwiGLUTriGLUSideWrapper
from qwen_single_layer_rl.vllm.custom_ffn_contract import write_dispatch_receipt_once


class Qwen3TriGLUTransformersModel(TransformersForCausalLM):
    def __init__(self, *, vllm_config, prefix: str = "", **kwargs) -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix, **kwargs)
        self._triglu_device = vllm_config.device_config.device
        wrappers = [
            module for module in self.model.modules() if isinstance(module, QwenSwiGLUTriGLUSideWrapper)
        ]
        params = dict(vllm_config.model_config.hf_config.triglu_variant)
        expected_layers = [int(index) for index in params.get("target_layers", [10])]
        if len(wrappers) != len(expected_layers):
            raise RuntimeError(
                f"expected {len(expected_layers)} TriGLU wrappers, found {len(wrappers)}"
            )
        self.triglu_wrapper_count = len(wrappers)
        self._finalize_triglu_runtime()

    def _finalize_triglu_runtime(self) -> None:
        for module in self.model.modules():
            if isinstance(module, QwenSwiGLUTriGLUSideWrapper):
                module.triglu_side.to(device=self._triglu_device, dtype=torch.float32)

    def load_weights(self, weights) -> set[str]:
        loaded = super().load_weights(weights)
        # The Transformers converter installs vLLM Linear modules and loads
        # them under the global model dtype after __init__. Reapply the
        # architecture-owned FP32 side-branch policy after that load.
        self._finalize_triglu_runtime()
        # Emit the runtime receipt before vLLM compiles its dummy forward.
        # Filesystem side effects inside forward are unsupported by Dynamo.
        for module in self.model.modules():
            if isinstance(module, QwenSwiGLUTriGLUSideWrapper):
                module.last_inference_backend = "reference_pytorch_cublas"
                write_dispatch_receipt_once(
                    module,
                    env_var="TRIGLU_DISPATCH_RECEIPT",
                    variant="qwen_swiglu_triglu_side",
                    backend=module.last_inference_backend,
                    payload={
                        "layer_index": module.custom_ffn_layer_index,
                        "side_dim": int(module.triglu_side["down"].weight.shape[0]),
                        "side_hidden": int(module.triglu_side["value"].weight.shape[0]),
                        "parameter_devices": sorted(
                            {str(parameter.device) for parameter in module.triglu_side.parameters()}
                        ),
                        "parameter_dtypes": sorted(
                            {str(parameter.dtype) for parameter in module.triglu_side.parameters()}
                        ),
                    },
                )
        return loaded

    def triglu_dispatch_state(self) -> dict[str, object]:
        wrappers = [
            module for module in self.model.modules() if isinstance(module, QwenSwiGLUTriGLUSideWrapper)
        ]
        parameters = [parameter for wrapper in wrappers for parameter in wrapper.triglu_side.parameters()]
        return {
            "wrapper_count": len(wrappers),
            "configured_backend": "reference_pytorch_cublas",
            "last_backends": [wrapper.last_inference_backend for wrapper in wrappers],
            "parameter_devices": sorted({str(parameter.device) for parameter in parameters}),
            "parameter_dtypes": sorted({str(parameter.dtype) for parameter in parameters}),
            "cuda_parameters": all(parameter.is_cuda for parameter in parameters),
        }
