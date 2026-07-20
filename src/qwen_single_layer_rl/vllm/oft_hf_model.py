"""Explicit Hugging Face OFT model contract for vLLM onboarding."""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM, Qwen3Model

from qwen_single_layer_rl.model_surgery.qwen_swiglu_variant_modules import QwenSwiGLUOFTWrapper

from . import OFT_ARCHITECTURE
from .custom_ffn_contract import CustomFFNExportSpec


OFT_EXPORT_SPEC = CustomFFNExportSpec(
    model_type="qwen3_oft",
    architecture=OFT_ARCHITECTURE,
    variant_config_key="oft_variant",
    configuration_module="configuration_qwen3_oft",
    configuration_class="Qwen3OFTConfig",
    modeling_module="modeling_qwen3_oft",
    model_class="Qwen3OFTModel",
    causal_lm_class="Qwen3OFTForCausalLM",
)


class Qwen3OFTConfig(Qwen3Config):
    model_type = OFT_EXPORT_SPEC.model_type

    def __init__(self, *, oft_variant: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.oft_variant = dict(oft_variant or {})
        self.architectures = [OFT_ARCHITECTURE]
        self.auto_map = OFT_EXPORT_SPEC.auto_map


def _replace_oft_mlp(model: Any, config: Qwen3OFTConfig) -> None:
    params = dict(config.oft_variant)
    target_layers = [int(index) for index in params.pop("target_layers", [10])]
    for index in target_layers:
        wrapper = QwenSwiGLUOFTWrapper(model.layers[index].mlp, params)
        wrapper.custom_ffn_layer_index = index
        wrapper.oft_gate.oft_like.float()
        wrapper.oft_up.oft_like.float()
        wrapper.oft_down.oft_like.float()
        model.layers[index].mlp = wrapper


def _finalize_oft_precision(model: Any) -> None:
    for module in model.modules():
        if isinstance(module, QwenSwiGLUOFTWrapper):
            module.oft_gate.oft_like.float()
            module.oft_up.oft_like.float()
            module.oft_down.oft_like.float()


class _OFTFromPretrainedMixin:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Any, *model_args: Any, **kwargs: Any) -> Any:
        result = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model = result[0] if isinstance(result, tuple) else result
        _finalize_oft_precision(model)
        return result


class Qwen3OFTModel(_OFTFromPretrainedMixin, Qwen3Model):
    config_class = Qwen3OFTConfig

    def __init__(self, config: Qwen3OFTConfig) -> None:
        super().__init__(config)
        _replace_oft_mlp(self, config)


class Qwen3OFTForCausalLM(_OFTFromPretrainedMixin, Qwen3ForCausalLM):
    config_class = Qwen3OFTConfig

    def __init__(self, config: Qwen3OFTConfig) -> None:
        super().__init__(config)
        _replace_oft_mlp(self.model, config)


def register_transformers_classes() -> None:
    AutoConfig.register(Qwen3OFTConfig.model_type, Qwen3OFTConfig, exist_ok=True)
    AutoModel.register(Qwen3OFTConfig, Qwen3OFTModel, exist_ok=True)
    AutoModelForCausalLM.register(Qwen3OFTConfig, Qwen3OFTForCausalLM, exist_ok=True)


def build_oft_export_config(base_config: dict[str, Any], oft_params: dict[str, Any]) -> dict[str, Any]:
    return OFT_EXPORT_SPEC.build_config(base_config, oft_params)


def expected_layer_key_prefixes(layer_index: int = 10) -> tuple[str, ...]:
    prefix = f"model.layers.{layer_index}.mlp"
    return (
        f"{prefix}.base_mlp.gate_proj.weight",
        f"{prefix}.base_mlp.up_proj.weight",
        f"{prefix}.base_mlp.down_proj.weight",
        f"{prefix}.oft_gate.oft_like.oft_like",
        f"{prefix}.oft_up.oft_like.oft_like",
        f"{prefix}.oft_down.oft_like.oft_like",
    )
