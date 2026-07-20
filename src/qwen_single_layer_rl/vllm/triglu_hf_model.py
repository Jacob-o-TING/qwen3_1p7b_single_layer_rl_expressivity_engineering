"""Explicit Hugging Face TriGLU model contract for vLLM onboarding."""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM, Qwen3Model

from qwen_single_layer_rl.model_surgery.qwen_swiglu_variant_modules import QwenSwiGLUTriGLUSideWrapper

from . import TRIGLU_ARCHITECTURE
from .custom_ffn_contract import CustomFFNExportSpec


TRIGLU_EXPORT_SPEC = CustomFFNExportSpec(
    model_type="qwen3_triglu",
    architecture=TRIGLU_ARCHITECTURE,
    variant_config_key="triglu_variant",
    configuration_module="configuration_qwen3_triglu",
    configuration_class="Qwen3TriGLUConfig",
    modeling_module="modeling_qwen3_triglu",
    model_class="Qwen3TriGLUModel",
    causal_lm_class="Qwen3TriGLUForCausalLM",
)


class Qwen3TriGLUConfig(Qwen3Config):
    model_type = TRIGLU_EXPORT_SPEC.model_type

    def __init__(self, *, triglu_variant: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.triglu_variant = dict(triglu_variant or {})
        self.architectures = [TRIGLU_ARCHITECTURE]
        self.auto_map = TRIGLU_EXPORT_SPEC.auto_map


def _replace_triglu_mlp(model: Any, config: Qwen3TriGLUConfig) -> None:
    params = dict(config.triglu_variant)
    target_layers = [int(index) for index in params.pop("target_layers", [10])]
    for index in target_layers:
        base_mlp = model.layers[index].mlp
        wrapper = QwenSwiGLUTriGLUSideWrapper(
            base_mlp,
            int(config.hidden_size),
            int(config.intermediate_size),
            params,
        )
        # The training implementation computes the full side branch in FP32.
        wrapper.triglu_side.float()
        wrapper.custom_ffn_layer_index = index
        model.layers[index].mlp = wrapper


class Qwen3TriGLUModel(Qwen3Model):
    config_class = Qwen3TriGLUConfig

    def __init__(self, config: Qwen3TriGLUConfig) -> None:
        super().__init__(config)
        _replace_triglu_mlp(self, config)


class Qwen3TriGLUForCausalLM(Qwen3ForCausalLM):
    config_class = Qwen3TriGLUConfig

    def __init__(self, config: Qwen3TriGLUConfig) -> None:
        super().__init__(config)
        _replace_triglu_mlp(self.model, config)


def register_transformers_classes() -> None:
    AutoConfig.register(Qwen3TriGLUConfig.model_type, Qwen3TriGLUConfig, exist_ok=True)
    AutoModel.register(Qwen3TriGLUConfig, Qwen3TriGLUModel, exist_ok=True)
    AutoModelForCausalLM.register(Qwen3TriGLUConfig, Qwen3TriGLUForCausalLM, exist_ok=True)


def build_triglu_export_config(base_config: dict[str, Any], triglu_params: dict[str, Any]) -> dict[str, Any]:
    return TRIGLU_EXPORT_SPEC.build_config(base_config, triglu_params)


def expected_layer_key_prefixes(layer_index: int = 10) -> tuple[str, ...]:
    prefix = f"model.layers.{layer_index}.mlp"
    return (
        f"{prefix}.base_mlp.gate_proj.weight",
        f"{prefix}.base_mlp.up_proj.weight",
        f"{prefix}.base_mlp.down_proj.weight",
        f"{prefix}.triglu_side.down.weight",
        f"{prefix}.triglu_side.value.weight",
        f"{prefix}.triglu_side.gate.weight",
        f"{prefix}.triglu_side.ffn_1.weight",
        f"{prefix}.triglu_side.ffn_2.weight",
        f"{prefix}.triglu_side.up.weight",
    )
