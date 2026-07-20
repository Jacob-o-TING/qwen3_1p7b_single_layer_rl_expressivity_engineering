"""Explicit Hugging Face SHS model contract used by the vLLM scaffold."""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM, Qwen3Model

from qwen_single_layer_rl.model_surgery.qwen_swiglu_variant_modules import QwenSwiGLUSHSWrapper

from . import SHS_ARCHITECTURE


class Qwen3SHSConfig(Qwen3Config):
    model_type = "qwen3_shs"

    def __init__(self, *, shs_variant: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.shs_variant = dict(shs_variant or {})
        self.architectures = [SHS_ARCHITECTURE]
        self.auto_map = {
            "AutoConfig": "configuration_qwen3_shs.Qwen3SHSConfig",
            "AutoModel": "modeling_qwen3_shs.Qwen3SHSModel",
            "AutoModelForCausalLM": "modeling_qwen3_shs.Qwen3SHSForCausalLM",
        }


def _replace_shs_mlp(model, config: Qwen3SHSConfig) -> None:
    params = dict(config.shs_variant)
    target_layers = [int(index) for index in params.pop("target_layers", [10])]
    for index in target_layers:
        base_mlp = model.layers[index].mlp
        wrapper = QwenSwiGLUSHSWrapper(base_mlp, config.hidden_size, params)
        wrapper.shs.float()
        model.layers[index].mlp = wrapper


class Qwen3SHSModel(Qwen3Model):
    config_class = Qwen3SHSConfig
    _keys_to_ignore_on_load_missing = [r"model\.layers\.\d+\.mlp\.shs\..*_(?:row|col)_block_ids"]

    def __init__(self, config: Qwen3SHSConfig) -> None:
        super().__init__(config)
        _replace_shs_mlp(self, config)


class Qwen3SHSForCausalLM(Qwen3ForCausalLM):
    config_class = Qwen3SHSConfig
    _keys_to_ignore_on_load_missing = [r"model\.layers\.\d+\.mlp\.shs\..*_(?:row|col)_block_ids"]

    def __init__(self, config: Qwen3SHSConfig) -> None:
        super().__init__(config)
        _replace_shs_mlp(self.model, config)


def register_transformers_classes() -> None:
    """Register idempotently without importing vLLM or initializing CUDA."""
    AutoConfig.register(Qwen3SHSConfig.model_type, Qwen3SHSConfig, exist_ok=True)
    AutoModel.register(Qwen3SHSConfig, Qwen3SHSModel, exist_ok=True)
    AutoModelForCausalLM.register(Qwen3SHSConfig, Qwen3SHSForCausalLM, exist_ok=True)


def build_shs_export_config(base_config: dict[str, Any], shs_params: dict[str, Any]) -> dict[str, Any]:
    """Build config metadata without rewriting checkpoint tensors or keys."""
    exported = dict(base_config)
    exported["model_type"] = Qwen3SHSConfig.model_type
    exported["architectures"] = [SHS_ARCHITECTURE]
    exported["shs_variant"] = dict(shs_params)
    exported["auto_map"] = {
        "AutoConfig": "qwen_single_layer_rl.vllm.shs_hf_model.Qwen3SHSConfig",
        "AutoModel": "qwen_single_layer_rl.vllm.shs_hf_model.Qwen3SHSForCausalLM",
        "AutoModelForCausalLM": "qwen_single_layer_rl.vllm.shs_hf_model.Qwen3SHSForCausalLM",
    }
    return exported


def expected_layer10_key_prefixes(layer_index: int = 10) -> tuple[str, ...]:
    prefix = f"model.layers.{layer_index}.mlp"
    return (
        f"{prefix}.base_mlp.gate_proj.weight",
        f"{prefix}.base_mlp.up_proj.weight",
        f"{prefix}.base_mlp.down_proj.weight",
        f"{prefix}.shs.grid_generator.out.weight",
        f"{prefix}.shs.gate.mul_row_block_ids",
        f"{prefix}.shs.gate.add_row_block_ids",
    )
