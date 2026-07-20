"""Lazy vLLM 0.10.x plugin registration for the SHS Transformers backend."""

from __future__ import annotations


def register() -> None:
    from vllm import ModelRegistry

    from . import SHS_ARCHITECTURE
    from .shs_hf_model import register_transformers_classes

    register_transformers_classes()
    ModelRegistry.register_model(
        SHS_ARCHITECTURE,
        "qwen_single_layer_rl.vllm.shs_vllm_model:Qwen3SHSTransformersModel",
    )
