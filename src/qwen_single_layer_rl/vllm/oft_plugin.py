"""Lazy vLLM 0.10.x plugin registration for the OFT Transformers backend."""

from __future__ import annotations


def register() -> None:
    from vllm import ModelRegistry

    from . import OFT_ARCHITECTURE
    from .oft_hf_model import register_transformers_classes

    register_transformers_classes()
    ModelRegistry.register_model(
        OFT_ARCHITECTURE,
        "qwen_single_layer_rl.vllm.oft_vllm_model:Qwen3OFTTransformersModel",
    )
