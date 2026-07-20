"""Lazy vLLM 0.10.x plugin registration for the TriGLU Transformers backend."""

from __future__ import annotations


def register() -> None:
    from vllm import ModelRegistry

    from . import TRIGLU_ARCHITECTURE
    from .triglu_hf_model import register_transformers_classes

    register_transformers_classes()
    ModelRegistry.register_model(
        TRIGLU_ARCHITECTURE,
        "qwen_single_layer_rl.vllm.triglu_vllm_model:Qwen3TriGLUTransformersModel",
    )
