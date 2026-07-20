from __future__ import annotations

from typing import Callable

from .base import ArchitectureVariant
from .identity import IdentityVariant
from .oft_like import OftLikeVariant
from .qwen_swiglu_variants import QwenSwiGLUOftVariant, QwenSwiGLUShsVariant, QwenSwiGLUTriGLUSideVariant

VariantFactory = Callable[[str, dict], ArchitectureVariant]
_REGISTRY: dict[str, VariantFactory] = {}


def register_variant(name: str, factory: VariantFactory) -> None:
    if not name:
        raise ValueError("Variant name cannot be empty")
    _REGISTRY[name] = factory


def build_variant(config: dict) -> ArchitectureVariant:
    spec = config.get("architecture_variant", {"name": "identity", "params": {}})
    name = spec.get("name", "identity")
    params = spec.get("params", {})
    try:
        factory = _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown architecture variant: {name}") from exc
    return factory(name, params)


def list_variants() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


register_variant("identity", lambda name, params: IdentityVariant(name=name, params=params))
register_variant("oft_like", lambda name, params: OftLikeVariant(name=name, params=params))
register_variant("qwen_swiglu_shs", lambda name, params: QwenSwiGLUShsVariant(name=name, params=params))
register_variant("qwen_swiglu_triglu_side", lambda name, params: QwenSwiGLUTriGLUSideVariant(name=name, params=params))
register_variant("qwen_swiglu_oft", lambda name, params: QwenSwiGLUOftVariant(name=name, params=params))
