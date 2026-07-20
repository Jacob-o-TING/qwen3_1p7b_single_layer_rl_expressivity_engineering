from __future__ import annotations

from typing import Any

from .base import ArchitectureVariant


def _target_layers(config: dict[str, Any], params: dict[str, Any]) -> list[int]:
    if "target_layers" in params:
        return [int(x) for x in params["target_layers"]]
    policy = config.get("freeze_policy", {})
    return [int(x) for x in policy.get("train_layers", [])]


class QwenSwiGLUShsVariant(ArchitectureVariant):
    """Inject SHS HyperGrid modulation into selected Qwen SwiGLU MLPs."""

    def apply(self, model: Any, config: dict[str, Any]) -> Any:
        from .qwen_swiglu_variant_modules import inject_qwen_swiglu_shs

        return inject_qwen_swiglu_shs(model, _target_layers(config, self.params), self.params)

    def trainable_name_hints(self) -> tuple[str, ...]:
        return (".shs.", ".grid_generator.")


class QwenSwiGLUTriGLUSideVariant(ArchitectureVariant):
    """Inject a residual-delta TriGLU side FFN multiplier into selected Qwen MLPs."""

    def apply(self, model: Any, config: dict[str, Any]) -> Any:
        from .qwen_swiglu_variant_modules import inject_qwen_swiglu_triglu_side

        return inject_qwen_swiglu_triglu_side(model, _target_layers(config, self.params), self.params)

    def trainable_name_hints(self) -> tuple[str, ...]:
        return (".triglu_side.",)


class QwenSwiGLUOftVariant(ArchitectureVariant):
    """Wrap selected Qwen SwiGLU projections with OFT input rotations."""

    def apply(self, model: Any, config: dict[str, Any]) -> Any:
        from .qwen_swiglu_variant_modules import inject_qwen_swiglu_oft

        return inject_qwen_swiglu_oft(model, _target_layers(config, self.params), self.params)

    def trainable_name_hints(self) -> tuple[str, ...]:
        return (".oft_like", ".oft_gate.", ".oft_up.", ".oft_down.")
