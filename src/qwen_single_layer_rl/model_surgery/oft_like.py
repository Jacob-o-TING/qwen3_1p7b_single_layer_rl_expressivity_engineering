from __future__ import annotations

from typing import Any

from .base import ArchitectureVariant


class OftLikeVariant(ArchitectureVariant):
    """Placeholder for OFT-style trainable transformations.

    This class records intent without importing torch. Replace `apply` with
    module injection once the exact transform, parameterization, and veRL/FSDP
    wrapping policy are chosen.
    """

    def apply(self, model: Any, config: dict[str, Any]) -> Any:
        manifest = {
            "variant": self.name,
            "target_modules": self.params.get("target_modules", []),
            "rank": self.params.get("rank"),
            "block_size": self.params.get("block_size"),
            "init": self.params.get("init", "identity"),
            "status": "planned_not_injected",
        }
        setattr(model, "_qwen_single_layer_rl_variant_manifest", manifest)
        return model

    def trainable_name_hints(self) -> tuple[str, ...]:
        return (".adapters.", ".oft_like.")
