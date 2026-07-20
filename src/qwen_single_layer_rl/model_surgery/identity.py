from __future__ import annotations

from typing import Any

from .base import ArchitectureVariant


class IdentityVariant(ArchitectureVariant):
    def apply(self, model: Any, config: dict[str, Any]) -> Any:
        return model
