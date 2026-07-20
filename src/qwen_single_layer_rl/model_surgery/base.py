from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ArchitectureVariant:
    """Base class for model surgery / adapter variants."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def apply(self, model: Any, config: dict[str, Any]) -> Any:
        raise NotImplementedError

    def trainable_name_hints(self) -> tuple[str, ...]:
        return ()
