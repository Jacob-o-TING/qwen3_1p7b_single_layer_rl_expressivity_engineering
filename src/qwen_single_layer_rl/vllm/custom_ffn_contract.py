"""Architecture-neutral metadata and dispatch helpers for custom Qwen FFNs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CustomFFNExportSpec:
    model_type: str
    architecture: str
    variant_config_key: str
    configuration_module: str
    configuration_class: str
    modeling_module: str
    model_class: str
    causal_lm_class: str

    @property
    def auto_map(self) -> dict[str, str]:
        return {
            "AutoConfig": f"{self.configuration_module}.{self.configuration_class}",
            "AutoModel": f"{self.modeling_module}.{self.model_class}",
            "AutoModelForCausalLM": f"{self.modeling_module}.{self.causal_lm_class}",
        }

    def build_config(self, base_config: dict[str, Any], variant_params: dict[str, Any]) -> dict[str, Any]:
        exported = dict(base_config)
        exported["model_type"] = self.model_type
        exported["architectures"] = [self.architecture]
        exported[self.variant_config_key] = dict(variant_params)
        exported["auto_map"] = self.auto_map
        return exported


def write_dispatch_receipt_once(
    module: Any,
    *,
    env_var: str,
    variant: str,
    backend: str,
    payload: dict[str, Any] | None = None,
) -> None:
    """Append one semantic runtime receipt per module when explicitly requested."""
    receipt_path = os.environ.get(env_var)
    if not receipt_path or getattr(module, "_custom_ffn_dispatch_receipt_written", False):
        return
    path = Path(receipt_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "pid": os.getpid(),
        "variant": variant,
        "backend": backend,
        "fallback": False,
        **(payload or {}),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    module._custom_ffn_dispatch_receipt_written = True


def archive_incomplete_dispatch_receipt(path: Path) -> Path | None:
    """Preserve a failed attempt's receipt before a resumable worker retry."""
    if not path.exists():
        return None
    attempt = 1
    while True:
        archived = path.with_name(f"{path.stem}.attempt-{attempt:02d}{path.suffix}")
        if not archived.exists():
            path.replace(archived)
            return archived
        attempt += 1


def validate_dispatch_receipts(
    path: Path,
    *,
    variant: str,
    backend: str,
    expected_count: int,
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"custom-FFN dispatch receipt was not created: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != expected_count:
        raise RuntimeError(f"expected {expected_count} dispatch receipts, found {len(rows)}")
    for row in rows:
        if row.get("variant") != variant or row.get("backend") != backend or row.get("fallback") is not False:
            raise RuntimeError(f"invalid custom-FFN dispatch receipt: {row}")
    return rows
