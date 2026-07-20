from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON/YAML config and resolve one-level `inherits` links."""
    cfg_path = Path(path)
    cfg = _read_structured(cfg_path)
    parent_ref = cfg.pop("inherits", None)
    if parent_ref:
        parent_path = (cfg_path.parent / parent_ref).resolve()
        parent = load_config(parent_path)
        return deep_merge(parent, cfg)
    return cfg


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def resolve_run_id(cfg: Mapping[str, Any]) -> str:
    experiment = cfg.get("experiment", {})
    arch = cfg.get("architecture_variant", {})
    logging = cfg.get("logging", {})
    template = logging.get("run_id_template", "{experiment_name}_{variant_name}_seed{seed}")
    return template.format(
        experiment_name=experiment.get("name", "experiment"),
        variant_name=arch.get("name", "identity"),
        seed=experiment.get("seed", "noseed"),
    )


def _read_structured(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except ModuleNotFoundError:
        data = _read_minimal_yaml(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _read_minimal_yaml(text: str) -> dict[str, Any]:
    """Tiny YAML subset parser for smoke configs when PyYAML is absent."""
    result: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, result)]
    pending_key: tuple[int, dict[str, Any], str] | None = None

    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if line.startswith("- "):
            value = _parse_scalar(line[2:].strip())
            if not isinstance(parent, list):
                if pending_key is None:
                    raise ValueError("List item without pending key")
                _, mapping, key = pending_key
                new_list: list[Any] = []
                mapping[key] = new_list
                stack.append((indent - 1, new_list))
                parent = new_list
            parent.append(value)
            continue

        key, sep, value_text = line.partition(":")
        if not sep:
            raise ValueError(f"Unsupported YAML line: {raw}")
        if value_text.strip():
            parent[key] = _parse_scalar(value_text.strip())
            pending_key = None
        else:
            child: dict[str, Any] = {}
            parent[key] = child
            pending_key = (indent, parent, key)
            stack.append((indent, child))

    return result


def _parse_scalar(value: str) -> Any:
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        return json.loads(value)
    try:
        if any(c in value for c in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"').strip("'")
