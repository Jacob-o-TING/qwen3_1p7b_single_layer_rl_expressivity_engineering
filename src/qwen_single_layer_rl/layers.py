from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class FreezeReport:
    backbone_train_mode: str
    train_adapter_modules: bool
    train_layers: tuple[int, ...]
    trainable_parameter_names: tuple[str, ...]
    frozen_parameter_names: tuple[str, ...]

    @property
    def trainable_count(self) -> int:
        return len(self.trainable_parameter_names)


def resolve_train_layers(config: dict[str, Any]) -> tuple[int, ...]:
    policy = config.get("freeze_policy", {})
    backbone_mode = resolve_backbone_train_mode(config)
    if backbone_mode == "frozen":
        return ()
    if backbone_mode == "full":
        num_layers = int(config.get("model", {}).get("num_layers", 0))
        return tuple(range(num_layers))

    if "train_layers" in policy:
        return tuple(int(x) for x in policy["train_layers"])

    set_name = policy.get("train_layers_from_set")
    if set_name:
        layer_sets = config.get("layer_sets", {})
        selected = layer_sets.get(set_name, {})
        return tuple(int(x) for x in selected.get("layers", []))

    mode = policy.get("mode", "single_layer")
    if mode == "full":
        num_layers = int(config.get("model", {}).get("num_layers", 0))
        return tuple(range(num_layers))
    raise ValueError("freeze_policy must define train_layers or train_layers_from_set")


def resolve_backbone_train_mode(config: dict[str, Any]) -> str:
    policy = config.get("freeze_policy", {})
    explicit = policy.get("backbone_train_mode")
    if explicit:
        if explicit not in {"frozen", "selected", "full"}:
            raise ValueError(f"Unsupported backbone_train_mode: {explicit}")
        return explicit

    legacy_mode = policy.get("mode", "single_layer")
    if legacy_mode in {"single_layer", "selected_layers"}:
        return "selected"
    if legacy_mode == "full":
        return "full"
    if legacy_mode == "adapter_only":
        return "frozen"
    raise ValueError(f"Unsupported freeze_policy mode: {legacy_mode}")


def resolve_train_adapter_modules(config: dict[str, Any]) -> bool:
    policy = config.get("freeze_policy", {})
    if "train_adapter_modules" in policy:
        return bool(policy["train_adapter_modules"])
    return bool(policy.get("allow_adapter_trainables", True))


def validate_layer_indices(train_layers: Iterable[int], num_layers: int) -> tuple[int, ...]:
    layers = tuple(int(x) for x in train_layers)
    bad = [x for x in layers if x < 0 or x >= num_layers]
    if bad:
        raise ValueError(f"Layer indices out of range for {num_layers} layers: {bad}")
    if len(set(layers)) != len(layers):
        raise ValueError(f"Duplicate layer indices: {layers}")
    return layers


def is_qwen_layer_parameter(name: str, layer_idx: int) -> bool:
    prefixes = (
        f"model.layers.{layer_idx}.",
        f"base_model.model.model.layers.{layer_idx}.",
        f"module.model.layers.{layer_idx}.",
    )
    return any(name.startswith(prefix) for prefix in prefixes)


def is_adapter_parameter(name: str, markers: Iterable[str]) -> bool:
    return any(marker in name for marker in markers)


def is_selected_mlp_base_parameter(name: str, train_layers: Iterable[int], module_names: Iterable[str]) -> bool:
    module_names = tuple(module_names)
    for idx in train_layers:
        prefixes = (
            f"model.layers.{idx}.mlp.",
            f"base_model.model.model.layers.{idx}.mlp.",
            f"module.model.layers.{idx}.mlp.",
        )
        for prefix in prefixes:
            if name.startswith(prefix) and any(module_name in name for module_name in module_names):
                return True
    return False


def apply_freeze_policy(model: Any, config: dict[str, Any]) -> FreezeReport:
    """Set requires_grad according to explicit backbone/adapter policy.

    The function is intentionally framework-light: any object with
    `named_parameters()` and parameter objects exposing `requires_grad` works.
    """
    num_layers = int(config.get("model", {}).get("num_layers", 28))
    train_layers = validate_layer_indices(resolve_train_layers(config), num_layers)
    policy = config.get("freeze_policy", {})
    backbone_train_mode = resolve_backbone_train_mode(config)
    train_adapter_modules = resolve_train_adapter_modules(config)
    adapter_markers = tuple(policy.get("adapter_name_markers", [".adapters.", ".adapter.", ".oft_like."]))
    freeze_selected_mlp_base = bool(policy.get("freeze_selected_mlp_base", False))
    selected_mlp_base_modules = tuple(
        policy.get(
            "selected_mlp_base_modules",
            ["gate_proj", "up_proj", "down_proj", "base_mlp.gate_proj", "base_mlp.up_proj", "base_mlp.down_proj"],
        )
    )

    trainable: list[str] = []
    frozen: list[str] = []
    for name, param in model.named_parameters():
        is_adapter = is_adapter_parameter(name, adapter_markers)
        if backbone_train_mode == "full":
            keep_trainable = not is_adapter
        else:
            keep_trainable = any(is_qwen_layer_parameter(name, idx) for idx in train_layers) and not is_adapter

        if train_adapter_modules and is_adapter:
            keep_trainable = True
        if freeze_selected_mlp_base and is_selected_mlp_base_parameter(name, train_layers, selected_mlp_base_modules):
            keep_trainable = False
        if policy.get("freeze_embeddings", True) and "embed_tokens" in name:
            keep_trainable = False
        if policy.get("freeze_lm_head", True) and "lm_head" in name:
            keep_trainable = False

        param.requires_grad = keep_trainable
        if keep_trainable:
            trainable.append(name)
        else:
            frozen.append(name)

    return FreezeReport(
        backbone_train_mode=backbone_train_mode,
        train_adapter_modules=train_adapter_modules,
        train_layers=train_layers,
        trainable_parameter_names=tuple(trainable),
        frozen_parameter_names=tuple(frozen),
    )
