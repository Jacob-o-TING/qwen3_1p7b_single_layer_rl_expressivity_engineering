from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from qwen_single_layer_rl.config import load_config, resolve_run_id
from qwen_single_layer_rl.layers import apply_freeze_policy
from qwen_single_layer_rl.model_surgery import build_variant
from qwen_single_layer_rl.seeding import seed_everything
from qwen_single_layer_rl.sft.checkpoint import load_trainable_state_dict


_DETERMINISTIC_SHS_BUFFER_NAMES = (
    "mul_row_block_ids",
    "mul_col_block_ids",
    "add_row_block_ids",
    "add_col_block_ids",
)


def _target_layers(cfg: dict[str, Any]) -> list[int]:
    params = cfg.get("architecture_variant", {}).get("params", {})
    fallback = cfg.get("freeze_policy", {}).get("train_layers", [])
    return [int(index) for index in params.get("target_layers", fallback)]


def _model_layers(model: Any) -> Any:
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise RuntimeError("Cannot locate transformer layers for veRL model surgery")
    return layers


def _classify_shs_construction(model: Any, cfg: dict[str, Any]) -> str:
    targets = _target_layers(cfg)
    if not targets:
        raise RuntimeError("SHS configuration has no target layers")
    layers = _model_layers(model)
    present = [
        hasattr(layers[index].mlp, "shs") and hasattr(layers[index].mlp, "base_mlp")
        for index in targets
    ]
    if all(present):
        return "preconstructed"
    if any(present):
        raise RuntimeError("Partial SHS construction detected; refusing ambiguous injection")
    return "inject"


def _classify_triglu_construction(model: Any, cfg: dict[str, Any]) -> str:
    targets = _target_layers(cfg)
    if not targets:
        raise RuntimeError("TriGLU configuration has no target layers")
    layers = _model_layers(model)
    present = [
        hasattr(layers[index].mlp, "triglu_side") and hasattr(layers[index].mlp, "base_mlp")
        for index in targets
    ]
    if all(present):
        return "preconstructed"
    if any(present):
        raise RuntimeError("Partial TriGLU construction detected; refusing ambiguous injection")
    return "inject"


def _classify_oft_construction(model: Any, cfg: dict[str, Any]) -> str:
    targets = _target_layers(cfg)
    if not targets:
        raise RuntimeError("OFT configuration has no target layers")
    layers = _model_layers(model)
    present = [
        all(hasattr(layers[index].mlp, name) for name in ("oft_gate", "oft_up", "oft_down", "base_mlp"))
        for index in targets
    ]
    if all(present):
        return "preconstructed"
    if any(present):
        raise RuntimeError("Partial OFT construction detected; refusing ambiguous injection")
    return "inject"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _exclude_deterministic_shs_buffers_from_state_dict(model: Any, target_layer_count: int) -> int:
    excluded = 0
    for module in model.modules():
        present = [name for name in _DETERMINISTIC_SHS_BUFFER_NAMES if name in module._buffers]
        if not present:
            continue
        if len(present) != len(_DETERMINISTIC_SHS_BUFFER_NAMES):
            raise RuntimeError(f"Incomplete deterministic SHS buffer set: {present}")
        module._non_persistent_buffers_set.update(present)
        excluded += len(present)
    expected = target_layer_count * 3 * len(_DETERMINISTIC_SHS_BUFFER_NAMES)
    if excluded != expected:
        raise RuntimeError(f"Expected {expected} deterministic SHS buffers, found {excluded}")
    return excluded


def apply_model_surgery_before_fsdp(model: Any, role: str = "actor", rank: int = 0) -> Any:
    config_path = os.environ.get("QWEN_SINGLE_LAYER_RL_CONFIG")
    if not config_path:
        return model

    cfg = load_config(config_path)
    if getattr(model, "_qwen_single_layer_hook_applied", False):
        raise RuntimeError("veRL model hook was invoked twice on the same model instance")
    seed = int(cfg.get("experiment", {}).get("init_seed", cfg.get("experiment", {}).get("seed", 0)))
    seed_everything(seed)

    variant = build_variant(cfg)
    construction_mode = "identity"
    if variant.name == "qwen_swiglu_shs":
        construction_mode = _classify_shs_construction(model, cfg)
    elif variant.name == "qwen_swiglu_triglu_side":
        construction_mode = _classify_triglu_construction(model, cfg)
    elif variant.name == "qwen_swiglu_oft":
        construction_mode = _classify_oft_construction(model, cfg)
    if construction_mode != "preconstructed":
        model = variant.apply(model, cfg)
    if variant.name == "qwen_swiglu_oft":
        for index in _target_layers(cfg):
            wrapper = _model_layers(model)[index].mlp
            for rotation in (wrapper.oft_gate, wrapper.oft_up, wrapper.oft_down):
                rotation.oft_like.float()

    checkpoint_dir_value = os.environ.get("QWEN_SINGLE_LAYER_RL_CHECKPOINT_DIR")
    initialization_contract = os.environ.get("QWEN_SINGLE_LAYER_RL_INITIALIZATION", "checkpoint_overlay")
    allows_exact_noop_init = variant.name in {
        "qwen_swiglu_triglu_side",
        "qwen_swiglu_oft",
    } and initialization_contract == "untuned_base_exact_noop"
    if variant.name != "identity" and not checkpoint_dir_value and not allows_exact_noop_init:
        raise RuntimeError(
            "QWEN_SINGLE_LAYER_RL_CHECKPOINT_DIR is required for variant actor construction "
            "unless the exact-noop TriGLU/OFT path explicitly selects untuned_base_exact_noop"
        )
    checkpoint_path: Path | None = None
    checkpoint_hash: str | None = None
    overlay_keys: list[str] = []
    if checkpoint_dir_value:
        import torch

        checkpoint_path = Path(checkpoint_dir_value).resolve() / "trainable_state.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Completed trainable checkpoint is missing: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(state, dict):
            raise TypeError("trainable_state.pt must contain a parameter mapping")
        overlay_keys = sorted(str(name) for name in state)
        load_trainable_state_dict(model, state)
        checkpoint_hash = _sha256(checkpoint_path)

    actor_freeze_report = apply_freeze_policy(model, cfg)
    expected_trainable = set(actor_freeze_report.trainable_parameter_names)
    if overlay_keys and set(overlay_keys) != expected_trainable:
        missing = sorted(expected_trainable - set(overlay_keys))
        unexpected = sorted(set(overlay_keys) - expected_trainable)
        raise RuntimeError(
            f"Checkpoint/freeze key mismatch: missing={missing[:20]}, unexpected={unexpected[:20]}"
        )
    excluded_sync_buffer_count = 0
    if variant.name == "qwen_swiglu_shs":
        excluded_sync_buffer_count = _exclude_deterministic_shs_buffers_from_state_dict(
            model, len(_target_layers(cfg))
        )
    if role != "actor":
        for _, param in model.named_parameters():
            param.requires_grad = False
    model._qwen_single_layer_hook_applied = True
    model._qwen_single_layer_checkpoint_overlay_count = 1 if checkpoint_path else 0

    audit_dir = os.environ.get("QWEN_SINGLE_LAYER_RL_AUDIT_DIR")
    if audit_dir:
        _write_audit(
            Path(audit_dir),
            cfg,
            variant.name,
            role,
            rank,
            actor_freeze_report,
            construction_mode=construction_mode,
            initialization_contract=initialization_contract,
            checkpoint_path=checkpoint_path,
            checkpoint_hash=checkpoint_hash,
            overlay_keys=overlay_keys,
            excluded_sync_buffer_count=excluded_sync_buffer_count,
            actual_trainable=[name for name, parameter in model.named_parameters() if parameter.requires_grad],
        )
    return model


def _write_audit(
    audit_dir: Path,
    cfg: dict[str, Any],
    variant_name: str,
    role: str,
    rank: int,
    freeze_report: Any,
    *,
    construction_mode: str,
    initialization_contract: str,
    checkpoint_path: Path | None,
    checkpoint_hash: str | None,
    overlay_keys: list[str],
    excluded_sync_buffer_count: int,
    actual_trainable: list[str],
) -> None:
    audit_dir.mkdir(parents=True, exist_ok=True)
    trainable = list(freeze_report.trainable_parameter_names)
    frozen = list(freeze_report.frozen_parameter_names)
    payload = {
        "run_id": resolve_run_id(cfg),
        "role": role,
        "rank": int(rank),
        "variant": variant_name,
        "construction_mode": construction_mode,
        "initialization_contract": initialization_contract,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_overlay_count": 1 if checkpoint_path else 0,
        "checkpoint_overlay_key_count": len(overlay_keys),
        "checkpoint_overlay_keys": overlay_keys,
        "excluded_deterministic_sync_buffer_count": excluded_sync_buffer_count,
        "train_layers": list(freeze_report.train_layers),
        "backbone_train_mode": freeze_report.backbone_train_mode,
        "train_adapter_modules": freeze_report.train_adapter_modules,
        "trainable_parameter_count": len(trainable),
        "frozen_parameter_count": len(frozen),
        "trainable_parameter_names": trainable,
        "actual_trainable_parameter_count": len(actual_trainable),
        "actual_trainable_parameter_names": sorted(actual_trainable),
        "frozen_parameter_names_sample": frozen[:200],
    }
    (audit_dir / f"{role}_rank{rank}_model_surgery_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
