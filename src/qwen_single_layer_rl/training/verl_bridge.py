from __future__ import annotations

import shlex
from typing import Any

from qwen_single_layer_rl.layers import (
    resolve_backbone_train_mode,
    resolve_train_adapter_modules,
    resolve_train_layers,
)


def render_verl_plan(config: dict[str, Any]) -> dict[str, Any]:
    experiment = config.get("experiment", {})
    model = config.get("model", {})
    grpo = config.get("grpo", {})
    runtime = config.get("runtime", {})
    arch = config.get("architecture_variant", {})
    freeze_policy = config.get("freeze_policy", {})
    layers = resolve_train_layers(config)

    return {
        "note": "Placeholder plan. Map these fields to your installed veRL revision.",
        "experiment_name": experiment.get("name"),
        "model_path": model.get("name_or_path"),
        "algorithm": grpo.get("algorithm", "GRPO"),
        "learning_rate": grpo.get("learning_rate"),
        "backbone_train_mode": resolve_backbone_train_mode(config),
        "train_adapter_modules": resolve_train_adapter_modules(config),
        "train_layers": list(layers),
        "architecture_variant": arch.get("name", "identity"),
        "architecture_params": arch.get("params", {}),
        "freeze_selected_mlp_base": bool(freeze_policy.get("freeze_selected_mlp_base", False)),
        "nproc_per_node": runtime.get("nproc_per_node", 1),
        "suggested_env": {
            "QWEN_SINGLE_LAYER_TRAIN_LAYERS": ",".join(str(x) for x in layers),
            "QWEN_BACKBONE_TRAIN_MODE": resolve_backbone_train_mode(config),
            "QWEN_TRAIN_ADAPTER_MODULES": str(resolve_train_adapter_modules(config)).lower(),
            "QWEN_ARCH_VARIANT": arch.get("name", "identity"),
            "QWEN_FREEZE_SELECTED_MLP_BASE": str(bool(freeze_policy.get("freeze_selected_mlp_base", False))).lower(),
        },
    }


def render_torchrun_command(config_path: str, config: dict[str, Any]) -> str:
    runtime = config.get("runtime", {})
    nproc = int(runtime.get("nproc_per_node", 1))
    return " ".join(
        [
            "torchrun",
            f"--nproc_per_node={nproc}",
            "-m",
            "qwen_single_layer_rl.training.verl_entrypoint_placeholder",
            "--config",
            shlex.quote(config_path),
        ]
    )
