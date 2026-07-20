from __future__ import annotations

import argparse
import json
from pathlib import Path

from qwen_single_layer_rl.config import load_config, resolve_run_id
from qwen_single_layer_rl.layers import (
    resolve_backbone_train_mode,
    resolve_train_adapter_modules,
    resolve_train_layers,
    validate_layer_indices,
)
from qwen_single_layer_rl.model_surgery import build_variant, list_variants
from qwen_single_layer_rl.seeding import seed_everything
from qwen_single_layer_rl.training.verl_bridge import render_verl_plan


def build_dry_run_manifest(config_path: Path, out_dir: Path) -> dict:
    cfg = load_config(config_path)
    seed_report = seed_everything(int(cfg.get("experiment", {}).get("seed", 0)))
    layers = validate_layer_indices(resolve_train_layers(cfg), int(cfg.get("model", {}).get("num_layers", 28)))
    variant = build_variant(cfg)
    run_id = resolve_run_id(cfg)
    return {
        "config_path": str(config_path),
        "run_id": run_id,
        "out_dir": str(out_dir),
        "seed_report": seed_report.__dict__,
        "backbone_train_mode": resolve_backbone_train_mode(cfg),
        "train_adapter_modules": resolve_train_adapter_modules(cfg),
        "train_layers": list(layers),
        "architecture_variant": variant.name,
        "available_variants": list(list_variants()),
        "verl_plan": render_verl_plan(cfg),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    manifest = build_dry_run_manifest(args.config, args.out)

    (args.out / "dry_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.out / "resolved_config.json").write_text(
        json.dumps(cfg, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.out / "planned_trainables.txt").write_text(
        "\n".join(f"model.layers.{idx}.*" for idx in manifest["train_layers"]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
