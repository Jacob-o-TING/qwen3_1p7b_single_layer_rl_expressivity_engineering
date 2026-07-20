#!/usr/bin/env python3
"""Make veRL's saved SHS Hugging Face metadata self-contained for merging."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


AUTO_MAP = {
    "AutoConfig": "configuration_qwen3_shs.Qwen3SHSConfig",
    "AutoModel": "modeling_qwen3_shs.Qwen3SHSModel",
    "AutoModelForCausalLM": "modeling_qwen3_shs.Qwen3SHSForCausalLM",
}
CONFIGURATION_WRAPPER = "from qwen_single_layer_rl.vllm.shs_hf_model import Qwen3SHSConfig\n"
MODELING_WRAPPER = (
    "from qwen_single_layer_rl.vllm.shs_hf_model import "
    "Qwen3SHSForCausalLM, Qwen3SHSModel\n"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def repair(metadata_dir: Path, receipt_path: Path) -> dict:
    metadata_dir = metadata_dir.resolve()
    config_path = metadata_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_shs":
        raise ValueError(f"expected model_type=qwen3_shs, got {config.get('model_type')!r}")

    before_hash = sha256(config_path)
    backup_path = metadata_dir / "config.pre_shs_metadata_repair.json"
    if backup_path.exists() and sha256(backup_path) != before_hash:
        raise RuntimeError("existing metadata backup does not match current pre-repair config")
    if not backup_path.exists():
        shutil.copy2(config_path, backup_path)

    previous_auto_map = config.get("auto_map")
    config["auto_map"] = AUTO_MAP
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    configuration_path = metadata_dir / "configuration_qwen3_shs.py"
    modeling_path = metadata_dir / "modeling_qwen3_shs.py"
    configuration_path.write_text(CONFIGURATION_WRAPPER, encoding="utf-8")
    modeling_path.write_text(MODELING_WRAPPER, encoding="utf-8")

    receipt = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metadata_dir": str(metadata_dir),
        "model_type": config["model_type"],
        "previous_auto_map": previous_auto_map,
        "repaired_auto_map": AUTO_MAP,
        "hashes": {
            "config_before": before_hash,
            "config_after": sha256(config_path),
            "config_backup": sha256(backup_path),
            "configuration_wrapper": sha256(configuration_path),
            "modeling_wrapper": sha256(modeling_path),
        },
        "tensor_files_modified": False,
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(repair(args.metadata_dir, args.receipt), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
