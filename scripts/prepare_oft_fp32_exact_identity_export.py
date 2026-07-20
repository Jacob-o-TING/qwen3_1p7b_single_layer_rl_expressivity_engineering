from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from qwen_single_layer_rl.config import load_config
from qwen_single_layer_rl.model_surgery.qwen_swiglu_variant_modules import QwenSwiGLUOFTWrapper
from qwen_single_layer_rl.vllm.oft_hf_model import Qwen3OFTConfig


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_layers(model):
    return model.model.layers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()

    base_model = args.base_model.resolve()
    output = args.output.resolve()
    project_root = args.project_root.resolve()
    allowed_root = (project_root / "runs" / "runtime_models").resolve()
    if not output.is_relative_to(allowed_root):
        raise RuntimeError(f"output must stay under {allowed_root}: {output}")
    complete = output / "EXPORT_COMPLETE"
    manifest_path = output / "oft_exact_identity_export_manifest.json"
    if complete.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "PASS":
            raise RuntimeError(f"existing export manifest is not PASS: {manifest_path}")
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    cfg = load_config(args.config)
    params = dict(cfg["architecture_variant"]["params"])
    if cfg["architecture_variant"]["name"] != "qwen_swiglu_oft":
        raise RuntimeError("config is not qwen_swiglu_oft")
    if not params.get("fp32_compute"):
        raise RuntimeError("the production OFT control must set fp32_compute=true")
    target_layers = [int(index) for index in params.get("target_layers", [10])]
    if target_layers != [10]:
        raise RuntimeError(f"expected only Layer 10, got {target_layers}")
    target_modules = list(params.get("target_modules", []))
    expected_modules = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    if target_modules != expected_modules:
        raise RuntimeError(f"OFT targets must be SwiGLU-only: {target_modules}")

    staging = output.with_name(f".{output.name}.staging-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    layer = model_layers(model)[10]
    base_mlp = layer.mlp
    generator = torch.Generator(device="cpu").manual_seed(20260707)
    probe = torch.randn(1, 4, int(model.config.hidden_size), generator=generator, dtype=torch.bfloat16)
    with torch.inference_mode():
        expected = base_mlp(probe)
    wrapper = QwenSwiGLUOFTWrapper(base_mlp, params)
    wrapper.custom_ffn_layer_index = 10
    layer.mlp = wrapper
    with torch.inference_mode():
        observed = wrapper(probe)
    if not torch.equal(expected, observed):
        delta = float((expected.float() - observed.float()).abs().max())
        raise RuntimeError(f"OFT exact-identity component parity failed: max_abs={delta}")

    oft_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if ".oft_" in name and ".oft_like.oft_like" in name
    }
    if len(oft_parameters) != 3:
        raise RuntimeError(f"expected three OFT parameters, found {sorted(oft_parameters)}")
    if {parameter.dtype for parameter in oft_parameters.values()} != {torch.float32}:
        raise RuntimeError("OFT parameters are not exclusively FP32")
    if any("self_attn" in name for name in oft_parameters):
        raise RuntimeError("attention OFT parameter detected")

    raw_config = model.config.to_dict()
    for key in ("model_type", "architectures", "auto_map", "oft_variant"):
        raw_config.pop(key, None)
    custom_config = Qwen3OFTConfig(oft_variant=params, **raw_config)
    model.config = custom_config
    model.save_pretrained(staging, safe_serialization=True, max_shard_size="2GB")
    custom_config.save_pretrained(staging)

    weight_names = {path.name for path in base_model.glob("*.safetensors")}
    for path in base_model.iterdir():
        if not path.is_file() or path.name == "config.json" or path.name in weight_names:
            continue
        destination = staging / path.name
        if not destination.exists():
            shutil.copy2(path, destination)
    (staging / "configuration_qwen3_oft.py").write_text(
        "from qwen_single_layer_rl.vllm.oft_hf_model import Qwen3OFTConfig\n",
        encoding="utf-8",
    )
    (staging / "modeling_qwen3_oft.py").write_text(
        "from qwen_single_layer_rl.vllm.oft_hf_model import Qwen3OFTForCausalLM, Qwen3OFTModel\n",
        encoding="utf-8",
    )

    saved_weights = sorted(staging.glob("*.safetensors"))
    if not saved_weights:
        raise RuntimeError("OFT export produced no safetensors")
    config_path = staging / "config.json"
    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    if saved_config.get("model_type") != "qwen3_oft":
        raise RuntimeError(f"wrong exported model_type: {saved_config.get('model_type')}")
    if saved_config.get("architectures") != ["Qwen3OFTForCausalLM"]:
        raise RuntimeError(f"wrong exported architecture: {saved_config.get('architectures')}")

    manifest = {
        "status": "PASS",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": str(base_model),
        "config": str(args.config.resolve()),
        "output": str(output),
        "target_layers": target_layers,
        "target_modules": target_modules,
        "attention_oft": False,
        "base_swiglu_trainable": False,
        "fp32_compute": True,
        "oft_parameter_count": len(oft_parameters),
        "oft_parameter_names": sorted(oft_parameters),
        "oft_parameter_dtypes": sorted({str(parameter.dtype) for parameter in oft_parameters.values()}),
        "exact_identity_component_parity": True,
        "weight_file_count": len(saved_weights),
        "weight_bytes": sum(path.stat().st_size for path in saved_weights),
        "config_sha256": sha256(config_path),
    }
    (staging / manifest_path.name).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (staging / "EXPORT_COMPLETE").touch()

    if output.exists():
        shutil.rmtree(output)
    staging.replace(output)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
