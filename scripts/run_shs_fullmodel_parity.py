#!/usr/bin/env python3
"""Build a disposable SHS export and validate full-model Triton parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen_single_layer_rl.config import load_config
from qwen_single_layer_rl.model_surgery import build_variant
from qwen_single_layer_rl.model_surgery.qwen_swiglu_variant_modules import ShuffledHyperGridDeltaLinear
from qwen_single_layer_rl.sft.checkpoint import load_trainable_state_dict
from qwen_single_layer_rl.vllm.shs_hf_model import Qwen3SHSConfig, Qwen3SHSForCausalLM
from qwen_single_layer_rl.vllm import SHS_ARCHITECTURE


RUN_ID = "shs_fullmodel_kernel_parity_20260712_v1"
SEED = 20260712
PROMPT_ID = "fullmodel_parity_arithmetic_v1"
PROMPT = "Solve carefully and give only the final answer: 17 + 25 ="
RESPONSE_CAP = 8
BF16_RTOL = 3.0e-2
BF16_ATOL = 8.0e-2
LOGITS_RELATIVE_L2_MAX = 1.0e-2
LOGITS_COSINE_MIN = 0.9999


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = bytes(tensor.detach().cpu().contiguous().view(torch.uint8).tolist())
    return hashlib.sha256(raw).hexdigest()


def deterministic_buffer_hashes(model) -> dict[str, str]:
    return {
        name: tensor_sha256(buffer)
        for name, buffer in model.named_buffers()
        if name.endswith(("_row_block_ids", "_col_block_ids"))
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def projections(model) -> list[ShuffledHyperGridDeltaLinear]:
    return [module for module in model.modules() if isinstance(module, ShuffledHyperGridDeltaLinear)]


def configure_backend(model, backend: str) -> None:
    found = projections(model)
    if len(found) != 3:
        raise RuntimeError(f"expected 3 SHS projections, found {len(found)}")
    for module in found:
        module.set_inference_mul_backend(backend)


def export_model(base_model: Path, config_path: Path, checkpoint_dir: Path, export_dir: Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    shs_params = dict(cfg["architecture_variant"]["params"])
    shs_params["inference_mul_backend"] = "triton"
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    torch.manual_seed(int(cfg["experiment"].get("init_seed", cfg["experiment"]["seed"])))
    model = build_variant(cfg).apply(model, cfg)
    trainable_state = torch.load(checkpoint_dir / "trainable_state.pt", map_location="cpu", weights_only=False)
    load_trainable_state_dict(model, trainable_state)

    base_dict = model.config.to_dict()
    for key in ("model_type", "architectures", "shs_variant", "auto_map"):
        base_dict.pop(key, None)
    custom_config = Qwen3SHSConfig(shs_variant=shs_params, **base_dict)
    model.config = custom_config
    export_dir.mkdir(parents=True, exist_ok=False)
    buffer_hashes = deterministic_buffer_hashes(model)
    export_state = {
        name: tensor
        for name, tensor in model.state_dict().items()
        if not name.endswith(("_row_block_ids", "_col_block_ids"))
    }
    model.save_pretrained(
        export_dir,
        state_dict=export_state,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    custom_config.architectures = [SHS_ARCHITECTURE]
    custom_config.auto_map = {
        "AutoConfig": "configuration_qwen3_shs.Qwen3SHSConfig",
        "AutoModel": "modeling_qwen3_shs.Qwen3SHSModel",
        "AutoModelForCausalLM": "modeling_qwen3_shs.Qwen3SHSForCausalLM",
    }
    custom_config.save_pretrained(export_dir)
    (export_dir / "configuration_qwen3_shs.py").write_text(
        "from qwen_single_layer_rl.vllm.shs_hf_model import Qwen3SHSConfig\n",
        encoding="utf-8",
    )
    (export_dir / "modeling_qwen3_shs.py").write_text(
        "from qwen_single_layer_rl.vllm.shs_hf_model import Qwen3SHSForCausalLM, Qwen3SHSModel\n",
        encoding="utf-8",
    )
    tokenizer.save_pretrained(export_dir)
    del model, trainable_state
    return {
        "export_files": sorted(path.name for path in export_dir.iterdir() if path.is_file()),
        "export_weight_bytes": sum(path.stat().st_size for path in export_dir.glob("*.safetensors")),
        "deterministic_buffer_hashes": buffer_hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    export_dir = output_dir / "deployment_export"
    output_dir.mkdir(parents=True, exist_ok=True)
    preregistration = {
        "run_id": RUN_ID,
        "status": "preregistered",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "prompt_ids": [PROMPT_ID],
        "prompts": [PROMPT],
        "response_cap": RESPONSE_CAP,
        "decoding": {"do_sample": False, "num_beams": 1},
        "backends": ["reference", "triton"],
        "parity_tolerance": {
            "layer10_rtol": BF16_RTOL,
            "layer10_atol": BF16_ATOL,
            "logits_relative_l2_max": LOGITS_RELATIVE_L2_MAX,
            "logits_cosine_min": LOGITS_COSINE_MIN,
            "logits_top1_equal_required": True,
            "greedy_tokens_equal_required": True,
        },
        "base_model": str(args.base_model.resolve()),
        "config": str(args.config.resolve()),
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "source_hashes": {
            "kernel": sha256(args.source_root / "src/qwen_single_layer_rl/kernels/shs_modulated_projection.py"),
            "variant_module": sha256(
                args.source_root / "src/qwen_single_layer_rl/model_surgery/qwen_swiglu_variant_modules.py"
            ),
            "hf_model": sha256(args.source_root / "src/qwen_single_layer_rl/vllm/shs_hf_model.py"),
            "config": sha256(args.config),
            "checkpoint_trainable": sha256(args.checkpoint_dir / "trainable_state.pt"),
            "checkpoint_trainer": sha256(args.checkpoint_dir / "trainer_state.pt"),
            "base_config": sha256(args.base_model / "config.json"),
            "base_weights": sha256(args.base_model / "model.safetensors"),
        },
    }
    write_json(output_dir / "preregistered_manifest.json", preregistration)

    started = time.perf_counter()
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_info = export_model(args.base_model, args.config, args.checkpoint_dir, export_dir)
    export_seconds = time.perf_counter() - started
    export_hashes = {path.name: sha256(path) for path in sorted(export_dir.iterdir()) if path.is_file()}

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    tokenizer = AutoTokenizer.from_pretrained(export_dir, trust_remote_code=True)
    load_started = time.perf_counter()
    model, loading_info = Qwen3SHSForCausalLM.from_pretrained(
        export_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    model = model.cuda().eval()
    model.config.use_cache = True
    load_seconds = time.perf_counter() - load_started
    persistent_buffer_devices = sorted({str(buffer.device) for module in projections(model) for buffer in module.buffers()})
    reloaded_buffer_hashes = deterministic_buffer_hashes(model)
    inputs = tokenizer(PROMPT, return_tensors="pt").to("cuda")

    captured: dict[str, torch.Tensor] = {}
    layer = model.model.layers[10]

    def capture(name: str):
        def hook(_module, _args, output):
            value = output[0] if isinstance(output, tuple) else output
            captured[name] = value.detach().float().cpu()
        return hook

    with torch.no_grad():
        configure_backend(model, "reference")
        handle = layer.register_forward_hook(capture("reference_layer10"))
        ref_logits = model(**inputs, use_cache=False).logits.detach().float().cpu()
        handle.remove()
        ref_generated = model.generate(**inputs, max_new_tokens=RESPONSE_CAP, do_sample=False)
        ref_text = tokenizer.decode(ref_generated[0], skip_special_tokens=True)

        configure_backend(model, "triton")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        triton_started = time.perf_counter()
        handle = layer.register_forward_hook(capture("triton_layer10"))
        tri_logits = model(**inputs, use_cache=False).logits.detach().float().cpu()
        handle.remove()
        tri_generated = model.generate(**inputs, max_new_tokens=RESPONSE_CAP, do_sample=False)
        torch.cuda.synchronize()
        triton_seconds = time.perf_counter() - triton_started
        triton_peak = torch.cuda.max_memory_allocated()
        tri_text = tokenizer.decode(tri_generated[0], skip_special_tokens=True)

    layer_error = (captured["reference_layer10"] - captured["triton_layer10"]).abs()
    logits_error = (ref_logits - tri_logits).abs()
    layer_limit = BF16_ATOL + BF16_RTOL * captured["reference_layer10"].abs()
    logits_limit = BF16_ATOL + BF16_RTOL * ref_logits.abs()
    layer_mismatch = layer_error > layer_limit
    logits_mismatch = logits_error > logits_limit
    logits_relative_l2 = torch.linalg.vector_norm(logits_error) / torch.linalg.vector_norm(ref_logits)
    logits_cosine = torch.nn.functional.cosine_similarity(ref_logits.flatten(), tri_logits.flatten(), dim=0)
    logits_top1_equal = torch.equal(ref_logits.argmax(dim=-1), tri_logits.argmax(dim=-1))
    dispatch = [module.last_inference_mul_backend for module in projections(model)]
    loaded = set(model.state_dict())
    checkpoint_keys = set(torch.load(args.checkpoint_dir / "trainable_state.pt", map_location="cpu", weights_only=False))
    missing_checkpoint_keys = sorted(checkpoint_keys - loaded)
    manifest = {
        **preregistration,
        "status": "passed",
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "export": {
            **export_info,
            "seconds": export_seconds,
            "hashes": export_hashes,
        },
        "loading": {
            "seconds": load_seconds,
            "missing_keys": sorted(loading_info.get("missing_keys", [])),
            "unexpected_keys": sorted(loading_info.get("unexpected_keys", [])),
            "mismatched_keys": sorted(str(item) for item in loading_info.get("mismatched_keys", [])),
            "error_messages": loading_info.get("error_msgs", []),
            "checkpoint_overlay_keys": len(checkpoint_keys),
            "missing_checkpoint_overlay_keys": missing_checkpoint_keys,
            "persistent_buffer_devices": persistent_buffer_devices,
            "deterministic_buffer_hashes": reloaded_buffer_hashes,
            "deterministic_buffer_hashes_match": (
                reloaded_buffer_hashes == export_info["deterministic_buffer_hashes"]
            ),
        },
        "parity": {
            "layer10_max_abs": float(layer_error.max()),
            "layer10_mean_abs": float(layer_error.mean()),
            "layer10_mismatch_count": int(layer_mismatch.sum()),
            "layer10_element_count": layer_error.numel(),
            "logits_max_abs": float(logits_error.max()),
            "logits_mean_abs": float(logits_error.mean()),
            "logits_mismatch_count": int(logits_mismatch.sum()),
            "logits_element_count": logits_error.numel(),
            "logits_relative_l2": float(logits_relative_l2),
            "logits_cosine": float(logits_cosine),
            "logits_top1_equal": bool(logits_top1_equal),
            "greedy_token_ids_equal": bool(torch.equal(ref_generated.cpu(), tri_generated.cpu())),
            "greedy_text_equal": ref_text == tri_text,
            "reference_text": ref_text,
            "triton_text": tri_text,
        },
        "triton": {
            "last_dispatch": dispatch,
            "all_three_dispatched": dispatch == ["triton", "triton", "triton"],
            "forward_and_generation_seconds": triton_seconds,
            "peak_allocated_bytes": triton_peak,
        },
    }
    failures = []
    if any(manifest["loading"][key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_messages")):
        failures.append("loading_info_not_clean")
    if missing_checkpoint_keys:
        failures.append("checkpoint_overlay_keys_missing")
    if not manifest["loading"]["deterministic_buffer_hashes_match"]:
        failures.append("deterministic_buffer_hash_mismatch")
    if not manifest["triton"]["all_three_dispatched"]:
        failures.append("triton_dispatch_incomplete")
    if not manifest["parity"]["greedy_token_ids_equal"]:
        failures.append("greedy_tokens_differ")
    if manifest["parity"]["layer10_mismatch_count"]:
        failures.append("numerical_tolerance_exceeded")
    if (
        manifest["parity"]["logits_relative_l2"] > LOGITS_RELATIVE_L2_MAX
        or manifest["parity"]["logits_cosine"] < LOGITS_COSINE_MIN
        or not manifest["parity"]["logits_top1_equal"]
    ):
        failures.append("logits_parity_exceeded")
    manifest["failures"] = failures
    manifest["status"] = "passed" if not failures else "failed"
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
