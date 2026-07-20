#!/usr/bin/env python3
"""Validate real-checkpoint SHS Triton-forward/reference-recompute autograd."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from qwen_single_layer_rl.eval.checkpoint_loader import load_sft_checkpoint_for_inference
from qwen_single_layer_rl.layers import apply_freeze_policy
from qwen_single_layer_rl.model_surgery.qwen_swiglu_variant_modules import (
    QwenSwiGLUSHSWrapper,
    ShuffledHyperGridDeltaLinear,
)
from qwen_single_layer_rl.sft.checkpoint import load_latest_checkpoint, save_checkpoint
from qwen_single_layer_rl.sft.data import collate_packed_items, load_packed_cache


RUN_ID = "shs_triton_autograd_parity_20260712_v1"
SEED = 20260712
TAIL_TOKENS = 256
LR = 5.0e-6
LOSS_ABS_MAX = 1.0e-2
GRADIENT_COSINE_MIN = 0.999
GRADIENT_RELATIVE_L2_MAX = 5.0e-2


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def modules(model) -> list[ShuffledHyperGridDeltaLinear]:
    found = [module for module in model.modules() if isinstance(module, ShuffledHyperGridDeltaLinear)]
    if len(found) != 3:
        raise RuntimeError(f"expected three SHS projections, found {len(found)}")
    return found


def configure(model, backend: str) -> None:
    for module in modules(model):
        module.set_inference_mul_backend(backend)


def load_actor(args, backend: str):
    model, _, cfg = load_sft_checkpoint_for_inference(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        model_path=args.base_model,
        device="cuda",
    )
    apply_freeze_policy(model, cfg)
    configure(model, backend)
    model.config.use_cache = False
    model.train()
    return model


def batch_from_cache(cache: Path) -> dict[str, torch.Tensor]:
    dataset, _ = load_packed_cache(cache)
    batch = collate_packed_items([dataset[0]])
    return {key: value[:, -TAIL_TOKENS:].cuda() for key, value in batch.items() if key in ("input_ids", "labels", "attention_mask")}


def loss_precisions(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    shifted_logits = logits[:, :-1].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    valid = shifted_labels != -100
    selected_logits = shifted_logits[valid]
    selected_labels = shifted_labels[valid]
    return {
        "fp32_cross_entropy": float(F.cross_entropy(selected_logits.float(), selected_labels).detach()),
        "bf16_cross_entropy": float(F.cross_entropy(selected_logits.bfloat16(), selected_labels).detach()),
    }


def gradient_group(name: str) -> str:
    if ".shs.grid_generator." in name:
        return "grid_generator"
    if any(f".shs.{projection}.mul_scale" in name for projection in ("gate", "up", "down")):
        return "multiplicative_scale"
    if any(f".shs.{projection}.add_" in name for projection in ("gate", "up", "down")):
        return "additive_low_rank"
    if any(f".base_mlp.{projection}_proj.weight" in name for projection in ("gate", "up", "down")):
        return "base_projection_weights"
    return "other_layer10_trainable"


def snapshot_gradients(model, layer10_input: torch.Tensor) -> dict[str, torch.Tensor]:
    gradients = {"__layer10_input__": layer10_input.grad.detach().float().cpu()}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if parameter.grad is None:
                raise RuntimeError(f"missing gradient: {name}")
            gradients[name] = parameter.grad.detach().float().cpu()
    return gradients


def compare_tensor(reference: torch.Tensor, actual: torch.Tensor) -> dict:
    error = actual - reference
    ref_norm = torch.linalg.vector_norm(reference)
    error_norm = torch.linalg.vector_norm(error)
    if torch.equal(reference, actual):
        cosine = 1.0
    elif ref_norm == 0 or torch.linalg.vector_norm(actual) == 0:
        cosine = 0.0
    else:
        cosine = float(F.cosine_similarity(reference.flatten(), actual.flatten(), dim=0))
    return {
        "reference_norm": float(ref_norm),
        "actual_norm": float(torch.linalg.vector_norm(actual)),
        "max_abs": float(error.abs().max()),
        "relative_l2": float(error_norm / ref_norm.clamp_min(torch.finfo(torch.float32).tiny)),
        "cosine": cosine,
        "finite": bool(torch.isfinite(actual).all()),
    }


def backward_capture(model, batch) -> tuple[dict, dict[str, torch.Tensor], list[str]]:
    model.zero_grad(set_to_none=True)
    captured = {}
    wrapper = next(module for module in model.modules() if isinstance(module, QwenSwiGLUSHSWrapper))

    def retain_input(_module, args):
        args[0].retain_grad()
        captured["input"] = args[0]

    handle = wrapper.register_forward_pre_hook(retain_input)
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = model(**batch, use_cache=False)
    precisions = loss_precisions(output.logits, batch["labels"])
    output.loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    handle.remove()
    gradients = snapshot_gradients(model, captured["input"])
    receipt = [module.last_inference_mul_backend for module in modules(model)]
    return {"model_loss": float(output.loss.detach()), **precisions, "forward_backward_seconds": elapsed}, gradients, receipt


def optimizer_step(model, optimizer, batch) -> dict:
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = model(**batch, use_cache=False)
    output.loss.backward()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
    optimizer.step()
    torch.cuda.synchronize()
    changed = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and not torch.equal(parameter.detach(), before[name]):
            changed.append(name)
    return {"loss": float(output.loss.detach()), "seconds": time.perf_counter() - started, "changed": changed}


def parameter_max_difference(left, right) -> float:
    right_parameters = dict(right.named_parameters())
    return max(
        float((parameter.detach().float() - right_parameters[name].detach().float()).abs().max())
        for name, parameter in left.named_parameters()
        if parameter.requires_grad
    )


def write_report(path: Path, manifest: dict) -> None:
    worst = manifest["gradient_parity"]["worst"]
    lines = [
        f"# {RUN_ID}",
        "",
        f"Status: **{manifest['status']}**",
        "",
        "Backend: Triton multiplicative forward plus PyTorch reference-recompute backward.",
        "This is explicitly not a custom backward kernel.",
        "",
        f"Loss absolute difference: `{manifest['loss_parity']['model_loss_abs_difference']}`.",
        f"Worst gradient cosine: `{worst['cosine']['value']}` (`{worst['cosine']['name']}`).",
        f"Worst gradient relative L2: `{worst['relative_l2']['value']}` (`{worst['relative_l2']['name']}`).",
        f"Dispatch: `{manifest['dispatch']['actual']}`; fallback: `{manifest['dispatch']['fallback']}`.",
        f"Checkpoint reload max difference: `{manifest['checkpoint']['reload_max_abs_difference']}`.",
        f"Resumed second-update max difference: `{manifest['checkpoint']['resumed_update_max_abs_difference']}`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = output_dir / "checkpoints"
    if checkpoint_root.exists():
        shutil.rmtree(checkpoint_root)
    prereg = {
        "run_id": RUN_ID,
        "status": "preregistered",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "tail_tokens": TAIL_TOKENS,
        "thresholds": {
            "loss_abs_max": LOSS_ABS_MAX,
            "gradient_cosine_min": GRADIENT_COSINE_MIN,
            "gradient_relative_l2_max": GRADIENT_RELATIVE_L2_MAX,
        },
        "backend": "triton_forward_reference_recompute_backward",
        "claims": {"triton_forward": True, "reference_recompute_backward": True, "custom_backward": False, "production_candidate": False, "production_ready": False},
        "checkpoint_trainable_sha256": sha256(args.checkpoint_dir / "trainable_state.pt"),
        "validation_cache_sha256": sha256(args.validation_cache),
        "source_hashes": {
            "kernel": sha256(args.source_root / "src/qwen_single_layer_rl/kernels/shs_modulated_projection.py"),
            "variant_module": sha256(args.source_root / "src/qwen_single_layer_rl/model_surgery/qwen_swiglu_variant_modules.py"),
            "runner": sha256(args.source_root / "scripts/run_shs_autograd_parity.py"),
        },
    }
    write_json(output_dir / "preregistered_manifest.json", prereg)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    batch = batch_from_cache(args.validation_cache)
    actor = load_actor(args, "reference")
    reference_losses, reference_gradients, reference_receipt = backward_capture(actor, batch)
    configure(actor, "triton_reference_recompute")
    triton_losses, triton_gradients, triton_receipt = backward_capture(actor, batch)

    comparisons = {name: compare_tensor(reference_gradients[name], gradient) for name, gradient in triton_gradients.items()}
    groups = {}
    for name, comparison in comparisons.items():
        group = "layer10_input" if name == "__layer10_input__" else gradient_group(name)
        groups.setdefault(group, []).append(name)
    worst_cosine_name = min(comparisons, key=lambda name: comparisons[name]["cosine"])
    worst_l2_name = max(comparisons, key=lambda name: comparisons[name]["relative_l2"])

    optimizer = torch.optim.AdamW([parameter for parameter in actor.parameters() if parameter.requires_grad], lr=LR, weight_decay=0.01)
    step1 = optimizer_step(actor, optimizer, batch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    saved = save_checkpoint(
        checkpoint_root,
        model=actor,
        optimizer=optimizer,
        scheduler=scheduler,
        trainer_state={"epoch": 0, "micro_batch_cursor": 1, "global_step": 1, "global_order_sha256": "autograd_parity_item_0"},
        manifest=prereg,
        keep_last=1,
    )
    resumed = load_actor(args, "triton_reference_recompute")
    resumed_optimizer = torch.optim.AdamW([parameter for parameter in resumed.parameters() if parameter.requires_grad], lr=LR, weight_decay=0.01)
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(resumed_optimizer, lambda _: 1.0)
    resumed_state = load_latest_checkpoint(checkpoint_root, model=resumed, optimizer=resumed_optimizer, scheduler=resumed_scheduler, device=torch.device("cuda"))
    reload_difference = parameter_max_difference(actor, resumed)
    step2 = optimizer_step(actor, optimizer, batch)
    resumed_step2 = optimizer_step(resumed, resumed_optimizer, batch)
    resumed_update_difference = parameter_max_difference(actor, resumed)

    loss_difference = abs(reference_losses["model_loss"] - triton_losses["model_loss"])
    failures = []
    if reference_receipt != ["reference"] * 3:
        failures.append("reference_dispatch_mismatch")
    expected_triton_receipt = ["triton_forward_reference_recompute_backward"] * 3
    if triton_receipt != expected_triton_receipt:
        failures.append("triton_dispatch_mismatch")
    if loss_difference > LOSS_ABS_MAX:
        failures.append("loss_parity_exceeded")
    for name, comparison in comparisons.items():
        if not comparison["finite"]:
            failures.append(f"nonfinite_gradient:{name}")
        if comparison["cosine"] < GRADIENT_COSINE_MIN and comparison["relative_l2"] > GRADIENT_RELATIVE_L2_MAX:
            failures.append(f"gradient_parity_exceeded:{name}")
    required_groups = {"layer10_input", "base_projection_weights", "grid_generator", "multiplicative_scale", "additive_low_rank"}
    if not required_groups.issubset(groups):
        failures.append(f"gradient_groups_missing:{sorted(required_groups - set(groups))}")
    if not step1["changed"]:
        failures.append("optimizer_changed_no_parameters")
    if reload_difference != 0.0:
        failures.append("checkpoint_reload_mismatch")
    if abs(step2["loss"] - resumed_step2["loss"]) > 1.0e-6 or resumed_update_difference != 0.0:
        failures.append("deterministic_resume_mismatch")

    manifest = {
        **prereg,
        "status": "passed" if not failures else "failed",
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
        "resolved_topology": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "visible_gpu_count": torch.cuda.device_count(),
            "launcher_world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "rank": int(os.environ.get("RANK", "0")),
        },
        "reference_losses": reference_losses,
        "triton_losses": triton_losses,
        "loss_parity": {"model_loss_abs_difference": loss_difference},
        "gradient_parity": {
            "parameters": comparisons,
            "groups": groups,
            "worst": {
                "cosine": {"name": worst_cosine_name, "value": comparisons[worst_cosine_name]["cosine"]},
                "relative_l2": {"name": worst_l2_name, "value": comparisons[worst_l2_name]["relative_l2"]},
            },
        },
        "dispatch": {"reference": reference_receipt, "actual": triton_receipt, "fallback": triton_receipt != expected_triton_receipt},
        "optimizer_step1": step1,
        "checkpoint": {
            "path": str(saved),
            "resumed_state": resumed_state,
            "reload_max_abs_difference": reload_difference,
            "step2": step2,
            "resumed_step2": resumed_step2,
            "resumed_update_max_abs_difference": resumed_update_difference,
        },
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "failures": failures,
    }
    write_json(output_dir / "manifest.json", manifest)
    write_report(output_dir / "report.md", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
