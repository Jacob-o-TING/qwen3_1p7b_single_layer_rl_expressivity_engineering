#!/usr/bin/env python3
"""Reference-backend SFT optimizer-step and deterministic resume smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from qwen_single_layer_rl.config import load_config
from qwen_single_layer_rl.eval.checkpoint_loader import load_sft_checkpoint_for_inference
from qwen_single_layer_rl.layers import apply_freeze_policy
from qwen_single_layer_rl.model_surgery.qwen_swiglu_variant_modules import ShuffledHyperGridDeltaLinear
from qwen_single_layer_rl.sft.checkpoint import load_latest_checkpoint, save_checkpoint
from qwen_single_layer_rl.sft.data import collate_packed_items, load_packed_cache


RUN_ID = "shs_triton_sft_step_smoke_20260712_v1"
SEED = 20260712
TAIL_TOKENS = 256
LR = 5.0e-6


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def configure_reference(model) -> None:
    modules = [m for m in model.modules() if isinstance(m, ShuffledHyperGridDeltaLinear)]
    if len(modules) != 3:
        raise RuntimeError(f"expected 3 SHS projections, found {len(modules)}")
    for module in modules:
        module.set_inference_mul_backend("reference")


def load_actor(config: Path, checkpoint: Path, base_model: Path):
    model, _, cfg = load_sft_checkpoint_for_inference(
        config_path=config,
        checkpoint_dir=checkpoint,
        model_path=base_model,
        device="cuda",
    )
    apply_freeze_policy(model, cfg)
    configure_reference(model)
    model.config.use_cache = False
    model.train()
    return model, cfg


def batch_from_cache(cache: Path) -> dict[str, torch.Tensor]:
    dataset, _ = load_packed_cache(cache)
    batch = collate_packed_items([dataset[0]])
    for key in ("input_ids", "labels", "attention_mask"):
        batch[key] = batch[key][:, -TAIL_TOKENS:].cuda()
    return batch


def train_step(model, optimizer, batch) -> dict:
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = model(**batch, use_cache=False)
    output.loss.backward()
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    grad_norm_sq = sum(float(parameter.grad.float().norm()) ** 2 for _, parameter in trainable if parameter.grad is not None)
    before = {name: parameter.detach().clone() for name, parameter in trainable}
    optimizer.step()
    torch.cuda.synchronize()
    changed = []
    max_update = 0.0
    for name, parameter in trainable:
        delta = (parameter.detach().float() - before[name].float()).abs().max().item()
        if delta:
            changed.append(name)
            max_update = max(max_update, delta)
    return {
        "loss": float(output.loss.detach()),
        "grad_norm": math.sqrt(grad_norm_sq),
        "changed_parameter_count": len(changed),
        "changed_parameter_names": changed,
        "max_abs_update": max_update,
        "seconds": time.perf_counter() - started,
    }


def parameter_max_difference(left, right) -> float:
    right_params = dict(right.named_parameters())
    return max(
        (parameter.detach().float() - right_params[name].detach().float()).abs().max().item()
        for name, parameter in left.named_parameters()
        if parameter.requires_grad
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--validation-cache", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = output_dir / "checkpoints"
    if checkpoint_root.exists():
        shutil.rmtree(checkpoint_root)
    prereg = {
        "run_id": RUN_ID,
        "status": "preregistered",
        "label": "reference_training_integration; inference-only Triton kernel disabled",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "checkpoint_trainable_sha256": sha256(args.checkpoint_dir / "trainable_state.pt"),
        "config": str(args.config.resolve()),
        "validation_cache": str(args.validation_cache.resolve()),
        "validation_cache_sha256": sha256(args.validation_cache),
        "dataset_item_indices": [0],
        "sequence_slice": f"tail_{TAIL_TOKENS}",
        "optimizer": {"name": "AdamW", "learning_rate": LR, "weight_decay": 0.01},
        "backend": "reference",
        "attention_backend": "deterministic math SDPA",
        "claims": {"triton_forward": False, "triton_backward": False, "optimized_training": False},
    }
    write_json(output_dir / "preregistered_manifest.json", prereg)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    batch = batch_from_cache(args.validation_cache)
    actor, cfg = load_actor(args.config, args.checkpoint_dir, args.base_model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in actor.parameters() if parameter.requires_grad], lr=LR, weight_decay=0.01
    )
    step1 = train_step(actor, optimizer, batch)
    saved = save_checkpoint(
        checkpoint_root,
        model=actor,
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0),
        trainer_state={"epoch": 0, "micro_batch_cursor": 1, "global_step": 1, "global_order_sha256": "smoke_item_0"},
        manifest=prereg,
        keep_last=2,
    )

    torch.manual_seed(SEED + 1)
    torch.cuda.manual_seed_all(SEED + 1)
    resumed, _ = load_actor(args.config, args.checkpoint_dir, args.base_model)
    resumed_optimizer = torch.optim.AdamW(
        [parameter for parameter in resumed.parameters() if parameter.requires_grad], lr=LR, weight_decay=0.01
    )
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(resumed_optimizer, lambda _: 1.0)
    resumed_state = load_latest_checkpoint(
        checkpoint_root,
        model=resumed,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        device=torch.device("cuda"),
    )
    reload_difference = parameter_max_difference(actor, resumed)
    step2_original = train_step(actor, optimizer, batch)
    step2_resumed = train_step(resumed, resumed_optimizer, batch)
    second_update_difference = parameter_max_difference(actor, resumed)
    manifest = {
        **prereg,
        "status": "passed",
        "step1": step1,
        "checkpoint": {
            "path": str(saved),
            "trainable_state_sha256": sha256(saved / "trainable_state.pt"),
            "trainer_state_sha256": sha256(saved / "trainer_state.pt"),
            "resumed_state": resumed_state,
            "reload_max_abs_parameter_difference": reload_difference,
        },
        "step2_original": step2_original,
        "step2_resumed": step2_resumed,
        "second_step_loss_abs_difference": abs(step2_original["loss"] - step2_resumed["loss"]),
        "second_step_max_abs_parameter_difference": second_update_difference,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    failures = []
    if step1["changed_parameter_count"] == 0:
        failures.append("optimizer_changed_no_parameters")
    if reload_difference != 0.0:
        failures.append("checkpoint_reload_parameter_mismatch")
    if manifest["second_step_loss_abs_difference"] > 1.0e-6 or second_update_difference != 0.0:
        failures.append("deterministic_resume_step_mismatch")
    manifest["failures"] = failures
    manifest["status"] = "passed" if not failures else "failed"
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
