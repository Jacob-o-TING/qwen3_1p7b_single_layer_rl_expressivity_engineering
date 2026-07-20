#!/usr/bin/env python3
"""Localize SHS grouped-kernel numerical drift through the full model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from qwen_single_layer_rl.eval.checkpoint_loader import load_sft_checkpoint_for_inference
from qwen_single_layer_rl.model_surgery.qwen_swiglu_variant_modules import (
    QwenSwiGLUSHSWrapper,
    ShuffledHyperGridDeltaLinear,
)


RUN_ID = "shs_triton_drift_localize_20260712_v1"
SEED = 20260712
PROMPT = "Solve carefully and give only the final answer: 17 + 25 ="
LOGITS_COSINE_MIN = 0.9999
MODES = {
    "reference_equivalent": "reference",
    "grouped_bf16": "triton",
    "grouped_fp32_accumulate": "triton_fp32",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def configure_backend(model, backend: str) -> list[ShuffledHyperGridDeltaLinear]:
    modules = [module for module in model.modules() if isinstance(module, ShuffledHyperGridDeltaLinear)]
    if len(modules) != 3:
        raise RuntimeError(f"expected three SHS projections, found {len(modules)}")
    for module in modules:
        module.set_inference_mul_backend(backend)
    return modules


def metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict:
    reference = reference.float()
    actual = actual.float()
    error = actual - reference
    reference_norm = torch.linalg.vector_norm(reference)
    relative_l2 = torch.linalg.vector_norm(error) / reference_norm.clamp_min(torch.finfo(torch.float32).tiny)
    cosine = torch.tensor(1.0) if torch.equal(reference, actual) else F.cosine_similarity(
        reference.flatten(), actual.flatten(), dim=0
    )
    k = min(10, reference.shape[-1])
    reference_topk = reference.topk(k, dim=-1).indices
    actual_topk = actual.topk(k, dim=-1).indices
    overlap = (reference_topk.unsqueeze(-1) == actual_topk.unsqueeze(-2)).any(dim=-1).float().mean()
    return {
        "shape": list(reference.shape),
        "max_abs": float(error.abs().max()),
        "mean_abs": float(error.abs().mean()),
        "relative_l2": float(relative_l2),
        "cosine": float(cosine),
        "top10_overlap": float(overlap),
    }


def capture_run(model, input_ids: torch.Tensor, mode: str) -> tuple[dict[str, torch.Tensor], dict]:
    backend = MODES[mode]
    projections = configure_backend(model, backend)
    wrapper = next(module for module in model.modules() if isinstance(module, QwenSwiGLUSHSWrapper))
    layers = model.model.layers
    captures: dict[str, torch.Tensor] = {}
    handles = []

    def output_hook(name: str):
        def hook(_module, _args, output):
            value = output[0] if isinstance(output, tuple) else output
            captures[name] = value.detach().cpu()
        return hook

    def input_hook(name: str):
        def hook(_module, args):
            captures[name] = args[0].detach().cpu()
        return hook

    for name in ("gate", "up"):
        handles.append(wrapper.shs[name].register_forward_hook(output_hook(f"layer10_{name}_projection")))
    handles.append(wrapper.shs["down"].register_forward_pre_hook(input_hook("layer10_swiglu")))
    handles.append(wrapper.shs["down"].register_forward_hook(output_hook("layer10_down_projection")))
    for layer_id in range(10, len(layers)):
        handles.append(layers[layer_id].register_forward_hook(output_hook(f"layer{layer_id}_residual")))

    torch.cuda.synchronize()
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False).logits
    torch.cuda.synchronize()
    captures["final_logits"] = logits.detach().cpu()
    for handle in handles:
        handle.remove()
    dispatch = [module.last_inference_mul_backend for module in projections]
    expected_dispatch = [backend if backend != "reference" else "reference"] * 3
    receipt = {
        "requested_mode": mode,
        "requested_backend": backend,
        "actual_projection_backends": dispatch,
        "expected_projection_backends": expected_dispatch,
        "fallback": dispatch != expected_dispatch,
    }
    if receipt["fallback"]:
        raise RuntimeError(f"dispatch mismatch: {receipt}")
    return captures, receipt


def write_report(path: Path, manifest: dict) -> None:
    lines = [
        f"# {RUN_ID}",
        "",
        f"Status: **{manifest['status']}**",
        "",
        f"Fixed logits cosine gate: `{LOGITS_COSINE_MIN}` (not relaxed).",
        "",
    ]
    for panel, result in manifest["panels"].items():
        lines.extend([f"## Panel {panel}", ""])
        for mode, mode_result in result["modes"].items():
            logits = mode_result["points"]["final_logits"]
            lines.append(
                f"- `{mode}`: logits cosine {logits['cosine']:.8f}, relative L2 {logits['relative_l2']:.8f}; "
                f"first amplification `{mode_result['first_amplification_point']}`; dispatch "
                f"`{mode_result['dispatch']['actual_projection_backends']}`."
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prereg = {
        "run_id": RUN_ID,
        "status": "preregistered",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "input_panels": ["full_context_reproduction", "flattened_pressure_1", "flattened_pressure_32"],
        "prompt": PROMPT,
        "modes": list(MODES),
        "reference_mode": "reference_equivalent",
        "logits_cosine_min": LOGITS_COSINE_MIN,
        "checkpoint_trainable_sha256": sha256(args.checkpoint_dir / "trainable_state.pt"),
        "source_hashes": {
            "kernel": sha256(args.source_root / "src/qwen_single_layer_rl/kernels/shs_modulated_projection.py"),
            "variant_module": sha256(
                args.source_root / "src/qwen_single_layer_rl/model_surgery/qwen_swiglu_variant_modules.py"
            ),
            "runner": sha256(args.source_root / "scripts/run_shs_drift_localization.py"),
        },
        "claims": {"production_candidate": False, "production_ready": False},
    }
    write_json(output_dir / "preregistered_manifest.json", prereg)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model, _, _ = load_sft_checkpoint_for_inference(
        config_path=args.config,
        checkpoint_dir=args.checkpoint_dir,
        model_path=args.base_model,
        device="cuda",
    )
    model.eval()
    model.config.use_cache = False
    prompt_ids = tokenizer(PROMPT, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
    token_id = prompt_ids[:, -1:]
    repeat_count = (32 + prompt_ids.shape[1] - 1) // prompt_ids.shape[1]
    panels = {
        "full_context_reproduction": prompt_ids,
        "flattened_pressure_1": token_id,
        "flattened_pressure_32": prompt_ids.repeat(1, repeat_count)[:, :32],
    }
    panel_results = {}
    failures = []
    for panel, input_ids in panels.items():
        captures_by_mode = {}
        receipts = {}
        for mode in MODES:
            captures_by_mode[mode], receipts[mode] = capture_run(model, input_ids, mode)
        reference = captures_by_mode["reference_equivalent"]
        modes = {}
        ordered_points = list(reference)
        for mode, captures in captures_by_mode.items():
            point_metrics = {name: metrics(reference[name], captures[name]) for name in ordered_points}
            layer10_l2 = point_metrics["layer10_residual"]["relative_l2"]
            first_amplification = None
            for name in ordered_points[ordered_points.index("layer10_residual") + 1 :]:
                if point_metrics[name]["relative_l2"] > max(2.0 * layer10_l2, 1.0e-7):
                    first_amplification = name
                    break
            modes[mode] = {
                "dispatch": receipts[mode],
                "points": point_metrics,
                "first_amplification_point": first_amplification,
            }
            if mode != "reference_equivalent" and point_metrics["final_logits"]["cosine"] < LOGITS_COSINE_MIN:
                failures.append(f"{panel}:{mode}:logits_cosine_below_gate")
        panel_results[panel] = {"flattened_tokens": input_ids.numel(), "modes": modes}

    manifest = {
        **prereg,
        "status": "passed" if not failures else "failed",
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "resolved_topology": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "visible_gpu_count": torch.cuda.device_count(),
            "visible_gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
            "launcher_world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "rank": int(os.environ.get("RANK", "0")),
            "tensor_parallel_size": 1,
            "replica_count": 1,
        },
        "panels": panel_results,
        "failures": failures,
    }
    write_json(output_dir / "manifest.json", manifest)
    write_report(output_dir / "report.md", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
