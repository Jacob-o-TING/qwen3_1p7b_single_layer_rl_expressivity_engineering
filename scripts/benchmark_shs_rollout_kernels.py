#!/usr/bin/env python3
"""Correctness and timing driver for the SHS multiplicative projection."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from qwen_single_layer_rl.kernels.shs_modulated_projection import (
    shs_modulated_projection,
    shs_modulated_projection_reference,
)
from qwen_single_layer_rl.model_surgery.qwen_swiglu_variant_modules import make_shuffled_block_ids


RUN_ID = "shs_modproj_triton_decode_bench_20260711_v1"
SHAPES = {
    "gate": (6144, 2048),
    "up": (6144, 2048),
    "down": (2048, 6144),
    "odd_tail": (2111, 2053),
}


def synchronize() -> None:
    torch.cuda.synchronize()


def gpu_time_ms(fn, warmup: int, repeats: int) -> tuple[float, list[float]]:
    for _ in range(warmup):
        fn()
    synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        synchronize()
        samples.append(float(start.elapsed_time(end)))
    return statistics.median(samples), samples


def tolerances(dtype: torch.dtype) -> tuple[float, float]:
    return (2e-4, 2e-4) if dtype == torch.float32 else (3e-2, 8e-2)


def make_case(m: int, n: int, k: int, dtype: torch.dtype, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m, k), generator=generator, device="cuda", dtype=dtype) * 0.2
    weight = torch.randn((n, k), generator=generator, device="cuda", dtype=dtype) * (k**-0.5)
    grid = torch.randn((m, 32, 32), generator=generator, device="cuda", dtype=dtype) * 0.5
    row_ids = make_shuffled_block_ids(n, 32, seed + 101).to("cuda")
    col_ids = make_shuffled_block_ids(k, 32, seed + 7919).to("cuda")
    return x, weight, grid, row_ids, col_ids


def correctness_case(label: str, m: int, n: int, k: int, dtype: torch.dtype, seed: int) -> dict:
    record = {"label": label, "m": m, "n": n, "k": k, "dtype": str(dtype), "status": "failed"}
    try:
        x, weight, grid, row_ids, col_ids = make_case(m, n, k, dtype, seed)
        expected = shs_modulated_projection_reference(x, weight, grid, row_ids, col_ids, 0.03125)
        synchronize()
        started = time.perf_counter()
        result = shs_modulated_projection(x, weight, grid, row_ids, col_ids, 0.03125, backend="triton")
        synchronize()
        cold_ms = (time.perf_counter() - started) * 1000
        actual = result.output
        difference = (actual.float() - expected.float()).abs()
        rtol, atol = tolerances(dtype)
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        record.update(
            status=result.backend,
            cold_compile_and_run_ms=cold_ms,
            max_abs_error=float(difference.max()),
            mean_abs_error=float(difference.mean()),
            rtol=rtol,
            atol=atol,
        )
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def exact_noop_checks(seed: int) -> dict:
    x, weight, grid, row_ids, col_ids = make_case(2, 127, 131, torch.float32, seed)
    expected = F.linear(x, weight)
    zero_grid = shs_modulated_projection_reference(x, weight, torch.zeros_like(grid), row_ids, col_ids, 0.03125)
    zero_scale = shs_modulated_projection_reference(x, weight, grid, row_ids, col_ids, 0.0)
    return {
        "reference_zero_grid_exact": bool(torch.equal(expected, zero_grid)),
        "reference_zero_scale_exact": bool(torch.equal(expected, zero_scale)),
        "production_default_unchanged": True,
        "note": "The Triton path is opt-in; the production SHS module remains on its exact-noop reference path.",
    }


def performance_case(label: str, m: int, n: int, k: int, dtype: torch.dtype, seed: int, warmup: int, repeats: int) -> dict:
    x, weight, grid, row_ids, col_ids = make_case(m, n, k, dtype, seed)
    scale = torch.tensor(0.03125, device="cuda", dtype=torch.float32)
    reference_fn = lambda: shs_modulated_projection_reference(x, weight, grid, row_ids, col_ids, scale)
    triton_fn = lambda: shs_modulated_projection(x, weight, grid, row_ids, col_ids, scale, backend="triton").output
    record = {"label": label, "m": m, "n": n, "k": k, "dtype": str(dtype), "status": "failed"}
    try:
        torch.cuda.reset_peak_memory_stats()
        reference_median, reference_samples = gpu_time_ms(reference_fn, warmup, repeats)
        reference_peak = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        triton_median, triton_samples = gpu_time_ms(triton_fn, warmup, repeats)
        triton_peak = torch.cuda.max_memory_allocated()
        record.update(
            status="triton",
            reference_median_ms=reference_median,
            reference_samples_ms=reference_samples,
            reference_peak_allocated_bytes=reference_peak,
            triton_median_ms=triton_median,
            triton_samples_ms=triton_samples,
            triton_peak_allocated_bytes=triton_peak,
            speedup_reference_over_triton=reference_median / triton_median,
        )
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def compiled_reference_case(seed: int, warmup: int, repeats: int) -> dict:
    m, n, k = 8, 6144, 2048
    x, weight, grid, row_ids, col_ids = make_case(m, n, k, torch.bfloat16, seed)
    eager = lambda a, b, c: shs_modulated_projection_reference(a, b, c, row_ids, col_ids, 0.03125)
    record = {"label": "up_m8_bfloat16", "status": "failed"}
    try:
        compiled = torch.compile(eager, fullgraph=False)
        synchronize()
        started = time.perf_counter()
        compiled(x, weight, grid)
        synchronize()
        record["cold_compile_and_run_ms"] = (time.perf_counter() - started) * 1000
        median, samples = gpu_time_ms(lambda: compiled(x, weight, grid), warmup, repeats)
        record.update(status="torch_compile", median_ms=median, samples_ms=samples)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def environment() -> dict:
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True
        ).strip()
    except Exception as exc:
        smi = f"unavailable: {exc}"
    import triton

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "nvidia_smi": smi,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    token_batches = [1] if args.smoke else [1, 8, 16, 32]
    shape_items = [("odd_tail", SHAPES["odd_tail"])] if args.smoke else list(SHAPES.items())
    dtype_items = [torch.float32] if args.smoke else [torch.float32, torch.bfloat16]
    correctness = []
    for dtype in dtype_items:
        for label, (n, k) in shape_items:
            for m in token_batches:
                case_label = f"{label}_m{m}_{str(dtype).split('.')[-1]}"
                print(f"CORRECTNESS_START {case_label}", flush=True)
                record = correctness_case(case_label, m, n, k, dtype, args.seed)
                correctness.append(record)
                print("CORRECTNESS_RESULT " + json.dumps(record, sort_keys=True), flush=True)

    performance = []
    perf_shapes = [("gate", SHAPES["gate"])] if args.smoke else [(k, v) for k, v in SHAPES.items() if k != "odd_tail"]
    for label, (n, k) in perf_shapes:
        for m in token_batches:
            case_label = f"{label}_m{m}_bfloat16"
            print(f"PERFORMANCE_START {case_label}", flush=True)
            record = performance_case(case_label, m, n, k, torch.bfloat16, args.seed, args.warmup, args.repeats)
            performance.append(record)
            print("PERFORMANCE_RESULT " + json.dumps(record, sort_keys=True), flush=True)

    manifest = {
        "run_id": RUN_ID,
        "scope": "SHS multiplicative modulated projection inference only",
        "equation": "y[m,n] = sum_k x[m,k] * W[n,k] * (1 + scale * tanh(grid[m,row_id[n],col_id[k]]))",
        "excluded": ["additive low-rank path", "backward", "end-to-end rollout", "GRPO"],
        "seed": args.seed,
        "environment": environment(),
        "exact_noop": exact_noop_checks(args.seed),
        "correctness": correctness,
        "performance": performance,
        "torch_compile": None if args.smoke else compiled_reference_case(args.seed, args.warmup, args.repeats),
        "overall_status": "passed" if all(r["status"] == "triton" for r in correctness + performance) else "failed",
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"MANIFEST {manifest_path}")
    print(f"OVERALL_STATUS {manifest['overall_status']}")
    return 0 if manifest["overall_status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
