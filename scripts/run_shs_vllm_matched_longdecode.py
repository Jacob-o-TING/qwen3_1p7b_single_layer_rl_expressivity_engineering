#!/usr/bin/env python3
"""Run matched naive/reference/Triton vLLM long-response timing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


RUN_ID = "shs_vllm_matched_longdecode_20260712_v1"
SEED = 20260712
PRESSURES = (1, 8, 16, 32, 64)
MIN_TOKENS = 800
MAX_TOKENS = 1024


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def metric_value(metrics, name: str):
    value = getattr(metrics, name, None)
    return None if value is None else float(value)


def worker(args) -> int:
    os.environ["VLLM_USE_V1"] = "1"
    if args.shs_backend:
        os.environ["SHS_INFERENCE_MUL_BACKEND"] = args.shs_backend
        os.environ["SHS_DISPATCH_RECEIPT"] = str(args.receipt.resolve())
        args.receipt.unlink(missing_ok=True)
    from vllm import LLM, SamplingParams

    prompts_payload = json.loads(args.prompts.read_text(encoding="utf-8"))
    prompts = [row["prompt"] for row in prompts_payload]
    engine_started = time.perf_counter()
    llm = LLM(
        model=str(args.model.resolve()),
        model_impl="transformers" if args.shs_backend else "auto",
        trust_remote_code=True,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=2048,
        max_num_seqs=64,
        max_num_batched_tokens=131072,
        gpu_memory_utilization=0.85,
        seed=SEED,
        disable_log_stats=False,
        enable_prefix_caching=False,
    )
    engine_load_seconds = time.perf_counter() - engine_started
    results = []
    for pressure in PRESSURES:
        params = [
            SamplingParams(
                temperature=0.8,
                top_p=0.95,
                min_tokens=MIN_TOKENS,
                max_tokens=MAX_TOKENS,
                seed=SEED + index,
            )
            for index in range(pressure)
        ]
        started = time.perf_counter()
        outputs = llm.generate(prompts[:pressure], params, use_tqdm=False)
        elapsed = time.perf_counter() - started
        rows = []
        latencies = []
        ttfts = []
        generated_tokens = 0
        prompt_tokens = 0
        cap_hits = 0
        for index, output in enumerate(outputs):
            candidate = output.outputs[0]
            metrics = output.metrics
            arrival = metric_value(metrics, "arrival_time")
            first = metric_value(metrics, "first_token_time")
            finished = metric_value(metrics, "finished_time")
            if arrival is not None and finished is not None:
                latencies.append(finished - arrival)
            if arrival is not None and first is not None:
                ttfts.append(first - arrival)
            count = len(candidate.token_ids)
            generated_tokens += count
            prompt_tokens += len(output.prompt_token_ids)
            cap_hits += int(count >= MAX_TOKENS)
            rows.append(
                {
                    "prompt_id": prompts_payload[index]["prompt_id"],
                    "prompt_tokens": len(output.prompt_token_ids),
                    "generated_tokens": count,
                    "finish_reason": candidate.finish_reason,
                    "stop_reason": candidate.stop_reason,
                    "token_ids_sha256": hashlib.sha256(json.dumps(list(candidate.token_ids)).encode()).hexdigest(),
                    "arrival_time": arrival,
                    "first_token_time": first,
                    "finished_time": finished,
                }
            )
        results.append(
            {
                "pressure": pressure,
                "wall_seconds": elapsed,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "generated_tokens_per_second": generated_tokens / elapsed,
                "request_latency_p50_seconds": percentile(latencies, 0.50),
                "request_latency_p90_seconds": percentile(latencies, 0.90),
                "request_latency_p99_seconds": percentile(latencies, 0.99),
                "ttft_p50_seconds": percentile(ttfts, 0.50),
                "cap_hits": cap_hits,
                "requests": rows,
            }
        )
    dispatch = []
    if args.shs_backend and args.receipt.exists():
        dispatch = [json.loads(line) for line in args.receipt.read_text().splitlines()]
    write_json(
        args.output,
        {
            "cell": args.cell,
            "engine_load_seconds": engine_load_seconds,
            "pressures": results,
            "dispatch_receipts": dispatch,
        },
    )
    try:
        llm.llm_engine.shutdown()
    except Exception:
        pass
    return 0


def main(args) -> int:
    if not all((args.output_dir, args.base_model, args.shs_model, args.validation_jsonl)):
        raise ValueError("main mode requires output, base/SHS models, and validation JSONL")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = []
    with args.validation_jsonl.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= 64:
                break
            record = json.loads(line)
            prompts.append(
                {
                    "prompt_id": f"numina_val_{index:04d}",
                    "prompt": record["problem"] + "\n\nSolve carefully and give the final answer.",
                }
            )
    if len(prompts) != 64:
        raise RuntimeError(f"expected 64 prompts, found {len(prompts)}")
    prompt_path = output_dir / "prompt_manifest.json"
    write_json(prompt_path, prompts)
    prereg = {
        "run_id": RUN_ID,
        "status": "preregistered",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "pressures": list(PRESSURES),
        "prompt_manifest_sha256": sha256(prompt_path),
        "decoding": {"temperature": 0.8, "top_p": 0.95, "min_tokens": MIN_TOKENS, "max_tokens": MAX_TOKENS},
        "engine": {"tp": 1, "max_num_seqs": 64, "max_num_batched_tokens": 131072, "prefix_caching": False},
        "cells": ["naive_qwen", "shs_reference", "shs_triton_fast"],
        "strict_triton_cell": "unavailable: grouped fast kernel failed fixed A1 cosine gate",
        "claims": {"production_candidate": False, "production_ready": False},
    }
    write_json(output_dir / "preregistered_manifest.json", prereg)
    cells = (
        ("naive_qwen", args.base_model, None),
        ("shs_reference", args.shs_model, "reference"),
        ("shs_triton_fast", args.shs_model, "triton"),
    )
    results = {}
    failures = []
    for cell, model, backend in cells:
        output = output_dir / f"{cell}.json"
        receipt = output_dir / f"{cell}_dispatch.jsonl"
        result = json.loads(output.read_text()) if output.exists() else None
        if result is None or [row["pressure"] for row in result.get("pressures", [])] != list(PRESSURES):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--cell", cell,
                "--model", str(model),
                "--prompts", str(prompt_path),
                "--output", str(output),
                "--receipt", str(receipt),
            ]
            if backend:
                command.extend(["--shs-backend", backend])
            subprocess.run(command, check=True, env={**os.environ, "PYTHONPATH": str(args.source_root / "src")})
            result = json.loads(output.read_text())
        results[cell] = result
        if backend:
            actual = [row.get("backend") for row in result["dispatch_receipts"]]
            if actual != [backend] * 3:
                failures.append(f"{cell}:dispatch_mismatch:{actual}")
        if [row["pressure"] for row in result["pressures"]] != list(PRESSURES):
            failures.append(f"{cell}:pressure_incomplete")
    manifest = {**prereg, "status": "passed" if not failures else "failed", "results": results, "failures": failures}
    write_json(output_dir / "manifest.json", manifest)
    lines = ["# SHS Matched Long-Decode Matrix", "", f"Status: **{manifest['status']}**", ""]
    lines.extend(["| Cell | P1 tok/s | P8 tok/s | P16 tok/s | P32 tok/s | P64 tok/s |", "|---|---:|---:|---:|---:|---:|"])
    for cell, result in results.items():
        speeds = [row["generated_tokens_per_second"] for row in result["pressures"]]
        lines.append(f"| {cell} | " + " | ".join(f"{value:.1f}" for value in speeds) + " |")
    lines.extend(["", "Strict Triton is unavailable because the fast kernel failed the unchanged A1 parity gate."])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return 0 if not failures else 1


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--base-model", type=Path)
    result.add_argument("--shs-model", type=Path)
    result.add_argument("--validation-jsonl", type=Path)
    result.add_argument("--source-root", type=Path, default=Path.cwd())
    result.add_argument("--worker", action="store_true")
    result.add_argument("--cell")
    result.add_argument("--model", type=Path)
    result.add_argument("--prompts", type=Path)
    result.add_argument("--output", type=Path)
    result.add_argument("--receipt", type=Path)
    result.add_argument("--shs-backend")
    return result


if __name__ == "__main__":
    parsed = parser().parse_args()
    raise SystemExit(worker(parsed) if parsed.worker else main(parsed))
