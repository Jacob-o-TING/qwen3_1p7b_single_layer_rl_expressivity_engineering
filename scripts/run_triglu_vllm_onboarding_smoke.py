#!/usr/bin/env python3
"""Export, validate, and benchmark the TriGLU vLLM Transformers path."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_ID = "triglu_vllm_onboarding_smoke_20260712_v1"
METHODOLOGY_ID = "custom_ffn_vllm_onboarding_methodology_20260712_v1"
SEED = 20260712
SHORT_PRESSURES = (1, 8, 16)
MATCHED_PRESSURES = (16, 32, 64)
MATCHED_MIN_TOKENS = 800
MATCHED_MAX_TOKENS = 1024
MAX_NUM_BATCHED_TOKENS = 32768
GPU_MEMORY_UTILIZATION = 0.85
HISTORICAL_SHS_P64_TOKENS_PER_SECOND = 1892.1


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def preserve_attempt(path: Path) -> Path | None:
    if not path.exists():
        return None
    attempt = 1
    while True:
        candidate = path.with_name(f"{path.stem}_attempt{attempt}_failed{path.suffix}")
        if not candidate.exists():
            shutil.copy2(path, candidate)
            return candidate
        attempt += 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def tensor_metrics(left, right) -> dict[str, Any]:
    import torch

    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    difference = left - right
    denominator = torch.linalg.vector_norm(left).clamp_min(1.0e-12)
    return {
        "max_abs": float(difference.abs().max()),
        "relative_l2": float(torch.linalg.vector_norm(difference) / denominator),
        "cosine": float(torch.nn.functional.cosine_similarity(left, right, dim=0)),
        "top1_equal": int(left.argmax()) == int(right.argmax()),
    }


def load_prompts(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= 64:
                break
            record = json.loads(line)
            problem = record.get("problem") or record.get("prompt") or record.get("question")
            if not problem:
                raise KeyError(f"validation row {index} has no problem/prompt/question")
            rows.append(
                {
                    "prompt_id": f"numina_val_{index:04d}",
                    "prompt": str(problem) + "\n\nSolve carefully and give the final answer.",
                }
            )
    if len(rows) != 64:
        raise RuntimeError(f"expected 64 validation prompts, found {len(rows)}")
    return rows


def export_and_hf_parity(args, prompts: list[dict[str, str]]) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from qwen_single_layer_rl.config import load_config
    from qwen_single_layer_rl.model_surgery import build_variant
    from qwen_single_layer_rl.sft.checkpoint import load_trainable_state_dict
    from qwen_single_layer_rl.vllm.triglu_hf_model import (
        Qwen3TriGLUConfig,
        Qwen3TriGLUForCausalLM,
    )

    output_dir = args.output_dir.resolve()
    export_dir = output_dir / "deployment_export"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    cfg = load_config(args.config)
    params = dict(cfg["architecture_variant"]["params"])
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    torch.manual_seed(int(cfg["experiment"].get("init_seed", cfg["experiment"]["seed"])))
    model = build_variant(cfg).apply(model, cfg)
    trainable_state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    load_trainable_state_dict(model, trainable_state)

    base_dict = model.config.to_dict()
    for key in ("model_type", "architectures", "triglu_variant", "auto_map"):
        base_dict.pop(key, None)
    custom_config = Qwen3TriGLUConfig(triglu_variant=params, **base_dict)
    model.config = custom_config
    export_dir.mkdir(parents=True)
    model.save_pretrained(export_dir, safe_serialization=True, max_shard_size="4GB")
    custom_config.save_pretrained(export_dir)
    (export_dir / "configuration_qwen3_triglu.py").write_text(
        "from qwen_single_layer_rl.vllm.triglu_hf_model import Qwen3TriGLUConfig\n",
        encoding="utf-8",
    )
    (export_dir / "modeling_qwen3_triglu.py").write_text(
        "from qwen_single_layer_rl.vllm.triglu_hf_model import "
        "Qwen3TriGLUForCausalLM, Qwen3TriGLUModel\n",
        encoding="utf-8",
    )
    tokenizer.save_pretrained(export_dir)

    device = torch.device("cuda:0")
    inputs = tokenizer(prompts[0]["prompt"], return_tensors="pt").to(device)
    model = model.to(device).eval()
    with torch.no_grad():
        surgery_logits = model(**inputs).logits[:, -1, :].detach().cpu()
        surgery_tokens = model.generate(**inputs, max_new_tokens=8, do_sample=False)[0]
    surgery_new_tokens = surgery_tokens[inputs["input_ids"].shape[1] :].detach().cpu().tolist()
    del model, surgery_tokens
    gc.collect()
    torch.cuda.empty_cache()

    explicit, loading_info = Qwen3TriGLUForCausalLM.from_pretrained(
        export_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    wrapper = explicit.model.layers[10].mlp
    side_dtypes = sorted({str(parameter.dtype) for parameter in wrapper.triglu_side.parameters()})
    explicit = explicit.to(device).eval()
    with torch.no_grad():
        explicit_logits = explicit(**inputs).logits[:, -1, :].detach().cpu()
        explicit_tokens = explicit.generate(**inputs, max_new_tokens=8, do_sample=False)[0]
    explicit_new_tokens = explicit_tokens[inputs["input_ids"].shape[1] :].detach().cpu().tolist()
    metrics = tensor_metrics(surgery_logits, explicit_logits)
    result = {
        "checkpoint_sha256": sha256(args.checkpoint),
        "export_config_sha256": sha256(export_dir / "config.json"),
        "export_weight_bytes": sum(path.stat().st_size for path in export_dir.glob("*.safetensors")),
        "loading_info": {
            "missing_keys": loading_info.get("missing_keys", []),
            "unexpected_keys": loading_info.get("unexpected_keys", []),
            "mismatched_keys": loading_info.get("mismatched_keys", []),
            "error_msgs": loading_info.get("error_msgs", []),
        },
        "side_parameter_dtypes": side_dtypes,
        "logit_parity": metrics,
        "surgery_greedy_token_ids": surgery_new_tokens,
        "explicit_greedy_token_ids": explicit_new_tokens,
        "greedy_tokens_equal": surgery_new_tokens == explicit_new_tokens,
    }
    result["passed"] = bool(
        not any(result["loading_info"].values())
        and side_dtypes == ["torch.float32"]
        and metrics["cosine"] >= 0.9999
        and metrics["relative_l2"] <= 0.01
        and metrics["top1_equal"]
        and result["greedy_tokens_equal"]
    )
    write_json(output_dir / "hf_export_parity.json", result)
    del explicit, explicit_tokens, inputs
    gc.collect()
    torch.cuda.empty_cache()
    return result


def metric_value(metrics, name: str) -> float | None:
    value = getattr(metrics, name, None)
    return None if value is None else float(value)


def run_worker(args) -> int:
    os.environ["VLLM_USE_V1"] = "1"
    receipt = args.output.parent / "triglu_dispatch.jsonl"
    if args.cell == "triglu":
        receipt.unlink(missing_ok=True)
        os.environ["TRIGLU_DISPATCH_RECEIPT"] = str(receipt.resolve())

    import torch
    from vllm import LLM, SamplingParams

    prompts_payload = json.loads(args.prompts.read_text(encoding="utf-8"))
    prompts = [row["prompt"] for row in prompts_payload]
    model = args.triglu_model if args.cell == "triglu" else args.base_model
    started = time.perf_counter()
    llm = LLM(
        model=str(model.resolve()),
        # auto resolves the exported architecture through our plugin to the
        # custom TransformersModel subclass. Forcing "transformers" bypasses
        # ModelRegistry and selects the generic TransformersForCausalLM wrapper.
        model_impl="auto",
        trust_remote_code=True,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=2048,
        max_num_seqs=64,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        seed=SEED,
        disable_log_stats=False,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
    )
    engine_load_seconds = time.perf_counter() - started
    short_results = []
    for pressure in SHORT_PRESSURES:
        params = [SamplingParams(temperature=0.0, max_tokens=8, seed=SEED + index) for index in range(pressure)]
        started = time.perf_counter()
        outputs = llm.generate(prompts[:pressure], params, use_tqdm=False)
        elapsed = time.perf_counter() - started
        short_results.append(
            {
                "pressure": pressure,
                "wall_seconds": elapsed,
                "generated_tokens": sum(len(output.outputs[0].token_ids) for output in outputs),
                "first_request_token_ids": list(outputs[0].outputs[0].token_ids),
            }
        )

    matched_results = []
    for pressure in MATCHED_PRESSURES:
        params = [
            SamplingParams(
                temperature=0.8,
                top_p=0.95,
                min_tokens=MATCHED_MIN_TOKENS,
                max_tokens=MATCHED_MAX_TOKENS,
                seed=SEED + index,
            )
            for index in range(pressure)
        ]
        started = time.perf_counter()
        outputs = llm.generate(prompts[:pressure], params, use_tqdm=False)
        elapsed = time.perf_counter() - started
        lengths = [len(output.outputs[0].token_ids) for output in outputs]
        matched_results.append(
            {
                "pressure": pressure,
                "wall_seconds": elapsed,
                "generated_tokens": sum(lengths),
                "generated_tokens_per_second": sum(lengths) / elapsed,
                "mean_generated_tokens": sum(lengths) / len(lengths),
                "min_generated_tokens": min(lengths),
                "max_generated_tokens": max(lengths),
                "cap_hits": sum(length >= MATCHED_MAX_TOKENS for length in lengths),
                "token_trace_hashes": [
                    hashlib.sha256(json.dumps(list(output.outputs[0].token_ids)).encode()).hexdigest()
                    for output in outputs
                ],
            }
        )

    dispatch_rows = []
    if args.cell == "triglu" and receipt.is_file():
        dispatch_rows = [json.loads(line) for line in receipt.read_text(encoding="utf-8").splitlines() if line]
    result = {
        "cell": args.cell,
        "engine_load_seconds": engine_load_seconds,
        "short_pressures": short_results,
        "matched_pressures": matched_results,
        "dispatch_receipts": dispatch_rows,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024 * 1024),
    }
    write_json(args.output, result)
    try:
        llm.llm_engine.shutdown()
    except Exception:
        pass
    del llm
    gc.collect()
    return 0


def launch_workers(args, export_dir: Path, prompt_path: Path) -> dict[str, Any]:
    processes = []
    results = {}
    exit_codes = {}
    for gpu, cell in enumerate(("triglu", "vanilla")):
        output = args.output_dir / f"{cell}_worker.json"
        log_path = args.output_dir / f"{cell}_worker.log"
        if args.resume and output.is_file():
            existing = json.loads(output.read_text(encoding="utf-8"))
            if [row.get("pressure") for row in existing.get("matched_pressures", [])] == list(MATCHED_PRESSURES):
                results[cell] = existing
                exit_codes[cell] = 0
                continue
        if args.resume and log_path.is_file():
            attempt_log = preserve_attempt(log_path)
            if attempt_log is not None:
                log_path.unlink()
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--cell",
            cell,
            "--output",
            str(output),
            "--prompts",
            str(prompt_path),
            "--triglu-model",
            str(export_dir),
            "--base-model",
            str(args.base_model),
        ]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(args.source_root / "src")}
        handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
        processes.append((cell, process, handle, output))
    for cell, process, handle, _ in processes:
        exit_codes[cell] = process.wait()
        handle.close()
    results.update(
        {
            cell: json.loads(output.read_text(encoding="utf-8"))
            for cell, _, _, output in processes
            if output.is_file()
        }
    )
    return {"exit_codes": exit_codes, "results": results}


def main(args) -> int:
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(args.validation_jsonl)
    prompt_path = args.output_dir / "prompt_manifest.json"
    write_json(prompt_path, prompts)
    preregistered = {
        "run_id": RUN_ID,
        "methodology_id": METHODOLOGY_ID,
        "status": "preregistered",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "cells": ["triglu", "vanilla"],
        "prompt_manifest_sha256": sha256(prompt_path),
        "engine": {
            "tp": 1,
            "max_model_len": 2048,
            "max_num_seqs": 64,
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "enforce_eager": True,
        },
        "short_pressures": list(SHORT_PRESSURES),
        "matched": {
            "pressures": list(MATCHED_PRESSURES),
            "temperature": 0.8,
            "top_p": 0.95,
            "min_tokens": MATCHED_MIN_TOKENS,
            "max_tokens": MATCHED_MAX_TOKENS,
        },
        "scope": {"eval_parity_matrix": False, "production_grpo": False, "shutdown": False},
    }
    write_json(args.output_dir / "preregistered_manifest.json", preregistered)

    previous_manifest = args.output_dir / "manifest.json"
    if args.resume and previous_manifest.is_file():
        preserve_attempt(previous_manifest)
    parity_path = args.output_dir / "hf_export_parity.json"
    export_dir = args.output_dir / "deployment_export"
    if args.resume and parity_path.is_file() and export_dir.is_dir():
        hf_parity = json.loads(parity_path.read_text(encoding="utf-8"))
        if not hf_parity.get("passed"):
            raise RuntimeError("refusing to resume from a failed HF export parity artifact")
    else:
        hf_parity = export_and_hf_parity(args, prompts)
    workers = launch_workers(args, export_dir, prompt_path)
    failures = []
    if not hf_parity["passed"]:
        failures.append("hf_export_parity_failed")
    for cell in ("triglu", "vanilla"):
        if workers["exit_codes"].get(cell) != 0 or cell not in workers["results"]:
            failures.append(f"{cell}_worker_failed")
    if not failures:
        triglu = workers["results"]["triglu"]
        vanilla = workers["results"]["vanilla"]
        receipt_rows = triglu["dispatch_receipts"]
        valid_receipt = (
            len(receipt_rows) == 1
            and receipt_rows[0].get("variant") == "qwen_swiglu_triglu_side"
            and receipt_rows[0].get("backend") == "reference_pytorch_cublas"
            and receipt_rows[0].get("fallback") is False
        )
        if not valid_receipt:
            failures.append("triglu_dispatch_receipt_failed")
        if triglu["short_pressures"][0]["first_request_token_ids"] != hf_parity["explicit_greedy_token_ids"]:
            failures.append("hf_vllm_greedy_tokens_differ")
        for cell, result in workers["results"].items():
            if [row["pressure"] for row in result["matched_pressures"]] != list(MATCHED_PRESSURES):
                failures.append(f"{cell}_pressure_matrix_incomplete")

    comparison = None
    if not failures:
        triglu_p64 = workers["results"]["triglu"]["matched_pressures"][-1]
        vanilla_p64 = workers["results"]["vanilla"]["matched_pressures"][-1]
        comparison = {
            "triglu_p64_tokens_per_second_per_gpu": triglu_p64["generated_tokens_per_second"],
            "vanilla_p64_tokens_per_second_per_gpu": vanilla_p64["generated_tokens_per_second"],
            "triglu_fraction_of_vanilla": (
                triglu_p64["generated_tokens_per_second"] / vanilla_p64["generated_tokens_per_second"]
            ),
            "historical_shs_p64_tokens_per_second_per_gpu": HISTORICAL_SHS_P64_TOKENS_PER_SECOND,
            "triglu_fraction_of_historical_shs": (
                triglu_p64["generated_tokens_per_second"] / HISTORICAL_SHS_P64_TOKENS_PER_SECOND
            ),
            "matched_claim_limit": (
                "Generated-token throughput under bounded 800-1024 lengths; sampled traces may differ "
                "across architectures and are not hardware or semantic parity evidence."
            ),
        }

    manifest = {
        **preregistered,
        "status": "passed" if not failures else "failed",
        "hf_export_parity": hf_parity,
        "workers": workers,
        "comparison": comparison,
        "failures": failures,
        "production_authorization": {
            "triglu_vllm_onboarding": not failures,
            "fifty_batch_grpo": False,
            "reason": "This bounded gate does not authorize production GRPO.",
        },
    }
    write_json(args.output_dir / "manifest.json", manifest)
    lines = [
        "# TriGLU vLLM Onboarding Smoke",
        "",
        f"Status: **{manifest['status']}**",
        "",
        f"HF export parity: **{'PASS' if hf_parity['passed'] else 'FAIL'}**",
    ]
    if comparison:
        lines.extend(
            [
                "",
                "| Cell | P64 generated tok/s/GPU |",
                "|---|---:|",
                f"| TriGLU reference | {comparison['triglu_p64_tokens_per_second_per_gpu']:.1f} |",
                f"| Vanilla Qwen | {comparison['vanilla_p64_tokens_per_second_per_gpu']:.1f} |",
                f"| Historical SHS reference | {comparison['historical_shs_p64_tokens_per_second_per_gpu']:.1f} |",
                "",
                f"TriGLU/vanilla: **{comparison['triglu_fraction_of_vanilla']:.1%}**.",
            ]
        )
    lines.extend(["", "This run does not authorize 50/98-batch GRPO or an Eval Parity Matrix."])
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return 0 if not failures else 1


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--base-model", type=Path, required=True)
    result.add_argument("--config", type=Path)
    result.add_argument("--checkpoint", type=Path)
    result.add_argument("--validation-jsonl", type=Path)
    result.add_argument("--source-root", type=Path, default=Path.cwd())
    result.add_argument("--worker", action="store_true")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--cell", choices=("triglu", "vanilla"))
    result.add_argument("--output", type=Path)
    result.add_argument("--prompts", type=Path)
    result.add_argument("--triglu-model", type=Path)
    return result


if __name__ == "__main__":
    parsed = parser().parse_args()
    raise SystemExit(run_worker(parsed) if parsed.worker else main(parsed))
