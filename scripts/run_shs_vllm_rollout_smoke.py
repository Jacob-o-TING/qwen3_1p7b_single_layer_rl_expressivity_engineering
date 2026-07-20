#!/usr/bin/env python3
"""Run HF parity and vLLM eager continuous-batching smoke for SHS."""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoTokenizer

from qwen_single_layer_rl.model_surgery.qwen_swiglu_variant_modules import ShuffledHyperGridDeltaLinear
from qwen_single_layer_rl.vllm.shs_hf_model import Qwen3SHSForCausalLM


RUN_ID = "shs_vllm_contbatch_rollout_smoke_20260712_v1"
SEED = 20260712
PRESSURES = [1, 8, 16, 32]
CAPS = [4, 8, 12, 16]
BASE_PROMPTS = [
    "Solve and give only the final answer: {a} + {b} =",
    "Compute exactly: {a} * {b} =",
    "What integer is one more than {a}? Answer:",
    "A box has {a} red balls and {b} blue balls. Total balls:",
]
PROMPTS = [
    BASE_PROMPTS[index % len(BASE_PROMPTS)].format(a=11 + index, b=3 + (index % 7))
    + (" Think carefully." * (index % 3))
    for index in range(32)
]
PROMPT_IDS = [f"runtime_math_{index:02d}" for index in range(32)]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def gpu_memory_mib() -> int:
    text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
    ).strip()
    return int(text.splitlines()[0])


def set_reference_backend(model) -> None:
    modules = [m for m in model.modules() if isinstance(m, ShuffledHyperGridDeltaLinear)]
    if len(modules) != 3:
        raise RuntimeError(f"expected 3 SHS projections, found {len(modules)}")
    for module in modules:
        module.set_inference_mul_backend("reference")


def metric_value(metrics, name: str):
    value = getattr(metrics, name, None)
    return None if value is None else float(value)


def run() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pressures", type=int, nargs="+", default=PRESSURES)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    model_dir = args.model.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt = output_dir / "dispatch.jsonl"
    receipt.unlink(missing_ok=True)
    prereg = {
        "run_id": RUN_ID,
        "status": "preregistered",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "prompt_ids": PROMPT_IDS,
        "prompts": PROMPTS,
        "response_caps": [CAPS[index % 4] for index in range(32)],
        "pressures": args.pressures,
        "backend": {
            "vllm": "0.10.2 Transformers backend V1 enforce-eager",
            "model_impl": "transformers",
            "tensor_parallel_size": 1,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "chunked_prefill": True,
            "enable_prefix_caching": False,
            "shs_mul": "triton split-base delta",
            "shs_add": "reference",
        },
        "decoding": {"temperature": 0.0, "seed": SEED, "stop_token_ids": [151643]},
        "model": str(model_dir),
    }
    write_json(output_dir / "preregistered_manifest.json", prereg)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    hf_started = time.perf_counter()
    hf = Qwen3SHSForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()
    set_reference_backend(hf)
    one_inputs = tokenizer(PROMPTS[0], return_tensors="pt").to("cuda")
    with torch.no_grad():
        hf_tokens = hf.generate(**one_inputs, max_new_tokens=CAPS[0], do_sample=False)[0]
    hf_new_tokens = hf_tokens[one_inputs["input_ids"].shape[1] :].cpu().tolist()
    torch.cuda.synchronize()
    hf_seconds = time.perf_counter() - hf_started
    del hf, one_inputs, hf_tokens
    gc.collect()
    torch.cuda.empty_cache()

    os.environ["SHS_DISPATCH_RECEIPT"] = str(receipt)
    os.environ["VLLM_USE_V1"] = "1"
    from vllm import LLM, SamplingParams

    engine_started = time.perf_counter()
    llm = LLM(
        model=str(model_dir),
        model_impl="transformers",
        trust_remote_code=True,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=SEED,
        disable_log_stats=False,
        enable_prefix_caching=False,
    )
    engine_load_seconds = time.perf_counter() - engine_started
    pressure_results = []
    one_vllm_tokens = None
    for pressure in args.pressures:
        if pressure < 1 or pressure > len(PROMPTS):
            raise ValueError(f"pressure must be in [1, {len(PROMPTS)}], got {pressure}")
        prompts = PROMPTS[:pressure]
        prefill_started = time.perf_counter()
        prefill_outputs = llm.generate(
            prompts,
            [SamplingParams(temperature=0.0, max_tokens=1, seed=SEED + index) for index in range(pressure)],
            use_tqdm=False,
        )
        prefill_probe_seconds = time.perf_counter() - prefill_started
        prefill_prompt_tokens = sum(len(output.prompt_token_ids) for output in prefill_outputs)
        params = [
            SamplingParams(temperature=0.0, max_tokens=CAPS[index % 4], seed=SEED + index)
            for index in range(pressure)
        ]
        started = time.perf_counter()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        elapsed = time.perf_counter() - started
        if pressure == 1:
            one_vllm_tokens = list(outputs[0].outputs[0].token_ids)
        prompt_tokens = sum(len(output.prompt_token_ids) for output in outputs)
        generated_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
        decode_residual_seconds = max(0.0, elapsed - prefill_probe_seconds)
        decode_residual_tokens = max(0, generated_tokens - pressure)
        ttfts = []
        decode_latencies = []
        rows = []
        for index, output in enumerate(outputs):
            metrics = output.metrics
            first = metric_value(metrics, "first_token_time")
            arrival = metric_value(metrics, "arrival_time")
            finished = metric_value(metrics, "finished_time")
            if first is not None and arrival is not None:
                ttfts.append(first - arrival)
            if first is not None and finished is not None:
                decode_latencies.append(finished - first)
            rows.append(
                {
                    "prompt_id": PROMPT_IDS[index],
                    "prompt_tokens": len(output.prompt_token_ids),
                    "generated_tokens": len(output.outputs[0].token_ids),
                    "token_ids": list(output.outputs[0].token_ids),
                    "finish_reason": output.outputs[0].finish_reason,
                    "stop_reason": output.outputs[0].stop_reason,
                    "arrival_time": arrival,
                    "first_token_time": first,
                    "finished_time": finished,
                }
            )
        pressure_results.append(
            {
                "pressure": pressure,
                "wall_seconds": elapsed,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "prefill_probe_method": "separate uncached max_tokens=1 request batch; includes first decode token",
                "prefill_probe_seconds": prefill_probe_seconds,
                "prefill_prompt_tokens": prefill_prompt_tokens,
                "prefill_tokens_per_second_proxy": prefill_prompt_tokens / prefill_probe_seconds,
                "decode_residual_method": "full generation wall minus uncached one-token prefill probe wall",
                "decode_residual_seconds": decode_residual_seconds,
                "decode_residual_tokens": decode_residual_tokens,
                "decode_tokens_per_second_proxy": (
                    decode_residual_tokens / decode_residual_seconds if decode_residual_seconds > 0 else None
                ),
                "end_to_end_generated_tokens_per_second": generated_tokens / elapsed,
                "ttft_median_seconds": statistics.median(ttfts) if ttfts else None,
                "decode_latency_median_seconds": statistics.median(decode_latencies) if decode_latencies else None,
                "gpu_memory_used_mib": gpu_memory_mib(),
                "requests": rows,
            }
        )

    dispatch_rows = [json.loads(line) for line in receipt.read_text(encoding="utf-8").splitlines()]
    manifest = {
        **prereg,
        "status": "passed",
        "hf_reference": {"load_and_generate_seconds": hf_seconds, "one_prompt_new_token_ids": hf_new_tokens},
        "vllm": {
            "engine_load_seconds": engine_load_seconds,
            "one_prompt_new_token_ids": one_vllm_tokens,
            "one_prompt_tokens_equal_hf": one_vllm_tokens == hf_new_tokens,
            "pressure_results": pressure_results,
        },
        "dispatch_receipts": dispatch_rows,
        "dispatch_pass": len(dispatch_rows) == 3 and all(row["backend"] == "triton" for row in dispatch_rows),
    }
    failures = []
    if not manifest["vllm"]["one_prompt_tokens_equal_hf"]:
        failures.append("one_prompt_tokens_differ")
    if not manifest["dispatch_pass"]:
        failures.append("triton_dispatch_receipt_incomplete")
    if [row["pressure"] for row in pressure_results] != args.pressures:
        failures.append("pressure_matrix_incomplete")
    manifest["failures"] = failures
    manifest["status"] = "passed" if not failures else "failed"
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    try:
        llm.llm_engine.shutdown()
    except Exception:
        pass
    del llm
    gc.collect()
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(run())
