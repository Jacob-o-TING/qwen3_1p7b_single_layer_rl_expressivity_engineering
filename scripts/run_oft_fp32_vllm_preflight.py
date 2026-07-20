#!/usr/bin/env python3
"""Bounded HF/vLLM gate for an exact-identity FP32 OFT export."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path


PROMPTS = (
    "Compute 17 * 23. Give only the final answer.",
    "Solve 3x + 7 = 31. Give only the final answer.",
    "What is the remainder when 2^10 is divided by 7? Give only the final answer.",
    "A rectangle has sides 9 and 14. What is its area? Give only the final answer.",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = args.model.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = output.with_name("oft_dispatch.jsonl")
    receipt.unlink(missing_ok=True)
    os.environ["OFT_DISPATCH_RECEIPT"] = str(receipt)
    os.environ["VLLM_USE_V1"] = "1"

    raw_config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    if raw_config.get("model_type") != "qwen3_oft":
        raise RuntimeError(f"unexpected model_type: {raw_config.get('model_type')}")
    params = raw_config.get("oft_variant", {})
    if params.get("target_layers") != [10] or not params.get("fp32_compute"):
        raise RuntimeError(f"unexpected OFT contract: {params}")
    if params.get("target_modules") != ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
        raise RuntimeError("OFT export is not SwiGLU-only")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    hf = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to("cuda:0").eval()
    wrapper = hf.model.layers[10].mlp
    oft_parameters = [
        parameter
        for module in (wrapper.oft_gate, wrapper.oft_up, wrapper.oft_down)
        for parameter in module.oft_like.parameters()
    ]
    if len(oft_parameters) != 3 or {parameter.dtype for parameter in oft_parameters} != {torch.float32}:
        raise RuntimeError("HF OFT parameters do not satisfy the three-FP32-rotation contract")
    if not wrapper.fp32_compute:
        raise RuntimeError("HF OFT wrapper is not using FP32 transform compute")

    hf_tokens: list[list[int]] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for prompt in PROMPTS:
            encoded = tokenizer(prompt, return_tensors="pt").to("cuda:0")
            generated = hf.generate(**encoded, do_sample=False, max_new_tokens=8)
            hf_tokens.append(generated[0, encoded.input_ids.shape[1] :].cpu().tolist())
    hf_seconds = time.perf_counter() - started
    del hf
    gc.collect()
    torch.cuda.empty_cache()

    from vllm import LLM, SamplingParams

    started = time.perf_counter()
    llm = LLM(
        model=str(model_path),
        model_impl="auto",
        trust_remote_code=True,
        tensor_parallel_size=1,
        enforce_eager=True,
        dtype="bfloat16",
        max_model_len=4096,
        max_num_seqs=16,
        max_num_batched_tokens=32768,
        gpu_memory_utilization=0.85,
        seed=20260707,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
    )
    load_seconds = time.perf_counter() - started
    started = time.perf_counter()
    outputs = llm.generate(
        list(PROMPTS),
        [SamplingParams(temperature=0.0, max_tokens=8, seed=20260707 + index) for index in range(len(PROMPTS))],
        use_tqdm=False,
    )
    vllm_seconds = time.perf_counter() - started
    vllm_tokens = [list(item.outputs[0].token_ids) for item in outputs]

    rows = [json.loads(line) for line in receipt.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"expected one OFT dispatch receipt, found {len(rows)}")
    dispatch = rows[0]
    checks = {
        "hf_vllm_greedy_tokens_equal": hf_tokens == vllm_tokens,
        "dispatch_variant": dispatch.get("variant") == "qwen_swiglu_oft",
        "dispatch_backend": dispatch.get("backend") == "reference_pytorch_cublas",
        "dispatch_layer10": dispatch.get("layer_index") == 10,
        "dispatch_fp32_compute": dispatch.get("fp32_compute") is True,
        "dispatch_fp32_parameters": dispatch.get("parameter_dtypes") == ["torch.float32"],
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "model": str(model_path),
        "prompts": len(PROMPTS),
        "checks": checks,
        "hf_token_ids": hf_tokens,
        "vllm_token_ids": vllm_tokens,
        "hf_generation_seconds": hf_seconds,
        "vllm_engine_load_seconds": load_seconds,
        "vllm_generation_seconds": vllm_seconds,
        "dispatch": dispatch,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
