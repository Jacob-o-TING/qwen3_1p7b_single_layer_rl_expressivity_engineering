#!/usr/bin/env python3
"""Bounded two-GPU GRPO prelaunch receipts for baseline and TriGLU."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUN_ID = "baseline_triglu_2x5090_grpo_prelaunch_gate_20260712_v1"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def repeated_ngram(text: str, n: int = 8, repeats: int = 3) -> bool:
    tokens = text.split()
    if len(tokens) < n * repeats:
        return False
    counts = Counter(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))
    return max(counts.values(), default=0) >= repeats


def classify_zero_reward(text: str, token_count: int, cap: int, finish_reason: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "empty"
    if finish_reason == "length" or token_count >= cap:
        return "token_cap"
    if repeated_ngram(stripped):
        return "repetition"
    if stripped.count("{") != stripped.count("}") or ("\\boxed{" in stripped and "}" not in stripped):
        return "invalid_latex"
    has_boxed = "\\boxed{" in stripped
    has_number_or_choice = bool(re.search(r"[-+]?\d|\b[A-E]\b", stripped))
    if not has_boxed and not has_number_or_choice:
        return "extraction_or_format"
    return "valid_but_wrong"


def summarize_deltas(left: list[list[float]], right: list[list[float]]) -> dict[str, Any]:
    deltas = [abs(a - b) for left_row, right_row in zip(left, right) for a, b in zip(left_row, right_row)]
    return {
        "count": len(deltas),
        "finite_count": sum(math.isfinite(value) for value in deltas),
        "mean_abs": sum(deltas) / max(1, len(deltas)),
        "max_abs": max(deltas, default=0.0),
    }


def ratio_summary(old: list[list[float]], current: list[list[float]], clip: float) -> dict[str, Any]:
    log_ratios = [b - a for old_row, current_row in zip(old, current) for a, b in zip(old_row, current_row)]
    ratios = [math.exp(max(-30.0, min(30.0, value))) for value in log_ratios]
    approx_kl = [(ratio - 1.0) - log_ratio for ratio, log_ratio in zip(ratios, log_ratios)]
    return {
        "count": len(ratios),
        "finite_count": sum(math.isfinite(value) for value in ratios),
        "min": min(ratios, default=1.0),
        "max": max(ratios, default=1.0),
        "mean": sum(ratios) / max(1, len(ratios)),
        "clip_fraction": sum(ratio < 1.0 - clip or ratio > 1.0 + clip for ratio in ratios) / max(1, len(ratios)),
        "approx_kl_mean": sum(approx_kl) / max(1, len(approx_kl)),
    }


def tensor_sha256(tensor: Any) -> str:
    cpu = tensor.detach().cpu().contiguous()
    payload = cpu.view(dtype=__import__("torch").uint8).numpy().tobytes()
    prefix = f"{cpu.dtype}|{tuple(cpu.shape)}|".encode()
    return sha256_bytes(prefix + payload)


def worker_load_and_hash(worker: Any, weights_path: str, expected_names: list[str], version: int) -> dict[str, Any]:
    import torch
    from safetensors import safe_open

    model = worker.model_runner.model
    named = dict(model.named_parameters())

    def resolve(name: str) -> tuple[str, Any]:
        if name in named:
            return name, named[name]
        candidates = [(key, value) for key, value in named.items() if key.endswith(name)]
        if len(candidates) != 1:
            raise RuntimeError(f"cannot uniquely resolve vLLM parameter {name}: {[key for key, _ in candidates]}")
        return candidates[0]

    before = {name: tensor_sha256(resolve(name)[1]) for name in expected_names}
    started = time.perf_counter()
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        weights = [(name, handle.get_tensor(name)) for name in handle.keys()]
    loaded = sorted(str(name) for name in model.load_weights(iter(weights)))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    named = dict(model.named_parameters())
    after = {name: tensor_sha256(resolve(name)[1]) for name in expected_names}
    dispatch = model.triglu_dispatch_state() if hasattr(model, "triglu_dispatch_state") else None
    return {
        "version": int(version),
        "load_seconds": elapsed,
        "loaded_count": len(loaded),
        "loaded_names": loaded,
        "before_hashes": before,
        "after_hashes": after,
        "dispatch": dispatch,
    }


def worker_model_state(worker: Any) -> dict[str, Any]:
    model = worker.model_runner.model
    return model.triglu_dispatch_state() if hasattr(model, "triglu_dispatch_state") else {}


def _serialize_outputs(requests: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prompt_index, request in enumerate(requests):
        for sample_index, output in enumerate(request.outputs):
            sampled_logprobs: list[float] = []
            for token_id, step in zip(output.token_ids, output.logprobs or []):
                entry = step.get(token_id)
                if entry is None:
                    raise RuntimeError(f"sampled token {token_id} absent from vLLM logprob payload")
                sampled_logprobs.append(float(entry.logprob))
            rows.append(
                {
                    "prompt_index": prompt_index,
                    "sample_index": sample_index,
                    "prompt_token_ids": list(request.prompt_token_ids),
                    "token_ids": list(output.token_ids),
                    "sampled_token_logprobs": sampled_logprobs,
                    "text": output.text,
                    "finish_reason": output.finish_reason,
                    "stop_reason": output.stop_reason,
                }
            )
    return rows


def rollout_worker(connection: Any, settings: dict[str, Any]) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(settings["rollout_gpu"])
    os.environ["VLLM_USE_V1"] = "1"
    # The worker callable is local trusted code sent over vLLM's same-host
    # control plane. vLLM 0.10.2 requires an explicit opt-in for this RPC.
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if settings["variant"] == "triglu":
        receipt = Path(settings["dispatch_receipt"])
        receipt.unlink(missing_ok=True)
        os.environ["TRIGLU_DISPATCH_RECEIPT"] = str(receipt)
    try:
        import torch
        from vllm import LLM, SamplingParams

        started = time.perf_counter()
        llm = LLM(
            model=settings["model"],
            model_impl="auto",
            trust_remote_code=True,
            tensor_parallel_size=1,
            enforce_eager=True,
            max_model_len=int(settings["max_model_len"]),
            max_num_seqs=int(settings["max_num_seqs"]),
            max_num_batched_tokens=int(settings["max_num_batched_tokens"]),
            gpu_memory_utilization=float(settings["gpu_memory_utilization"]),
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            seed=int(settings["seed"]),
        )
        connection.send({"type": "ready", "engine_load_seconds": time.perf_counter() - started})
        while True:
            request = connection.recv()
            command = request["command"]
            if command == "stop":
                break
            if command == "generate":
                params = [
                    SamplingParams(
                        n=int(request["group_size"]),
                        temperature=float(request["temperature"]),
                        top_p=float(request["top_p"]),
                        max_tokens=int(request["max_tokens"]),
                        seed=int(request["seed"]) + index,
                        logprobs=1,
                    )
                    for index in range(len(request["prompts"]))
                ]
                started = time.perf_counter()
                outputs = llm.generate(request["prompts"], params, use_tqdm=False)
                connection.send(
                    {
                        "type": "generated",
                        "wall_seconds": time.perf_counter() - started,
                        "rows": _serialize_outputs(outputs),
                    }
                )
            elif command == "greedy":
                params = [
                    SamplingParams(temperature=0.0, max_tokens=int(request["max_tokens"]), logprobs=1)
                    for _ in request["prompts"]
                ]
                outputs = llm.generate(request["prompts"], params, use_tqdm=False)
                connection.send({"type": "greedy", "rows": _serialize_outputs(outputs)})
            elif command == "load_weights":
                results = llm.collective_rpc(
                    worker_load_and_hash,
                    timeout=300,
                    args=(request["weights_path"], request["expected_names"], request["version"]),
                )
                connection.send({"type": "weight_sync", "workers": results})
            elif command == "model_state":
                connection.send({"type": "model_state", "workers": llm.collective_rpc(worker_model_state)})
            else:
                raise ValueError(f"unknown rollout command: {command}")
        try:
            llm.llm_engine.shutdown()
        except Exception:
            pass
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        connection.send({"type": "stopped"})
    except BaseException as exc:
        connection.send({"type": "error", "error": repr(exc)})
        raise
    finally:
        connection.close()


class RolloutProcess:
    def __init__(self, settings: dict[str, Any]) -> None:
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        self.connection = parent
        self.process = context.Process(target=rollout_worker, args=(child, settings), daemon=False)
        self.process.start()
        ready = self._receive(timeout=180)
        if ready["type"] != "ready":
            raise RuntimeError(f"rollout worker failed to initialize: {ready}")
        self.engine_load_seconds = float(ready["engine_load_seconds"])

    def _receive(self, timeout: float = 600) -> dict[str, Any]:
        if not self.connection.poll(timeout):
            raise TimeoutError("timed out waiting for rollout worker")
        payload = self.connection.recv()
        if payload.get("type") == "error":
            raise RuntimeError(payload["error"])
        return payload

    def request(self, payload: dict[str, Any], timeout: float = 600) -> dict[str, Any]:
        self.connection.send(payload)
        return self._receive(timeout)

    def stop(self) -> None:
        if self.process.is_alive():
            self.connection.send({"command": "stop"})
            self._receive(timeout=120)
        self.process.join(timeout=30)
        if self.process.exitcode not in (0, None):
            raise RuntimeError(f"rollout worker exited with {self.process.exitcode}")


def load_prompt_ledger(dataset: Path, tokenizer: Any, count: int) -> tuple[list[dict[str, Any]], str]:
    rows: list[dict[str, Any]] = []
    with dataset.open(encoding="utf-8") as handle:
        for index in range(count):
            source = json.loads(next(handle))
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": source["problem"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            row = {
                "global_index": index,
                "source": source.get("source"),
                "problem": source["problem"],
                "answer": str(source["answer"]),
                "rendered_prompt": rendered,
            }
            row["row_sha256"] = canonical_json_sha256(row)
            rows.append(row)
    return rows, canonical_json_sha256(rows)


def prepare_triglu_export(base_model: Path, cfg: dict[str, Any], output: Path, prompt: str) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from qwen_single_layer_rl.model_surgery import build_variant
    from qwen_single_layer_rl.vllm.triglu_hf_model import Qwen3TriGLUConfig

    if output.exists():
        shutil.rmtree(output)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    torch.manual_seed(int(cfg["experiment"]["init_seed"]))
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to("cuda:0").eval()
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        before = model(**inputs, use_cache=False).logits[:, -1].detach()
    # Variant injection creates new modules on CPU even when the backbone is
    # already resident on CUDA. Rebind the complete model before the no-op
    # parity pass while preserving the side branch's FP32 dtype.
    model = build_variant(cfg).apply(model, cfg).to("cuda:0")
    with torch.no_grad():
        after = model(**inputs, use_cache=False).logits[:, -1].detach()
    exact_noop = bool(torch.equal(before, after))
    max_abs = float((before.float() - after.float()).abs().max())
    base_dict = model.config.to_dict()
    for key in ("model_type", "architectures", "triglu_variant", "auto_map"):
        base_dict.pop(key, None)
    params = dict(cfg["architecture_variant"]["params"])
    custom_config = Qwen3TriGLUConfig(triglu_variant=params, **base_dict)
    model.config = custom_config
    output.mkdir(parents=True)
    model.save_pretrained(output, safe_serialization=True, max_shard_size="4GB")
    custom_config.save_pretrained(output)
    (output / "configuration_qwen3_triglu.py").write_text(
        "from qwen_single_layer_rl.vllm.triglu_hf_model import Qwen3TriGLUConfig\n", encoding="utf-8"
    )
    (output / "modeling_qwen3_triglu.py").write_text(
        "from qwen_single_layer_rl.vllm.triglu_hf_model import Qwen3TriGLUForCausalLM, Qwen3TriGLUModel\n",
        encoding="utf-8",
    )
    tokenizer.save_pretrained(output)
    weight_bytes = sum(path.stat().st_size for path in output.glob("*.safetensors"))
    del model, inputs, before, after
    gc.collect()
    torch.cuda.empty_cache()
    return {"exact_noop": exact_noop, "max_abs_logit_delta": max_abs, "weight_bytes": weight_bytes}


def load_actor(model_path: Path, variant: str, cfg: dict[str, Any], train: bool) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    from qwen_single_layer_rl.layers import apply_freeze_policy
    from qwen_single_layer_rl.vllm.triglu_hf_model import Qwen3TriGLUForCausalLM

    torch.manual_seed(int(cfg["experiment"]["init_seed"]))
    cls = Qwen3TriGLUForCausalLM if variant == "triglu" else AutoModelForCausalLM
    model = cls.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to("cuda:0")
    model.config.use_cache = False
    if train:
        apply_freeze_policy(model, cfg)
        model.train()
    else:
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()
    return model


def make_batch(rows: list[dict[str, Any]], pad_id: int) -> tuple[Any, Any, Any]:
    import torch

    sequences = [row["prompt_token_ids"] + row["token_ids"] for row in rows]
    max_length = max(map(len, sequences))
    input_ids = torch.full((len(rows), max_length), pad_id, dtype=torch.long, device="cuda:0")
    attention = torch.zeros_like(input_ids)
    completion_mask = torch.zeros((len(rows), max_length - 1), dtype=torch.bool, device="cuda:0")
    for index, (row, sequence) in enumerate(zip(rows, sequences)):
        input_ids[index, : len(sequence)] = torch.tensor(sequence, device="cuda:0")
        attention[index, : len(sequence)] = 1
        prompt_length = len(row["prompt_token_ids"])
        completion_mask[index, prompt_length - 1 : len(sequence) - 1] = True
    return input_ids, attention, completion_mask


def token_logprobs(model: Any, input_ids: Any, attention: Any, completion_mask: Any) -> tuple[Any, list[list[float]]]:
    import torch.nn.functional as functional

    logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits[:, :-1].float()
    targets = input_ids[:, 1:]
    values = functional.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    rows = [values[index][completion_mask[index]].detach().cpu().tolist() for index in range(values.shape[0])]
    return values, rows


def group_advantages(rewards: list[float], group_size: int) -> Any:
    import torch

    tensor = torch.tensor(rewards, device="cuda:0", dtype=torch.float32).view(-1, group_size)
    means = tensor.mean(dim=1, keepdim=True)
    std = tensor.std(dim=1, keepdim=True, unbiased=False)
    return ((tensor - means) / std.clamp_min(1.0e-8)).reshape(-1)


def train_step(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    batch: tuple[Any, Any, Any],
    old_rows: list[list[float]],
    ref_rows: list[list[float]],
    advantages: Any,
    clip: float,
    kl_beta: float,
) -> dict[str, Any]:
    import torch

    input_ids, attention, mask = batch
    old = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(row, device="cuda:0") for row in old_rows], batch_first=True
    )
    reference = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(row, device="cuda:0") for row in ref_rows], batch_first=True
    )
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    values, current_rows = token_logprobs(model, input_ids, attention, mask)
    current = torch.nn.utils.rnn.pad_sequence(
        [values[index][mask[index]] for index in range(values.shape[0])], batch_first=True
    )
    token_mask = torch.nn.utils.rnn.pad_sequence(
        [torch.ones(len(row), device="cuda:0", dtype=torch.bool) for row in old_rows], batch_first=True
    )
    ratios = torch.exp(current - old)
    token_advantages = advantages[:, None].expand_as(ratios)
    unclipped = ratios * token_advantages
    clipped = torch.clamp(ratios, 1.0 - clip, 1.0 + clip) * token_advantages
    policy_loss = -torch.minimum(unclipped, clipped)[token_mask].mean()
    kl_loss = ((current - reference).pow(2))[token_mask].mean()
    loss = policy_loss + kl_beta * kl_loss
    loss.backward()
    grad_norm = math.sqrt(
        sum(float(parameter.grad.float().norm()) ** 2 for parameter in model.parameters() if parameter.grad is not None)
    )
    optimizer.step()
    scheduler.step()
    torch.cuda.synchronize()
    return {
        "policy_loss": float(policy_loss.detach()),
        "kl_loss": float(kl_loss.detach()),
        "total_loss": float(loss.detach()),
        "grad_norm": grad_norm,
        "seconds": time.perf_counter() - started,
        "pre_update_current_logprobs": current_rows,
    }


def compare_trainables(left: Any, right: Any) -> tuple[float, list[str]]:
    right_named = dict(right.named_parameters())
    maximum = 0.0
    mismatched: list[str] = []
    for name, parameter in left.named_parameters():
        if not parameter.requires_grad:
            continue
        delta = float((parameter.detach().float() - right_named[name].detach().float()).abs().max())
        maximum = max(maximum, delta)
        if delta != 0.0:
            mismatched.append(name)
    return maximum, mismatched


def save_trainable_safetensors(model: Any, path: Path) -> dict[str, str]:
    from safetensors.torch import save_file

    tensors = {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    save_file(tensors, path)
    return {name: tensor_sha256(tensor) for name, tensor in tensors.items()}


def export_updated_model(model: Any, tokenizer: Any, output: Path) -> None:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    model.save_pretrained(output, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(output)
    (output / "configuration_qwen3_triglu.py").write_text(
        "from qwen_single_layer_rl.vllm.triglu_hf_model import Qwen3TriGLUConfig\n", encoding="utf-8"
    )
    (output / "modeling_qwen3_triglu.py").write_text(
        "from qwen_single_layer_rl.vllm.triglu_hf_model import Qwen3TriGLUForCausalLM, Qwen3TriGLUModel\n",
        encoding="utf-8",
    )


def reward_rows(rows: list[dict[str, Any]], ledger: list[dict[str, Any]], cap: int) -> tuple[list[float], dict[str, Any]]:
    from qwen_single_layer_rl.rewards.math_reward import binary_math_reward

    rewards: list[float] = []
    details: list[dict[str, Any]] = []
    categories: Counter[str] = Counter()
    wiring_failures: list[str] = []
    for index, row in enumerate(rows):
        target = ledger[row["prompt_index"]]["answer"]
        result = binary_math_reward(row["text"], target)
        reward = float(result.reward)
        rewards.append(reward)
        category = "correct" if reward > 0 else classify_zero_reward(
            row["text"], len(row["token_ids"]), cap, row["finish_reason"]
        )
        categories[category] += 1
        if reward not in (0.0, 1.0) or result.verifier != "verl_math_verify":
            wiring_failures.append(f"row_{index}")
        details.append(
            {
                "prompt_index": row["prompt_index"],
                "sample_index": row["sample_index"],
                "reward": reward,
                "category": category,
                "predicted": result.predicted,
                "target": result.target,
                "finish_reason": row["finish_reason"],
                "token_count": len(row["token_ids"]),
                "text_sha256": sha256_bytes(row["text"].encode()),
            }
        )
    return rewards, {
        "total": len(rows),
        "reward_one": sum(value == 1.0 for value in rewards),
        "reward_zero": sum(value == 0.0 for value in rewards),
        "categories": dict(sorted(categories.items())),
        "verifier_wiring_failures": wiring_failures,
        "details": details,
    }


def run_variant(
    variant: str,
    model_path: Path,
    cfg: dict[str, Any],
    ledger: list[dict[str, Any]],
    tokenizer: Any,
    output_dir: Path,
    rollout_gpu: int,
) -> dict[str, Any]:
    import torch

    from qwen_single_layer_rl.sft.checkpoint import load_latest_checkpoint, save_checkpoint

    gate = cfg["prelaunch_gate"]
    variant_dir = output_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    settings = {
        "variant": variant,
        "model": str(model_path.resolve()),
        "rollout_gpu": rollout_gpu,
        "dispatch_receipt": str((variant_dir / "dispatch.jsonl").resolve()),
        "max_model_len": gate["max_model_len"],
        "max_num_seqs": gate["max_num_seqs"],
        "max_num_batched_tokens": gate["max_num_batched_tokens"],
        "gpu_memory_utilization": gate["gpu_memory_utilization"],
        "seed": cfg["experiment"]["rollout_seed"],
    }
    rollout = RolloutProcess(settings)
    generated = rollout.request(
        {
            "command": "generate",
            "prompts": [row["rendered_prompt"] for row in ledger],
            "group_size": gate["group_size"],
            "temperature": gate["temperature"],
            "top_p": gate["top_p"],
            "max_tokens": gate["response_cap"],
            "seed": cfg["experiment"]["rollout_seed"],
        },
        timeout=900,
    )
    rows = generated["rows"]
    write_json(variant_dir / "rollout_rows.json", rows)
    rewards, reward_audit = reward_rows(rows, ledger, int(gate["response_cap"]))
    write_json(variant_dir / "reward_audit.json", reward_audit)

    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    actor = load_actor(model_path, variant, cfg, train=True)
    trainable_names = [name for name, parameter in actor.named_parameters() if parameter.requires_grad]
    trainable_shapes = {name: list(parameter.shape) for name, parameter in actor.named_parameters() if parameter.requires_grad}
    reference = load_actor(model_path, variant, cfg, train=False)
    batch = make_batch(rows, tokenizer.pad_token_id or tokenizer.eos_token_id)
    with torch.no_grad():
        _, old_rows = token_logprobs(actor, *batch)
        _, reference_rows = token_logprobs(reference, *batch)
    vllm_rows = [row["sampled_token_logprobs"] for row in rows]
    on_policy = {
        "rollout_vs_actor": summarize_deltas(vllm_rows, old_rows),
        "actor_vs_reference": summarize_deltas(old_rows, reference_rows),
        "old_logprobs": old_rows,
        "reference_logprobs": reference_rows,
    }
    del reference
    gc.collect()
    torch.cuda.empty_cache()

    optimizer = torch.optim.AdamW(
        [parameter for parameter in actor.parameters() if parameter.requires_grad],
        lr=float(gate["learning_rate"]),
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    prompts_per_update = int(gate["prompts_per_update"])
    group_size = int(gate["group_size"])
    split = prompts_per_update * group_size
    advantages = group_advantages(rewards, group_size)
    step1_batch = tuple(value[:split] for value in batch)
    step2_batch = tuple(value[split:] for value in batch)
    step1 = train_step(
        actor,
        optimizer,
        scheduler,
        step1_batch,
        old_rows[:split],
        reference_rows[:split],
        advantages[:split],
        float(gate["clip_range"]),
        float(gate["kl_coefficient"]),
    )
    checkpoint_root = variant_dir / "checkpoints"
    saved = save_checkpoint(
        checkpoint_root,
        model=actor,
        optimizer=optimizer,
        scheduler=scheduler,
        trainer_state={
            "global_step": 1,
            "epoch": 0,
            "global_prompt_cursor": prompts_per_update,
            "global_group_cursor": split,
            "global_order_sha256": canonical_json_sha256([row["row_sha256"] for row in ledger]),
            "consumed_prompt_ids": [row["row_sha256"] for row in ledger[:prompts_per_update]],
        },
        manifest={"run_id": RUN_ID, "variant": variant, "initialization": "untuned_base"},
        keep_last=1,
    )
    step2 = train_step(
        actor,
        optimizer,
        scheduler,
        step2_batch,
        old_rows[split:],
        reference_rows[split:],
        advantages[split:],
        float(gate["clip_range"]),
        float(gate["kl_coefficient"]),
    )
    with torch.no_grad():
        _, post_rows = token_logprobs(actor, *batch)
    post_ratio = ratio_summary(old_rows, post_rows, float(gate["clip_range"]))

    resumed = load_actor(model_path, variant, cfg, train=True)
    resumed_optimizer = torch.optim.AdamW(
        [parameter for parameter in resumed.parameters() if parameter.requires_grad],
        lr=float(gate["learning_rate"]),
        weight_decay=0.01,
    )
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(resumed_optimizer, lambda _: 1.0)
    resume_state = load_latest_checkpoint(
        checkpoint_root,
        model=resumed,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        device=torch.device("cuda:0"),
    )
    resumed_step2 = train_step(
        resumed,
        resumed_optimizer,
        resumed_scheduler,
        step2_batch,
        old_rows[split:],
        reference_rows[split:],
        advantages[split:],
        float(gate["clip_range"]),
        float(gate["kl_coefficient"]),
    )
    resume_delta, resume_mismatches = compare_trainables(actor, resumed)
    consumed_before = set(resume_state["consumed_prompt_ids"])
    consumed_after = {row["row_sha256"] for row in ledger[prompts_per_update:]}
    cursor_receipt = {
        "saved_state": resume_state,
        "resume_parameter_max_abs_delta": resume_delta,
        "resume_mismatched_parameters": resume_mismatches,
        "repeat_count": len(consumed_before & consumed_after),
        "covered_count": len(consumed_before | consumed_after),
        "expected_count": len(ledger),
        "optimizer_state_entries": len(resumed_optimizer.state),
        "scheduler_last_epoch": resumed_scheduler.last_epoch,
        "resumed_step2": resumed_step2,
        "checkpoint": str(saved),
    }
    del resumed, resumed_optimizer
    gc.collect()
    torch.cuda.empty_cache()

    sync_receipt = None
    oracle = None
    if variant == "triglu":
        weights_path = variant_dir / "live_sync_trainables.safetensors"
        sender_hashes = save_trainable_safetensors(actor, weights_path)
        sync_started = time.perf_counter()
        sync_payload = rollout.request(
            {
                "command": "load_weights",
                "weights_path": str(weights_path.resolve()),
                "expected_names": sorted(sender_hashes),
                "version": 2,
            },
            timeout=600,
        )
        sync_wall = time.perf_counter() - sync_started
        live_greedy = rollout.request(
            {
                "command": "greedy",
                "prompts": [row["rendered_prompt"] for row in ledger[:2]],
                "max_tokens": 16,
            }
        )["rows"]
        receiver = sync_payload["workers"][0]
        sync_receipt = {
            "sender_version": 2,
            "sender_hashes": sender_hashes,
            "receiver": receiver,
            "wall_seconds": sync_wall,
            "all_receiver_hashes_match": receiver["after_hashes"] == sender_hashes,
            "changed_receiver_tensor_count": sum(
                receiver["before_hashes"][name] != receiver["after_hashes"][name] for name in sender_hashes
            ),
            "live_greedy": live_greedy,
        }
        updated_export = variant_dir / "updated_export"
        export_updated_model(actor, tokenizer, updated_export)
        rollout.stop()
        del actor, optimizer
        gc.collect()
        torch.cuda.empty_cache()
        fresh_settings = {**settings, "model": str(updated_export.resolve())}
        fresh = RolloutProcess(fresh_settings)
        fresh_greedy = fresh.request(
            {
                "command": "greedy",
                "prompts": [row["rendered_prompt"] for row in ledger[:2]],
                "max_tokens": 16,
            }
        )["rows"]
        fresh.stop()
        live_tokens = [row["token_ids"] for row in live_greedy]
        fresh_tokens = [row["token_ids"] for row in fresh_greedy]
        live_logprobs = [row["sampled_token_logprobs"] for row in live_greedy]
        fresh_logprobs = [row["sampled_token_logprobs"] for row in fresh_greedy]
        oracle = {
            "live_tokens": live_tokens,
            "fresh_tokens": fresh_tokens,
            "greedy_tokens_equal": live_tokens == fresh_tokens,
            "logprob_delta": summarize_deltas(live_logprobs, fresh_logprobs),
        }
    else:
        rollout.stop()
        del actor, optimizer
        gc.collect()
        torch.cuda.empty_cache()

    acceptance = gate["acceptance"]
    failures: list[str] = []
    if on_policy["rollout_vs_actor"]["finite_count"] != on_policy["rollout_vs_actor"]["count"]:
        failures.append("nonfinite_rollout_actor_logprobs")
    if on_policy["rollout_vs_actor"]["mean_abs"] > float(acceptance["rollout_actor_mean_abs_logprob_delta_max"]):
        failures.append("rollout_actor_mean_logprob_delta")
    if on_policy["rollout_vs_actor"]["max_abs"] > float(acceptance["rollout_actor_max_abs_logprob_delta_max"]):
        failures.append("rollout_actor_max_logprob_delta")
    if reward_audit["verifier_wiring_failures"]:
        failures.append("reward_verifier_wiring")
    if not trainable_names or max(step1["grad_norm"], step2["grad_norm"]) == 0.0:
        failures.append("actor_update_missing")
    if variant == "triglu" and not any(".triglu_side." in name for name in trainable_names):
        failures.append("triglu_side_trainables_missing")
    if resume_delta != float(acceptance["resume_parameter_max_abs_delta"]):
        failures.append("resume_parameter_mismatch")
    if cursor_receipt["repeat_count"] or cursor_receipt["covered_count"] != cursor_receipt["expected_count"]:
        failures.append("resume_cursor_repeat_or_skip")
    if variant == "triglu":
        if not sync_receipt["all_receiver_hashes_match"]:
            failures.append("live_sync_receiver_hash_mismatch")
        if not oracle["greedy_tokens_equal"]:
            failures.append("live_sync_fresh_reload_token_mismatch")
    result = {
        "variant": variant,
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "model_path": str(model_path.resolve()),
        "engine_load_seconds": rollout.engine_load_seconds,
        "generation_seconds": generated["wall_seconds"],
        "on_policy": on_policy,
        "post_update_ratio": post_ratio,
        "reward_audit": reward_audit,
        "trainable_contract": {
            "count": len(trainable_names),
            "names": trainable_names,
            "shapes": trainable_shapes,
        },
        "updates": {"step1": step1, "step2": step2},
        "resume": cursor_receipt,
        "live_weight_sync": sync_receipt,
        "direct_sync_vs_fresh_reload": oracle,
        "peak_actor_allocated_gib": torch.cuda.max_memory_allocated(0) / (1024**3),
        "peak_actor_reserved_gib": torch.cuda.max_memory_reserved(0) / (1024**3),
        "ready_for_8gpu_production_shaped_canary": not failures,
    }
    write_json(variant_dir / "result.json", result)
    return result


def main(args: argparse.Namespace) -> int:
    import torch
    from transformers import AutoTokenizer

    from qwen_single_layer_rl.config import load_config
    from qwen_single_layer_rl.seeding import seed_everything

    source_root = args.source_root.resolve()
    cfg = load_config(args.config)
    gate = cfg["prelaunch_gate"]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_model = Path(gate["base_model"]).resolve()
    dataset = (source_root / gate["dataset_jsonl"]).resolve()
    selection_ledger = (source_root / gate["selection_ledger"]).resolve()
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    ledger, ledger_sha = load_prompt_ledger(dataset, tokenizer, int(gate["num_prompts"]))
    write_json(output_dir / "prompt_group_ledger.json", {"sha256": ledger_sha, "rows": ledger})
    preregistered = {
        "run_id": RUN_ID,
        "status": "running",
        "started_at": utc_now(),
        "config": str(args.config.resolve()),
        "base_model": str(base_model),
        "base_model_config_sha256": sha256_file(base_model / "config.json"),
        "dataset": str(dataset),
        "selection_ledger": str(selection_ledger),
        "selection_ledger_sha256": sha256_file(selection_ledger),
        "prompt_group_ledger_sha256": ledger_sha,
        "scientific_contract": {
            "initialization": "same untuned Qwen3-1.7B-Base revision",
            "variants": ["baseline", "triglu exact-noop"],
            "group_size": gate["group_size"],
            "seed": cfg["experiment"]["seed"],
            "production_98_batches_authorized": False,
        },
        "pending_obligations": cfg["pending_obligations_carried_forward"],
    }
    write_json(output_dir / "preregistered_manifest.json", preregistered)
    seed_everything(int(cfg["experiment"]["seed"]))
    torch.cuda.set_device(0)
    triglu_export = output_dir / "triglu_exact_noop_export"
    export_receipt = prepare_triglu_export(
        base_model, cfg, triglu_export, ledger[0]["rendered_prompt"]
    )
    write_json(output_dir / "triglu_exact_noop_export_receipt.json", export_receipt)
    if not export_receipt["exact_noop"]:
        raise RuntimeError(f"TriGLU exact-noop initialization failed: {export_receipt}")

    results: dict[str, Any] = {}
    for variant, model_path in (("baseline", base_model), ("triglu", triglu_export)):
        print(f"PRELAUNCH_VARIANT_START variant={variant} time={utc_now()}", flush=True)
        results[variant] = run_variant(
            variant,
            model_path,
            cfg,
            ledger,
            tokenizer,
            output_dir,
            int(cfg["runtime"]["rollout_gpu"]),
        )
        print(
            f"PRELAUNCH_VARIANT_END variant={variant} status={results[variant]['status']} "
            f"ready_8gpu={results[variant]['ready_for_8gpu_production_shaped_canary']}",
            flush=True,
        )
    failures = [f"{variant}:{failure}" for variant, result in results.items() for failure in result["failures"]]
    manifest = {
        **preregistered,
        "status": "passed" if not failures else "failed",
        "ended_at": utc_now(),
        "triglu_exact_noop_export": export_receipt,
        "variants": results,
        "failures": failures,
        "ready_for_later_8gpu_production_shaped_canary": not failures,
        "production_98_batches_authorized": False,
    }
    write_json(output_dir / "manifest.json", manifest)
    print(
        f"PRELAUNCH_GATE_END status={manifest['status']} failures={len(failures)} "
        f"ready_8gpu={manifest['ready_for_later_8gpu_production_shaped_canary']}",
        flush=True,
    )
    return 0 if not failures else 2


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--source-root", type=Path, default=Path.cwd())
    result.add_argument("--output-dir", type=Path, required=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main(parser().parse_args()))
