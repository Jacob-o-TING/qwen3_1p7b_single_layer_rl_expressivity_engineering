#!/usr/bin/env python3
"""Tiny real GRPO plumbing smoke with vLLM rollout and actor update."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from qwen_single_layer_rl.config import load_config
from qwen_single_layer_rl.layers import apply_freeze_policy
from qwen_single_layer_rl.model_surgery.qwen_swiglu_variant_modules import ShuffledHyperGridDeltaLinear
from qwen_single_layer_rl.rewards.math_reward import extract_answer
from qwen_single_layer_rl.sft.checkpoint import load_latest_checkpoint, save_checkpoint
from qwen_single_layer_rl.vllm.shs_hf_model import Qwen3SHSForCausalLM


RUN_ID = "shs_triton_grpo_onebatch_smoke_20260712_v1"
SEED = 20260712
PROMPT_ID = "grpo_numeric_37x19_v1"
PROMPT = "Compute 37 * 19. Give the final integer answer."
TARGET = 703.0
GROUP_SIZE = 4
RESPONSE_CAP = 16
LR = 5.0e-6
CLIP = 0.2
KL_BETA = 0.001


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def worker_rollout(args) -> int:
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["SHS_DISPATCH_RECEIPT"] = str(args.receipt.resolve())
    args.receipt.unlink(missing_ok=True)
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(args.model.resolve()),
        model_impl="transformers",
        trust_remote_code=True,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=256,
        max_num_seqs=max(1, args.n),
        max_num_batched_tokens=256,
        gpu_memory_utilization=0.20,
        enable_prefix_caching=False,
        seed=SEED,
    )
    params = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=args.max_tokens,
        seed=SEED,
    )
    started = time.perf_counter()
    request = llm.generate([args.prompt], params, use_tqdm=False)[0]
    elapsed = time.perf_counter() - started
    rows = [
        {
            "index": index,
            "token_ids": list(output.token_ids),
            "text": output.text,
            "finish_reason": output.finish_reason,
            "stop_reason": output.stop_reason,
        }
        for index, output in enumerate(request.outputs)
    ]
    write_json(
        args.output,
        {
            "prompt": args.prompt,
            "prompt_token_ids": list(request.prompt_token_ids),
            "outputs": rows,
            "wall_seconds": elapsed,
        },
    )
    return 0


def parse_reward(text: str, token_ids: list[int]) -> dict:
    extracted = extract_answer(text)
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", extracted)
    predicted = float(matches[-1]) if matches else None
    numeric = 0.0 if predicted is None else 1.0 / (1.0 + abs(predicted - TARGET))
    diversity = len(set(token_ids)) / max(1, len(token_ids))
    reward = numeric + 0.001 * diversity
    return {
        "reward": reward,
        "numeric_distance_reward": numeric,
        "lexical_diversity_smoke_shaping": 0.001 * diversity,
        "extracted": extracted,
        "predicted_numeric": predicted,
        "target": TARGET,
    }


def configure_reference(model) -> None:
    modules = [m for m in model.modules() if isinstance(m, ShuffledHyperGridDeltaLinear)]
    if len(modules) != 3:
        raise RuntimeError(f"expected 3 SHS projections, found {len(modules)}")
    for module in modules:
        module.set_inference_mul_backend("reference")


def load_actor(model_dir: Path, cfg: dict, *, train: bool):
    model = Qwen3SHSForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda()
    configure_reference(model)
    model.config.use_cache = False
    if train:
        apply_freeze_policy(model, cfg)
        model.train()
    else:
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()
    return model


def make_batch(prompt_ids: list[int], completions: list[list[int]], pad_id: int):
    sequences = [prompt_ids + completion for completion in completions]
    max_length = max(map(len, sequences))
    input_ids = torch.full((len(sequences), max_length), pad_id, dtype=torch.long, device="cuda")
    attention = torch.zeros_like(input_ids)
    completion_mask = torch.zeros((len(sequences), max_length - 1), dtype=torch.bool, device="cuda")
    for row, sequence in enumerate(sequences):
        input_ids[row, : len(sequence)] = torch.tensor(sequence, device="cuda")
        attention[row, : len(sequence)] = 1
        completion_mask[row, len(prompt_ids) - 1 : len(sequence) - 1] = True
    return input_ids, attention, completion_mask


def sequence_logprobs(model, input_ids, attention, completion_mask):
    logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits[:, :-1].float()
    targets = input_ids[:, 1:]
    token_logprobs = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (token_logprobs * completion_mask).sum(-1) / completion_mask.sum(-1).clamp_min(1)


def parameter_difference(left, right) -> float:
    right_params = dict(right.named_parameters())
    return max(
        (parameter.detach().float() - right_params[name].detach().float()).abs().max().item()
        for name, parameter in left.named_parameters()
        if parameter.requires_grad
    )


def export_updated(model, tokenizer, output: Path) -> None:
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    state = {
        name: tensor
        for name, tensor in model.state_dict().items()
        if not name.endswith(("_row_block_ids", "_col_block_ids"))
    }
    model.save_pretrained(output, state_dict=state, safe_serialization=True, max_shard_size="4GB")
    model.config.architectures = ["Qwen3SHSForCausalLM"]
    model.config.auto_map = {
        "AutoConfig": "configuration_qwen3_shs.Qwen3SHSConfig",
        "AutoModel": "modeling_qwen3_shs.Qwen3SHSModel",
        "AutoModelForCausalLM": "modeling_qwen3_shs.Qwen3SHSForCausalLM",
    }
    model.config.save_pretrained(output)
    tokenizer.save_pretrained(output)
    (output / "configuration_qwen3_shs.py").write_text(
        "from qwen_single_layer_rl.vllm.shs_hf_model import Qwen3SHSConfig\n", encoding="utf-8"
    )
    (output / "modeling_qwen3_shs.py").write_text(
        "from qwen_single_layer_rl.vllm.shs_hf_model import Qwen3SHSForCausalLM, Qwen3SHSModel\n",
        encoding="utf-8",
    )


def main(args) -> int:
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
        "prompt_id": PROMPT_ID,
        "prompt": PROMPT,
        "target": TARGET,
        "group_size": GROUP_SIZE,
        "response_cap": RESPONSE_CAP,
        "rollout": {"backend": "vLLM 0.10.2 Transformers V1 eager", "temperature": 0.8, "top_p": 0.95},
        "reward": "numeric distance 1/(1+abs(error)) + 0.001 lexical-diversity smoke shaping",
        "objective": {"algorithm": "single-group GRPO/PPO clipped ratio", "clip": CLIP, "kl_beta": KL_BETA},
        "actor_backend": "reference training path; Triton inference-only",
        "weight_sync": "full updated export plus vLLM engine rebuild",
        "model": str(args.model.resolve()),
        "config": str(args.config.resolve()),
    }
    write_json(output_dir / "preregistered_manifest.json", prereg)

    worker_output = output_dir / "rollout_group.json"
    worker_receipt = output_dir / "rollout_dispatch.jsonl"
    worker_cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-rollout",
        "--model", str(args.model.resolve()),
        "--output", str(worker_output),
        "--receipt", str(worker_receipt),
        "--prompt", PROMPT,
        "--n", str(GROUP_SIZE),
        "--temperature", "0.8",
        "--max-tokens", str(RESPONSE_CAP),
    ]
    rollout_started = time.perf_counter()
    subprocess.run(worker_cmd, check=True, env={**os.environ, "PYTHONPATH": str(args.source_root / "src")})
    rollout_wall = time.perf_counter() - rollout_started
    rollout = json.loads(worker_output.read_text(encoding="utf-8"))
    dispatch = [json.loads(line) for line in worker_receipt.read_text(encoding="utf-8").splitlines()]
    completions = [row["token_ids"] for row in rollout["outputs"]]
    rewards = [parse_reward(row["text"], row["token_ids"]) for row in rollout["outputs"]]
    reward_tensor = torch.tensor([row["reward"] for row in rewards], device="cuda", dtype=torch.float32)
    advantages = (reward_tensor - reward_tensor.mean()) / reward_tensor.std(unbiased=False).clamp_min(1.0e-8)

    cfg = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    actor = load_actor(args.model, cfg, train=True)
    reference = load_actor(args.model, cfg, train=False)
    input_ids, attention, completion_mask = make_batch(
        rollout["prompt_token_ids"], completions, tokenizer.pad_token_id or tokenizer.eos_token_id
    )
    with torch.no_grad():
        old_logprobs = sequence_logprobs(actor, input_ids, attention, completion_mask)
        reference_logprobs = sequence_logprobs(reference, input_ids, attention, completion_mask)
    del reference
    gc.collect()
    torch.cuda.empty_cache()

    optimizer = torch.optim.AdamW(
        [parameter for parameter in actor.parameters() if parameter.requires_grad], lr=LR, weight_decay=0.01
    )
    optimizer.zero_grad(set_to_none=True)
    before = {name: parameter.detach().clone() for name, parameter in actor.named_parameters() if parameter.requires_grad}
    update_started = time.perf_counter()
    current_logprobs = sequence_logprobs(actor, input_ids, attention, completion_mask)
    ratio = torch.exp(current_logprobs - old_logprobs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - CLIP, 1.0 + CLIP) * advantages
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    kl = (current_logprobs - reference_logprobs).pow(2).mean()
    loss = policy_loss + KL_BETA * kl
    loss.backward()
    grad_norm = math.sqrt(
        sum(float(parameter.grad.float().norm()) ** 2 for parameter in actor.parameters() if parameter.grad is not None)
    )
    optimizer.step()
    torch.cuda.synchronize()
    update_seconds = time.perf_counter() - update_started
    changed = []
    max_update = 0.0
    for name, parameter in actor.named_parameters():
        if not parameter.requires_grad:
            continue
        delta = (parameter.detach().float() - before[name].float()).abs().max().item()
        if delta:
            changed.append(name)
            max_update = max(max_update, delta)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    saved = save_checkpoint(
        checkpoint_root,
        model=actor,
        optimizer=optimizer,
        scheduler=scheduler,
        trainer_state={"epoch": 0, "micro_batch_cursor": 1, "global_step": 1, "global_order_sha256": PROMPT_ID},
        manifest=prereg,
        keep_last=1,
    )
    resumed = load_actor(args.model, cfg, train=True)
    resumed_optimizer = torch.optim.AdamW(
        [parameter for parameter in resumed.parameters() if parameter.requires_grad], lr=LR, weight_decay=0.01
    )
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(resumed_optimizer, lambda _: 1.0)
    resume_state = load_latest_checkpoint(
        checkpoint_root,
        model=resumed,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        device=torch.device("cuda"),
    )
    resume_difference = parameter_difference(actor, resumed)
    del resumed, resumed_optimizer
    gc.collect()
    torch.cuda.empty_cache()

    updated_export = output_dir / "updated_deployment_export"
    export_started = time.perf_counter()
    export_updated(actor, tokenizer, updated_export)
    export_seconds = time.perf_counter() - export_started
    del actor, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    post_output = output_dir / "post_sync_rollout.json"
    post_receipt = output_dir / "post_sync_dispatch.jsonl"
    post_cmd = [
        sys.executable, str(Path(__file__).resolve()), "--worker-rollout",
        "--model", str(updated_export), "--output", str(post_output), "--receipt", str(post_receipt),
        "--prompt", PROMPT, "--n", "1", "--temperature", "0.0", "--max-tokens", "4",
    ]
    sync_started = time.perf_counter()
    subprocess.run(post_cmd, check=True, env={**os.environ, "PYTHONPATH": str(args.source_root / "src")})
    sync_engine_seconds = time.perf_counter() - sync_started
    post_dispatch = [json.loads(line) for line in post_receipt.read_text(encoding="utf-8").splitlines()]

    manifest = {
        **prereg,
        "status": "passed",
        "rollout_only": {
            "wall_seconds_including_engine": rollout_wall,
            "engine_generation_seconds": rollout["wall_seconds"],
            "outputs": rollout["outputs"],
            "rewards": rewards,
            "advantages": advantages.cpu().tolist(),
            "dispatch_receipts": dispatch,
        },
        "logprobs": {
            "old": old_logprobs.cpu().tolist(),
            "reference": reference_logprobs.cpu().tolist(),
            "old_reference_max_abs_difference": float((old_logprobs - reference_logprobs).abs().max()),
        },
        "update": {
            "policy_loss": float(policy_loss.detach()),
            "kl": float(kl.detach()),
            "total_loss": float(loss.detach()),
            "grad_norm": grad_norm,
            "changed_parameter_count": len(changed),
            "changed_parameter_names": changed,
            "max_abs_update": max_update,
            "seconds": update_seconds,
        },
        "checkpoint": {"path": str(saved), "resume_state": resume_state, "resume_max_abs_difference": resume_difference},
        "weight_sync": {
            "method": "full export and vLLM engine rebuild",
            "export_seconds": export_seconds,
            "engine_rebuild_and_generate_seconds": sync_engine_seconds,
            "updated_export": str(updated_export),
            "post_sync_dispatch_receipts": post_dispatch,
            "post_sync_output": json.loads(post_output.read_text(encoding="utf-8")),
        },
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    failures = []
    if len(dispatch) != 3 or len(post_dispatch) != 3:
        failures.append("triton_dispatch_incomplete")
    if float(advantages.std(unbiased=False)) < 0.99:
        failures.append("reward_advantage_variation_missing")
    if not changed or grad_norm == 0.0:
        failures.append("actor_update_missing")
    if resume_difference != 0.0:
        failures.append("checkpoint_resume_mismatch")
    manifest["failures"] = failures
    manifest["status"] = "passed" if not failures else "failed"
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0 if not failures else 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-rollout", action="store_true")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.worker_rollout:
        raise SystemExit(worker_rollout(parsed))
    if parsed.output_dir is None or parsed.config is None:
        raise SystemExit("--output-dir and --config are required for the main smoke")
    raise SystemExit(main(parsed))
