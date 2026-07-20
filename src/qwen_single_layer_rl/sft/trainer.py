from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

from qwen_single_layer_rl.config import load_config, resolve_run_id
from qwen_single_layer_rl.layers import apply_freeze_policy
from qwen_single_layer_rl.model_surgery import build_variant
from qwen_single_layer_rl.model_surgery.qwen_swiglu_variant_modules import ShuffledHyperGridDeltaLinear
from qwen_single_layer_rl.seeding import seed_everything
from qwen_single_layer_rl.sft.checkpoint import load_trainable_state_dict

from .checkpoint import load_latest_checkpoint, save_checkpoint, unwrap_model
from .data import (
    build_and_save_packed_cache,
    collate_packed_items,
    load_packed_cache,
    packed_cache_path,
)
from .distributed import build_rank_micro_batch_schedule


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _distributed_context() -> tuple[int, int, int, Any]:
    import torch
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("The SFT trainer requires a CUDA GPU")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank, device


def _barrier(world_size: int) -> None:
    if world_size > 1:
        import torch.distributed as dist

        dist.barrier()


def _all_reduce_sum(value: float, device: Any, world_size: int) -> float:
    if world_size == 1:
        return float(value)
    import torch
    import torch.distributed as dist

    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


def _scheduler(optimizer: Any, total_steps: int, warmup_ratio: float) -> Any:
    from torch.optim.lr_scheduler import LambdaLR

    warmup_steps = int(total_steps * warmup_ratio)

    def scale(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1.0e-8, step / warmup_steps)
        remaining = max(1, total_steps - warmup_steps)
        return max(0.0, (total_steps - step) / remaining)

    return LambdaLR(optimizer, scale)


def _checkpoint_steps(total_steps: int, fractions: list[float]) -> list[int]:
    steps = {
        min(total_steps, max(1, int(math.ceil(total_steps * float(fraction)))))
        for fraction in fractions
    }
    steps.add(total_steps)
    return sorted(steps)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _load_model_and_tokenizer(
    cfg: dict[str, Any],
    device: Any,
    rank: int,
    *,
    initial_checkpoint_dir: Path | None = None,
    shs_mul_backend: str | None = None,
) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_cfg = cfg.get("model", {})
    model_path = _resolve_path(model_cfg.get("local_path", model_cfg.get("name_or_path")))
    if not model_path.exists():
        model_path = Path(str(model_cfg.get("name_or_path")))
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )
    model.config.use_cache = False
    seed = int(cfg.get("experiment", {}).get("init_seed", cfg.get("experiment", {}).get("seed", 0)))
    seed_everything(seed)
    variant = build_variant(cfg)
    model = variant.apply(model, cfg)
    if initial_checkpoint_dir is not None:
        state = torch.load(initial_checkpoint_dir / "trainable_state.pt", map_location="cpu", weights_only=False)
        load_trainable_state_dict(model, state)
    if shs_mul_backend is not None:
        shs_modules = [module for module in model.modules() if isinstance(module, ShuffledHyperGridDeltaLinear)]
        if len(shs_modules) != 3:
            raise RuntimeError(f"expected three SHS projections, found {len(shs_modules)}")
        for module in shs_modules:
            module.set_inference_mul_backend(shs_mul_backend)
    freeze_report = apply_freeze_policy(model, cfg)
    if not freeze_report.trainable_parameter_names:
        raise RuntimeError("Freeze policy produced no trainable parameters")
    model.to(device)
    if rank == 0:
        trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        print(
            f"SFT_MODEL_AUDIT variant={variant.name} trainable={trainable_parameters} "
            f"total={total_parameters} trainable_names={len(freeze_report.trainable_parameter_names)}",
            flush=True,
        )
    return model, tokenizer, freeze_report


def _prepare_dataset(
    cfg: dict[str, Any],
    tokenizer: Any,
    rank: int,
    world_size: int,
    max_length: int,
    dataset_key: str = "sft_train",
) -> tuple[Any, dict[str, Any], Path]:
    dataset_cfg = cfg.get("dataset", {})
    sft_cfg = cfg.get("sft", {})
    source_path = _resolve_path(dataset_cfg[dataset_key])
    cache_dir = _resolve_path(sft_cfg.get("cache_dir", "data/sft_cache"))
    cache_path = packed_cache_path(cache_dir, tokenizer, source_path, max_length=max_length)
    if rank == 0 and not cache_path.exists():
        print(f"SFT_CACHE_BUILD_START source={source_path} cache={cache_path}", flush=True)
        manifest = build_and_save_packed_cache(tokenizer, source_path, cache_path, max_length=max_length)
        print(
            f"SFT_CACHE_BUILD_END packed={manifest['packed_sequence_count']} "
            f"truncated_tokens={manifest['truncated_tokens']}",
            flush=True,
        )
    _barrier(world_size)
    dataset, manifest = load_packed_cache(cache_path)
    return dataset, manifest, cache_path


def _validate(
    model: Any,
    dataset: Any,
    *,
    device: Any,
    rank: int,
    world_size: int,
    micro_batch_size: int,
    max_micro_batches: int,
) -> dict[str, float]:
    import torch

    schedule = build_rank_micro_batch_schedule(
        len(dataset),
        seed=0,
        epoch=0,
        shuffle=False,
        rank=rank,
        world_size=world_size,
        micro_batch_size=micro_batch_size,
    )
    micro_batches = schedule.micro_batches[:max_micro_batches]
    local_loss_numerator = 0.0
    local_assistant_tokens = 0
    local_non_padding_tokens = 0
    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        for indices in micro_batches:
            batch = _batch_to_device(dataset, indices, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                use_cache=False,
            )
            assistant_tokens = int(batch["assistant_tokens"])
            local_loss_numerator += float(outputs.loss.item()) * assistant_tokens
            local_assistant_tokens += assistant_tokens
            local_non_padding_tokens += int(batch["non_padding_tokens"])
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    loss_numerator = _all_reduce_sum(local_loss_numerator, device, world_size)
    assistant_tokens = _all_reduce_sum(local_assistant_tokens, device, world_size)
    non_padding_tokens = _all_reduce_sum(local_non_padding_tokens, device, world_size)
    model.train()
    return {
        "validation_loss": loss_numerator / max(1.0, assistant_tokens),
        "validation_assistant_tokens": assistant_tokens,
        "validation_non_padding_tokens": non_padding_tokens,
        "validation_seconds": elapsed,
    }


def _batch_to_device(dataset: Any, indices: tuple[int, ...], device: Any) -> dict[str, Any]:
    batch = collate_packed_items([dataset[index] for index in indices])
    for key in ("input_ids", "labels", "attention_mask"):
        batch[key] = batch[key].to(device, non_blocking=True)
    return batch


def _dynamo_counters() -> dict[str, Any]:
    try:
        from torch._dynamo.utils import counters

        return {
            category: {str(key): int(value) for key, value in values.items()}
            for category, values in counters.items()
            if values
        }
    except Exception:
        return {}


def train(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch.nn.parallel import DistributedDataParallel

    cfg = load_config(args.config)
    rank, world_size, local_rank, device = _distributed_context()
    seed = int(cfg.get("experiment", {}).get("seed", 0))
    seed_everything(seed)
    sft_cfg = cfg.get("sft", {})
    max_length = int(args.max_seq_length or sft_cfg.get("max_seq_length", 2048))
    micro_batch_size = int(args.micro_batch_size or sft_cfg.get("per_device_micro_batch_size", 1))
    accumulation = int(args.gradient_accumulation_steps or sft_cfg.get("gradient_accumulation_steps", 1))
    epochs = int(args.epochs or sft_cfg.get("epochs", 2))
    compile_mode = args.compile_mode or str(sft_cfg.get("compile_mode", "eager"))
    run_id = args.run_id or resolve_run_id(cfg)
    run_dir = _resolve_path(args.output_dir or Path("runs") / run_id)
    metrics_path = run_dir / "metrics.jsonl"
    checkpoint_root = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)

    initial_checkpoint_dir = _resolve_path(args.initial_checkpoint_dir) if args.initial_checkpoint_dir else None
    model, tokenizer, freeze_report = _load_model_and_tokenizer(
        cfg,
        device,
        rank,
        initial_checkpoint_dir=initial_checkpoint_dir,
        shs_mul_backend=args.shs_mul_backend,
    )
    dataset, data_manifest, cache_path = _prepare_dataset(cfg, tokenizer, rank, world_size, max_length)
    max_packed_sequences = sft_cfg.get("max_packed_sequences")
    if max_packed_sequences is not None:
        max_packed_sequences = int(max_packed_sequences)
        if max_packed_sequences <= 0:
            raise ValueError("sft.max_packed_sequences must be positive when set")
        dataset = dataset[:max_packed_sequences]
    validation_dataset = None
    validation_manifest = None
    validation_cache_path = None
    validation_source = cfg.get("dataset", {}).get("sft_val")
    if not args.benchmark and validation_source and _resolve_path(validation_source).exists():
        validation_dataset, validation_manifest, validation_cache_path = _prepare_dataset(
            cfg,
            tokenizer,
            rank,
            world_size,
            max_length,
            dataset_key="sft_val",
        )
    schedule0 = build_rank_micro_batch_schedule(
        len(dataset),
        seed=int(cfg.get("dataset", {}).get("dataloader_seed", seed)),
        epoch=0,
        shuffle=bool(cfg.get("dataset", {}).get("shuffle", True)),
        rank=rank,
        world_size=world_size,
        micro_batch_size=micro_batch_size,
    )
    steps_per_epoch = len(schedule0.micro_batches) // accumulation
    total_steps = steps_per_epoch * epochs
    if total_steps <= 0:
        raise RuntimeError("SFT schedule has no complete optimizer step")
    checkpoint_fractions = [float(value) for value in sft_cfg.get("checkpoint_fractions", [0.10, 0.25, 0.50, 0.75, 1.0])]
    checkpoint_steps = _checkpoint_steps(total_steps, checkpoint_fractions)

    if bool(sft_cfg.get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()
    if compile_mode != "eager":
        try:
            from torch._dynamo.utils import counters

            counters.clear()
        except Exception:
            pass
        model = torch.compile(model, mode=compile_mode, fullgraph=False, dynamic=False)
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=True,
            find_unused_parameters=False,
        )

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(sft_cfg.get("learning_rate", 5.0e-6)),
        weight_decay=float(sft_cfg.get("weight_decay", 0.01)),
    )
    scheduler = _scheduler(optimizer, total_steps, float(sft_cfg.get("warmup_ratio", 0.03)))
    trainer_state = None if args.no_resume or args.benchmark else load_latest_checkpoint(
        checkpoint_root,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )
    start_epoch = int(trainer_state.get("epoch", 0)) if trainer_state else 0
    start_cursor = int(trainer_state.get("micro_batch_cursor", 0)) if trainer_state else 0
    global_step = int(trainer_state.get("global_step", 0)) if trainer_state else 0

    manifest = {
        "run_id": run_id,
        "config_path": str(Path(args.config).resolve()),
        "variant": cfg.get("architecture_variant", {}).get("name", "identity"),
        "compile_mode": compile_mode,
        "initial_checkpoint_dir": str(initial_checkpoint_dir) if initial_checkpoint_dir else None,
        "shs_mul_backend_requested": args.shs_mul_backend,
        "seed": seed,
        "world_size": world_size,
        "resolved_topology": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "visible_gpu_count": torch.cuda.device_count(),
            "launcher_world_size": world_size,
            "rank": rank,
            "local_rank": local_rank,
        },
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": accumulation,
        "effective_packed_batch_size": micro_batch_size * accumulation * world_size,
        "max_seq_length": max_length,
        "max_packed_sequences": max_packed_sequences,
        "packed_sequences_used": len(dataset),
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "data_cache": str(cache_path),
        "data_manifest": data_manifest,
        "validation_data_cache": str(validation_cache_path) if validation_cache_path else None,
        "validation_data_manifest": validation_manifest,
        "global_order_sha256_epoch0": schedule0.global_order_sha256,
        "trainable_parameter_names": list(freeze_report.trainable_parameter_names),
        "checkpoint_policy": {
            "fractions": checkpoint_fractions,
            "steps": checkpoint_steps,
            "keep_last": int(sft_cfg.get("keep_last_checkpoints", 5)),
            "format": "trainable_weights_plus_optimizer_rng_sampler_cursor",
        },
    }
    if rank == 0:
        _write_json(run_dir / "run_manifest.json", manifest)
        print(
            f"SFT_RUN_START run_id={run_id} compile={compile_mode} world_size={world_size} "
            f"packed_sequences={len(dataset)} steps_per_epoch={steps_per_epoch}",
            flush=True,
        )

    correctness_steps = 1 if args.benchmark else 0
    benchmark_limit = correctness_steps + int(args.warmup_steps) + int(args.timed_steps)
    benchmark_step_times: list[float] = []
    benchmark_timed_assistant_tokens = 0
    benchmark_timed_non_padding_tokens = 0
    benchmark_cold_seconds = 0.0
    benchmark_correctness_loss = 0.0
    benchmark_start_step = correctness_steps + int(args.warmup_steps)
    run_start = time.perf_counter()

    model.train()
    for epoch in range(start_epoch, epochs):
        schedule = build_rank_micro_batch_schedule(
            len(dataset),
            seed=int(cfg.get("dataset", {}).get("dataloader_seed", seed)),
            epoch=epoch,
            shuffle=bool(cfg.get("dataset", {}).get("shuffle", True)),
            rank=rank,
            world_size=world_size,
            micro_batch_size=micro_batch_size,
        )
        cursor = start_cursor if epoch == start_epoch else 0
        while cursor + accumulation <= len(schedule.micro_batches):
            if args.benchmark and global_step >= benchmark_limit:
                break
            micro_indices = schedule.micro_batches[cursor : cursor + accumulation]
            local_assistant_tokens = sum(
                int(dataset[index]["assistant_tokens"])
                for indices in micro_indices
                for index in indices
            )
            global_assistant_tokens = _all_reduce_sum(local_assistant_tokens, device, world_size)
            if global_assistant_tokens <= 0:
                raise RuntimeError("Distributed optimizer window has no assistant training tokens")

            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            step_start = time.perf_counter()
            local_loss_numerator = 0.0
            local_non_padding_tokens = 0
            for micro_index, indices in enumerate(micro_indices):
                batch = _batch_to_device(dataset, indices, device)
                is_final_micro = micro_index == len(micro_indices) - 1
                sync_context = contextlib.nullcontext()
                if world_size > 1 and not is_final_micro:
                    sync_context = model.no_sync()
                with sync_context:
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        use_cache=False,
                    )
                    assistant_tokens = int(batch["assistant_tokens"])
                    scaled_loss = outputs.loss * (assistant_tokens * world_size / global_assistant_tokens)
                    scaled_loss.backward()
                local_loss_numerator += float(outputs.loss.detach().item()) * assistant_tokens
                local_non_padding_tokens += int(batch["non_padding_tokens"])

            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                float(sft_cfg.get("max_grad_norm", 1.0)),
            )
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize(device)
            step_seconds = time.perf_counter() - step_start
            cursor += accumulation
            global_step += 1

            global_loss_numerator = _all_reduce_sum(local_loss_numerator, device, world_size)
            global_non_padding_tokens = _all_reduce_sum(local_non_padding_tokens, device, world_size)
            loss = global_loss_numerator / global_assistant_tokens
            metric = {
                "event": "step",
                "run_id": run_id,
                "epoch": epoch,
                "global_step": global_step,
                "micro_batch_cursor": cursor,
                "loss": loss,
                "grad_norm": float(grad_norm),
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "step_seconds": step_seconds,
                "assistant_tokens": int(global_assistant_tokens),
                "non_padding_tokens": int(global_non_padding_tokens),
                "assistant_tokens_per_second": global_assistant_tokens / step_seconds,
                "non_padding_tokens_per_second": global_non_padding_tokens / step_seconds,
                "max_memory_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
                "max_memory_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024**3),
            }
            if rank == 0:
                _append_jsonl(metrics_path, metric)
                print(
                    "SFT_STEP "
                    f"step={global_step}/{total_steps} epoch={epoch} loss={loss:.6f} "
                    f"seconds={step_seconds:.3f} assistant_tok_s={metric['assistant_tokens_per_second']:.1f} "
                    f"memory_gb={metric['max_memory_allocated_gb']:.2f}",
                    flush=True,
                )

            if args.benchmark:
                if global_step == 1:
                    benchmark_cold_seconds = step_seconds
                    benchmark_correctness_loss = loss
                if global_step == benchmark_start_step:
                    torch.cuda.reset_peak_memory_stats(device)
                if global_step > benchmark_start_step:
                    benchmark_step_times.append(step_seconds)
                    benchmark_timed_assistant_tokens += int(global_assistant_tokens)
                    benchmark_timed_non_padding_tokens += int(global_non_padding_tokens)
            else:
                if (
                    validation_dataset is not None
                    and bool(sft_cfg.get("validation_at_checkpoint_fractions", True))
                    and global_step in checkpoint_steps
                ):
                    validation_metric = _validate(
                        model,
                        validation_dataset,
                        device=device,
                        rank=rank,
                        world_size=world_size,
                        micro_batch_size=micro_batch_size,
                        max_micro_batches=int(sft_cfg.get("validation_max_micro_batches", 16)),
                    )
                    validation_metric.update(
                        {
                            "event": "validation",
                            "run_id": run_id,
                            "epoch": epoch,
                            "global_step": global_step,
                        }
                    )
                    if rank == 0:
                        _append_jsonl(metrics_path, validation_metric)
                        print(
                            f"SFT_VALIDATION step={global_step} "
                            f"loss={validation_metric['validation_loss']:.6f} "
                            f"seconds={validation_metric['validation_seconds']:.3f}",
                            flush=True,
                        )
                if rank == 0 and global_step in checkpoint_steps:
                    saved = save_checkpoint(
                        checkpoint_root,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        trainer_state={
                            "epoch": epoch,
                            "micro_batch_cursor": cursor,
                            "global_step": global_step,
                            "global_order_sha256": schedule.global_order_sha256,
                        },
                        manifest=manifest,
                        keep_last=int(sft_cfg.get("keep_last_checkpoints", 5)),
                    )
                    progress = global_step / total_steps
                    print(
                        f"SFT_CHECKPOINT_SAVED path={saved} progress={progress:.4f} "
                        f"milestone_step={global_step}",
                        flush=True,
                    )
                _barrier(world_size)

        start_cursor = 0
        if args.benchmark and global_step >= benchmark_limit:
            break

    wall_seconds = time.perf_counter() - run_start
    result = {
        "run_id": run_id,
        "compile_mode": compile_mode,
        "global_step": global_step,
        "wall_seconds": wall_seconds,
        "benchmark": bool(args.benchmark),
    }
    if args.benchmark:
        receipt_modules = [module for module in unwrap_model(model).modules() if isinstance(module, ShuffledHyperGridDeltaLinear)]
        actual_backends = [module.last_inference_mul_backend for module in receipt_modules]
        expected_backend = (
            "triton_forward_reference_recompute_backward"
            if args.shs_mul_backend == "triton_reference_recompute"
            else args.shs_mul_backend
        )
        result.update(
            {
                "cold_step_seconds": benchmark_cold_seconds,
                "correctness_loss": benchmark_correctness_loss,
                "warmup_steps": int(args.warmup_steps),
                "timed_steps": len(benchmark_step_times),
                "step_seconds_mean": statistics.mean(benchmark_step_times),
                "step_seconds_median": statistics.median(benchmark_step_times),
                "step_seconds_p10": _percentile(benchmark_step_times, 0.10),
                "step_seconds_p90": _percentile(benchmark_step_times, 0.90),
                "step_seconds_stdev": statistics.pstdev(benchmark_step_times),
                "timed_assistant_tokens": benchmark_timed_assistant_tokens,
                "timed_non_padding_tokens": benchmark_timed_non_padding_tokens,
                "timed_assistant_tokens_per_second": benchmark_timed_assistant_tokens /
                max(1.0e-12, sum(benchmark_step_times)),
                "timed_non_padding_tokens_per_second": benchmark_timed_non_padding_tokens /
                max(1.0e-12, sum(benchmark_step_times)),
                "max_memory_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
                "max_memory_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024**3),
                "dynamo_counters": _dynamo_counters(),
                "dispatch_receipt": {
                    "requested": args.shs_mul_backend,
                    "actual": actual_backends,
                    "fallback": bool(expected_backend and actual_backends != [expected_backend] * 3),
                    "custom_backward": False if args.shs_mul_backend == "triton_reference_recompute" else None,
                },
            }
        )
    if rank == 0:
        _write_json(run_dir / ("benchmark_result.json" if args.benchmark else "train_result.json"), result)
        print(f"SFT_RUN_END run_id={run_id} steps={global_step} wall_seconds={wall_seconds:.3f}", flush=True)
    _barrier(world_size)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--compile-mode", choices=["eager", "default", "reduce-overhead", "max-autotune"])
    parser.add_argument("--max-seq-length", type=int)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--timed-steps", type=int, default=20)
    parser.add_argument("--initial-checkpoint-dir", type=Path)
    parser.add_argument(
        "--shs-mul-backend",
        choices=["reference", "triton_reference_recompute"],
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
