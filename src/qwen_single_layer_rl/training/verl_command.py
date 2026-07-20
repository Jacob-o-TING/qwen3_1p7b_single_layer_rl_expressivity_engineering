from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

from qwen_single_layer_rl.config import load_config, resolve_run_id


VERL_FIXED_SCHEDULE_MARKER = "optimizer_schedule_total_training_steps"


def _latest_checkpoint_step(out_dir: Path) -> int:
    tracker = out_dir / "checkpoints" / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        return 0
    digits = "".join(character for character in tracker.read_text(encoding="utf-8") if character.isdigit())
    return int(digits or 0)


def _continuation_lr_decay(
    cfg: dict[str, Any],
    *,
    out_dir: Path,
    verl_root: Path,
) -> tuple[list[str], dict[str, Any]]:
    schedule = cfg.get("continuation", {}).get("lr_decay", {})
    enabled = bool(schedule.get("enabled", False))
    completed_step = _latest_checkpoint_step(out_dir)
    start_step = int(schedule.get("start_global_step", 0))
    end_step = int(schedule.get("end_global_step", 0))
    active = enabled and completed_step >= start_step
    receipt = {
        "enabled": enabled,
        "active": active,
        "completed_step_at_command_build": completed_step,
        "scheduler": str(schedule.get("scheduler", "constant")),
        "start_global_step": start_step,
        "end_global_step": end_step,
        "min_lr_ratio": float(schedule.get("min_lr_ratio", 1.0)),
    }
    if not active:
        return [], receipt
    if receipt["scheduler"] != "cosine":
        raise ValueError(f"Unsupported continuation LR scheduler: {receipt['scheduler']}")
    if not 0 <= receipt["min_lr_ratio"] <= 1:
        raise ValueError("continuation.lr_decay.min_lr_ratio must be in [0, 1]")
    if start_step <= 0 or end_step <= start_step:
        raise ValueError("continuation LR decay requires 0 < start_global_step < end_global_step")

    trainer_source = verl_root / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    if not trainer_source.is_file() or VERL_FIXED_SCHEDULE_MARKER not in trainer_source.read_text(
        encoding="utf-8"
    ):
        raise RuntimeError(
            "Pinned veRL lacks the fixed optimizer schedule-horizon patch; run "
            "scripts/apply_verl_optimizer_schedule_horizon_patch.py first"
        )
    return [
        f"++trainer.{VERL_FIXED_SCHEDULE_MARKER}={end_step}",
        "++actor_rollout_ref.actor.optim.lr_scheduler_type=cosine",
        f"++actor_rollout_ref.actor.optim.lr_warmup_steps={start_step}",
        f"++actor_rollout_ref.actor.optim.min_lr_ratio={receipt['min_lr_ratio']}",
        "++actor_rollout_ref.actor.optim.num_cycles=0.5",
    ], receipt


def build_verl_command(
    config_path: Path,
    *,
    project_root: Path,
    verl_root: Path,
    model_path: Path,
    data_dir: Path,
    run_root: Path,
    checkpoint_dir: Path | None = None,
    resume_from_path: Path | None = None,
    reference_model_path: Path | None = None,
) -> tuple[list[str], dict[str, Any]]:
    cfg = load_config(config_path)
    run_id = resolve_run_id(cfg)
    grpo = cfg.get("grpo", {})
    runtime = cfg.get("runtime", {})
    architecture_name = str(cfg.get("architecture_variant", {}).get("name", "identity"))
    rollout_runtime = runtime.get("rollout", {})
    inference_backend = str(runtime.get("inference_backend", "hf"))
    experiment = cfg.get("experiment", {})
    initialization_contract = str(experiment.get("initialization_contract", "checkpoint_overlay"))
    train_batch = int(grpo.get("train_batch_size", 512))
    mini_batch = int(grpo.get("ppo_mini_batch_size", 128))
    micro_batch = int(grpo.get("ppo_micro_batch_size", 8))
    group_size = int(grpo.get("group_size", 4))
    nproc = int(runtime.get("nproc_per_node", 4))
    dataloader_num_workers = int(runtime.get("dataloader_num_workers", 8))
    rollout_max_num_seqs = int(
        rollout_runtime.get("hf_micro_batch_size", rollout_runtime.get("max_num_seqs", 64))
    )
    rollout_tensor_parallel_size = int(rollout_runtime.get("tensor_model_parallel_size", 1))
    rollout_gpu_memory_utilization = float(rollout_runtime.get("gpu_memory_utilization", 0.5))
    rollout_max_num_batched_tokens = int(rollout_runtime.get("max_num_batched_tokens", 8192))
    if micro_batch % nproc != 0:
        raise ValueError(
            f"ppo_micro_batch_size={micro_batch} must be divisible by nproc_per_node={nproc} "
            "for veRL per-GPU micro-batch overrides."
        )
    micro_batch_per_gpu = micro_batch // nproc
    out_dir = run_root / run_id
    allows_exact_noop_init = architecture_name in {
        "qwen_swiglu_triglu_side",
        "qwen_swiglu_oft",
    } and initialization_contract == "untuned_base_exact_noop"
    if architecture_name != "identity" and checkpoint_dir is None and not allows_exact_noop_init:
        raise ValueError(f"checkpoint_dir is required for architecture variant {architecture_name}")
    python_bin = os.environ.get("PYTHON_BIN", sys.executable)
    reward_path = project_root / "src" / "qwen_single_layer_rl" / "rewards" / "verl_math_reward.py"

    args = [
        python_bin,
        "-m",
        "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        f"custom_reward_function.path={reward_path}",
        "custom_reward_function.name=compute_score",
        f"data.train_files={data_dir / 'train.parquet'}",
        f"data.val_files={data_dir / 'val.parquet'}",
        f"data.train_batch_size={train_batch}",
        "data.val_batch_size=128",
        f"data.max_prompt_length={int(grpo.get('max_prompt_length', 1024))}",
        f"data.max_response_length={int(grpo.get('max_response_length', 3072))}",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        "data.shuffle=True",
        f"data.dataloader_num_workers={dataloader_num_workers}",
        f"data.seed={int(cfg.get('dataset', {}).get('dataloader_seed', experiment.get('seed', 0)))}",
        f"actor_rollout_ref.model.path={model_path}",
        "actor_rollout_ref.model.external_lib=qwen_single_layer_rl.training.verl_model_hook",
        "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
        f"actor_rollout_ref.actor.optim.lr={grpo.get('learning_rate', 5.0e-6)}",
        "actor_rollout_ref.model.use_remove_padding=False",
        "actor_rollout_ref.model.use_fused_kernels=False",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={mini_batch}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={micro_batch_per_gpu}",
        "actor_rollout_ref.actor.use_kl_loss=True",
        f"actor_rollout_ref.actor.kl_loss_coef={grpo.get('kl_coefficient', 0.001)}",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        f"actor_rollout_ref.actor.clip_ratio={grpo.get('clip_range', 0.2)}",
        f"actor_rollout_ref.actor.clip_ratio_low={grpo.get('clip_range', 0.2)}",
        f"actor_rollout_ref.actor.clip_ratio_high={grpo.get('clip_range', 0.2)}",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.use_torch_compile=False",
        "actor_rollout_ref.ref.use_torch_compile=False",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        "actor_rollout_ref.actor.fsdp_config.use_orig_params=True",
        "actor_rollout_ref.actor.fsdp_config.use_torch_compile=False",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "actor_rollout_ref.ref.fsdp_config.use_orig_params=True",
        "actor_rollout_ref.ref.fsdp_config.use_torch_compile=False",
        f"actor_rollout_ref.rollout.name={inference_backend}",
        "actor_rollout_ref.rollout.mode=sync",
        f"actor_rollout_ref.rollout.top_k={0 if inference_backend == 'hf' else -1}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={rollout_tensor_parallel_size}",
        f"actor_rollout_ref.rollout.max_num_seqs={rollout_max_num_seqs}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={rollout_gpu_memory_utilization}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={rollout_max_num_batched_tokens}",
        f"actor_rollout_ref.rollout.n={group_size}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={micro_batch_per_gpu}",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={micro_batch_per_gpu}",
        "critic.enable=False",
        "trainer.critic_warmup=0",
        "trainer.logger=['console']",
        "trainer.project_name=qwen3_single_layer_rl",
        f"trainer.experiment_name={run_id}",
        f"trainer.n_gpus_per_node={nproc}",
        f"trainer.nnodes={int(runtime.get('nnodes', 1))}",
        f"trainer.default_local_dir={out_dir / 'checkpoints'}",
        f"trainer.validation_data_dir={out_dir / 'validation_data'}",
        f"trainer.val_before_train={str(bool(grpo.get('val_before_train', False))).lower()}",
        f"trainer.save_freq={int(grpo.get('save_freq', 999999))}",
        f"trainer.test_freq={int(grpo.get('test_freq', 999999))}",
        f"trainer.total_epochs={int(grpo.get('epochs', 4))}",
        f"trainer.resume_mode={'resume_path' if resume_from_path is not None else 'auto'}",
    ]
    if resume_from_path is not None:
        args.append(f"trainer.resume_from_path={resume_from_path.resolve()}")
    if reference_model_path is not None:
        args.append(f"+actor_rollout_ref.ref.model.path={reference_model_path.resolve()}")
    if "total_training_steps" in grpo:
        args.append(f"trainer.total_training_steps={int(grpo['total_training_steps'])}")
    if inference_backend == "vllm" and architecture_name == "qwen_swiglu_shs":
        args.extend(
            [
                "actor_rollout_ref.model.trust_remote_code=True",
                "actor_rollout_ref.rollout.enforce_eager=True",
                "+actor_rollout_ref.rollout.engine_kwargs.vllm.model_impl=transformers",
            ]
        )
    if inference_backend == "vllm" and architecture_name == "qwen_swiglu_triglu_side":
        args.extend(
            [
                "actor_rollout_ref.model.trust_remote_code=True",
                "actor_rollout_ref.rollout.enforce_eager=True",
                "+actor_rollout_ref.rollout.engine_kwargs.vllm.model_impl=auto",
            ]
        )
    if inference_backend == "vllm" and architecture_name == "qwen_swiglu_oft":
        args.extend(
            [
                "actor_rollout_ref.model.trust_remote_code=True",
                "actor_rollout_ref.rollout.enforce_eager=True",
                "+actor_rollout_ref.rollout.engine_kwargs.vllm.model_impl=auto",
            ]
        )

    lr_decay_args, lr_decay_receipt = _continuation_lr_decay(
        cfg,
        out_dir=out_dir,
        verl_root=verl_root,
    )
    args.extend(lr_decay_args)

    env = {
        "PYTHONPATH": f"{project_root / 'src'}:{verl_root}",
        "QWEN_SINGLE_LAYER_RL_CONFIG": str(config_path.resolve()),
        "QWEN_SINGLE_LAYER_RL_AUDIT_DIR": str(out_dir / "audits"),
        "PYTHONHASHSEED": str(experiment.get("seed", 0)),
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "NCCL_DEBUG": "WARN",
    }
    if checkpoint_dir is not None:
        env["QWEN_SINGLE_LAYER_RL_CHECKPOINT_DIR"] = str(checkpoint_dir.resolve())
    env["QWEN_SINGLE_LAYER_RL_INITIALIZATION"] = initialization_contract
    if architecture_name == "qwen_swiglu_shs":
        env["SHS_INFERENCE_MUL_BACKEND"] = "reference"
        env["SHS_DISPATCH_RECEIPT"] = str(out_dir / "rollout_dispatch.jsonl")
    if architecture_name == "qwen_swiglu_triglu_side":
        env["TRIGLU_DISPATCH_RECEIPT"] = str(out_dir / "rollout_dispatch.jsonl")
    if architecture_name == "qwen_swiglu_oft":
        env["OFT_DISPATCH_RECEIPT"] = str(out_dir / "rollout_dispatch.jsonl")
    if inference_backend == "hf":
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    manifest = {
        "run_id": run_id,
        "config_path": str(config_path),
        "command": args,
        "env": env,
        "out_dir": str(out_dir),
        "verl_root": str(verl_root),
        "model_path": str(model_path),
        "data_dir": str(data_dir),
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
        "reference_model_path": str(reference_model_path.resolve()) if reference_model_path else None,
        "architecture_variant": architecture_name,
        "paper_hyperparams": {
            "train_batch_size": train_batch,
            "ppo_mini_batch_size": mini_batch,
            "ppo_micro_batch_size": micro_batch,
            "ppo_micro_batch_size_per_gpu": micro_batch_per_gpu,
            "group_size": group_size,
            "total_training_steps": grpo.get("total_training_steps"),
            "save_freq": int(grpo.get("save_freq", 999999)),
            "hf_rollout_micro_batch_size": rollout_max_num_seqs,
            "rollout_tensor_model_parallel_size": rollout_tensor_parallel_size,
            "rollout_max_num_seqs": rollout_max_num_seqs,
            "rollout_gpu_memory_utilization": rollout_gpu_memory_utilization,
            "rollout_max_num_batched_tokens": rollout_max_num_batched_tokens,
            "dataloader_num_workers": dataloader_num_workers,
            "resume_mode": "resume_path" if resume_from_path is not None else "auto",
            "resume_from_path": str(resume_from_path.resolve()) if resume_from_path is not None else None,
            "reference_policy_contract": (
                "explicit_frozen_reference" if reference_model_path is not None else "shared_initial_model"
            ),
            "continuation_lr_decay": lr_decay_receipt,
        },
    }
    return args, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--verl-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--resume-from-path", type=Path)
    parser.add_argument("--reference-model-path", type=Path)
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--print-shell", action="store_true")
    args = parser.parse_args()

    command, manifest = build_verl_command(
        args.config,
        project_root=args.project_root,
        verl_root=args.verl_root,
        model_path=args.model_path,
        data_dir=args.data_dir,
        run_root=args.run_root,
        checkpoint_dir=args.checkpoint_dir,
        resume_from_path=args.resume_from_path,
        reference_model_path=args.reference_model_path,
    )
    if args.manifest_out:
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    if args.print_shell:
        env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in manifest["env"].items())
        print(env_prefix + " " + " ".join(shlex.quote(part) for part in command))
    else:
        print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
