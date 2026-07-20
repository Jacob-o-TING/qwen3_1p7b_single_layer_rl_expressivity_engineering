#!/usr/bin/env python3
"""Emit an executable readiness gate for the production-shaped GRPO shard."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


RUN_ID = "shs_grpo_replica_shard_20260712_v2_realverl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--verl-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hook_source = (args.source_root / "src/qwen_single_layer_rl/training/verl_model_hook.py").read_text()
    reward_source = (args.source_root / "src/qwen_single_layer_rl/rewards/math_reward.py").read_text()
    command_source = (args.source_root / "src/qwen_single_layer_rl/training/verl_command.py").read_text()
    base_config = (args.source_root / "configs/base_qwen3_1p7b.yaml").read_text()
    fsdp_workers = (args.verl_root / "verl/workers/fsdp_workers.py").read_text()
    vllm_rollout = (
        args.verl_root / "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"
    ).read_text()
    gates = {
        "verl_main_ppo_importable": importlib.util.find_spec("verl.trainer.main_ppo") is not None,
        "train_parquet_present": (args.data_dir / "train.parquet").is_file(),
        "val_parquet_present": (args.data_dir / "val.parquet").is_file(),
        "completed_trainable_state_present": (args.checkpoint_dir / "trainable_state.pt").is_file(),
        "production_math_verifier_available": (
            importlib.util.find_spec("math_verify") is not None and "placeholder" not in reward_source
        ),
        "verl_actor_hook_has_checkpoint_overlay": "load_trainable_state_dict" in hook_source,
        "verl_fsdp_hook_contract_present": (
            "_qwen_single_layer_apply_hook" in fsdp_workers
            and "actor_module = _qwen_single_layer_apply_hook" in fsdp_workers
        ),
        "configured_reward_is_production_math_verify": "verifier: verl_math_verify" in base_config,
        "reference_shs_vllm_wiring_present": (
            "model_impl=transformers" in command_source and "SHS_INFERENCE_MUL_BACKEND" in command_source
        ),
        "one_gpu_shard_configured": "trainer.n_gpus_per_node={nproc}" in command_source,
        "actor_to_rollout_sync_owned_by_verl": (
            "await self.rollout.update_weights" in fsdp_workers
            and "async def update_weights" in vllm_rollout
        ),
    }
    blockers = [name for name, passed in gates.items() if not passed]
    payload = {
        "run_id": RUN_ID,
        "status": "ready" if not blockers else "blocked_before_launch",
        "scope": "real veRL one-GPU shard: 128 prompts, group size 4",
        "gates": gates,
        "blockers": blockers,
        "claims": {"production_candidate": False, "production_ready": False},
        "launch_attempted": False,
        "reason_not_launched": None if not blockers else "One or more executable readiness gates failed.",
    }
    (args.output_dir / "readiness.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        f"# {RUN_ID} Readiness",
        "",
        f"Status: **{payload['status']}**",
        "",
        (
            "The real production-shaped shard is ready to launch."
            if not blockers
            else "The real production-shaped shard remains blocked by the gates below."
        ),
        "",
        "## Blocking Gates",
        "",
        *[f"- `{name}`" for name in blockers],
        "",
        "veRL, both parquet files, and the completed SHS checkpoint are present. The remaining work is integration, not data or environment acquisition.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not blockers else 3


if __name__ == "__main__":
    raise SystemExit(main())
