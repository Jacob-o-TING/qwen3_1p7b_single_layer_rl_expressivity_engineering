from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_with_copy(path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".stage-copy")
    shutil.copy2(path, temporary)
    os.replace(temporary, path)


def rebase_scheduler(state: dict, *, base_lr: float, global_step: int) -> dict:
    scheduler = state.get("lr_scheduler")
    if not isinstance(scheduler, dict):
        raise RuntimeError("extra-state has no lr_scheduler mapping")
    scheduler["base_lrs"] = [base_lr for _ in scheduler.get("base_lrs", [base_lr])]
    scheduler["_last_lr"] = [base_lr for _ in scheduler.get("_last_lr", [base_lr])]
    scheduler["last_epoch"] = global_step
    scheduler["_step_count"] = global_step + 1
    return state


def write_latest_checkpoint_tracker(target_root: Path, global_step: int) -> Path:
    tracker = target_root / "latest_checkpointed_iteration.txt"
    tracker.write_text(f"{global_step}\n", encoding="utf-8")
    return tracker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--target-checkpoints-root", type=Path, required=True)
    parser.add_argument("--global-step", type=int, required=True)
    parser.add_argument("--base-lr", type=float, required=True)
    parser.add_argument("--expected-source-lr", type=float, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_checkpoint.resolve()
    target_root = args.target_checkpoints_root.resolve()
    target = target_root / f"global_step_{args.global_step}"
    if source.name != target.name or not source.is_dir():
        raise RuntimeError(f"invalid source checkpoint: {source}")
    if target.exists():
        shutil.rmtree(target)
    target_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, copy_function=os.link, symlinks=True)

    extra_states = sorted((target / "actor").glob("extra_state_world_size_6_rank_*.pt"))
    optimizers = sorted((target / "actor").glob("optim_world_size_6_rank_*.pt"))
    if len(extra_states) != 6 or len(optimizers) != 6:
        raise RuntimeError("expected six optimizer and six extra-state shards")

    rank_receipts = []
    for extra_path, optimizer_path in zip(extra_states, optimizers, strict=True):
        replace_with_copy(extra_path)
        extra = torch.load(extra_path, map_location="cpu", weights_only=False)
        before = dict(extra["lr_scheduler"])
        rebase_scheduler(extra, base_lr=args.base_lr, global_step=args.global_step)
        torch.save(extra, extra_path)

        optimizer = torch.load(optimizer_path, map_location="cpu", weights_only=False)
        observed_lrs = [float(group["lr"]) for group in optimizer["param_groups"]]
        if any(abs(lr - args.expected_source_lr) > 1e-12 for lr in observed_lrs):
            raise RuntimeError(f"unexpected optimizer LR in {optimizer_path}: {observed_lrs}")
        rank_receipts.append(
            {
                "rank_file": extra_path.name,
                "scheduler_before": before,
                "scheduler_after": extra["lr_scheduler"],
                "optimizer_lrs": observed_lrs,
                "sha256": sha256(extra_path),
            }
        )

    write_latest_checkpoint_tracker(target_root, args.global_step)
    payload = {
        "status": "PASS",
        "source_checkpoint": str(source),
        "target_checkpoint": str(target),
        "copy_policy": "hardlink_tree_with_private_extra_state_copies",
        "model_optimizer_data_content_shared": True,
        "source_checkpoint_mutated": False,
        "global_step": args.global_step,
        "new_scheduler_base_lr": args.base_lr,
        "expected_optimizer_lr": args.expected_source_lr,
        "ranks": rank_receipts,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
