from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def inspect_completed_training(checkpoint_root: Path, target_steps: int) -> dict[str, Any]:
    tracker = checkpoint_root / "latest_checkpointed_iteration.txt"
    payload: dict[str, Any] = {
        "checkpoint_root": str(checkpoint_root.resolve()),
        "target_steps": int(target_steps),
        "tracker_path": str(tracker.resolve()),
        "status": "incomplete",
        "completed_step": None,
        "new_optimizer_steps": None,
    }
    if not tracker.is_file():
        return payload
    try:
        completed_step = int(tracker.read_text(encoding="utf-8").strip())
    except ValueError as exc:
        raise RuntimeError(f"Invalid checkpoint tracker: {tracker}") from exc
    if completed_step < 0:
        raise RuntimeError(f"Negative checkpoint step in {tracker}: {completed_step}")
    payload["completed_step"] = completed_step
    if completed_step < target_steps:
        return payload

    actor_dir = checkpoint_root / f"global_step_{completed_step}" / "actor"
    required_patterns = (
        "model_world_size_*_rank_0.pt",
        "optim_world_size_*_rank_0.pt",
        "extra_state_world_size_*_rank_0.pt",
    )
    missing = [pattern for pattern in required_patterns if not any(actor_dir.glob(pattern))]
    if missing:
        raise RuntimeError(f"Completed checkpoint is missing actor files: {missing}")
    payload.update(
        {
            "status": "complete",
            "actor_checkpoint_dir": str(actor_dir.resolve()),
            "new_optimizer_steps": 0,
        }
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--target-steps", type=int, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    payload = inspect_completed_training(args.checkpoint_root, args.target_steps)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "complete":
        return 3
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
