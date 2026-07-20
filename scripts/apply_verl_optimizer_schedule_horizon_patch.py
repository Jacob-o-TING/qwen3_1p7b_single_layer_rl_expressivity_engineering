from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "optimizer_schedule_total_training_steps"
OLD = """                if OmegaConf.select(self.config, \"actor_rollout_ref.actor.optim\"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, \"critic.optim\"):
                    self.config.critic.optim.total_training_steps = total_training_steps
"""
NEW = """                optimizer_schedule_total_training_steps = int(
                    self.config.trainer.get(\"optimizer_schedule_total_training_steps\", total_training_steps)
                )
                if OmegaConf.select(self.config, \"actor_rollout_ref.actor.optim\"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = (
                        optimizer_schedule_total_training_steps
                    )
                if OmegaConf.select(self.config, \"critic.optim\"):
                    self.config.critic.optim.total_training_steps = optimizer_schedule_total_training_steps
"""


def apply_patch(verl_root: Path) -> str:
    target = verl_root / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    text = target.read_text(encoding="utf-8")
    if MARKER in text:
        return "already_applied"
    if text.count(OLD) != 1:
        raise RuntimeError(f"Expected exactly one pinned veRL injection block in {target}")
    target.write_text(text.replace(OLD, NEW), encoding="utf-8")
    return "applied"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verl-root", type=Path, required=True)
    args = parser.parse_args()
    status = apply_patch(args.verl_root.resolve())
    print(f"VERL_OPTIMIZER_SCHEDULE_HORIZON_PATCH status={status} root={args.verl_root.resolve()}")


if __name__ == "__main__":
    main()
