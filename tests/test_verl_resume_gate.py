from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qwen_single_layer_rl.training.resume_gate import inspect_completed_training


class VerlResumeGateTests(unittest.TestCase):
    def test_missing_tracker_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = inspect_completed_training(Path(tmp), target_steps=1)
        self.assertEqual(payload["status"], "incomplete")
        self.assertIsNone(payload["completed_step"])

    def test_completed_checkpoint_requires_all_actor_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "latest_checkpointed_iteration.txt").write_text("1\n", encoding="utf-8")
            actor = root / "global_step_1" / "actor"
            actor.mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, "missing actor files"):
                inspect_completed_training(root, target_steps=1)
            for name in (
                "model_world_size_1_rank_0.pt",
                "optim_world_size_1_rank_0.pt",
                "extra_state_world_size_1_rank_0.pt",
            ):
                (actor / name).touch()
            payload = inspect_completed_training(root, target_steps=1)

        self.assertEqual(payload["status"], "complete")
        self.assertEqual(payload["completed_step"], 1)
        self.assertEqual(payload["new_optimizer_steps"], 0)

    def test_checkpoint_before_target_remains_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "latest_checkpointed_iteration.txt").write_text("1\n", encoding="utf-8")
            payload = inspect_completed_training(root, target_steps=2)
        self.assertEqual(payload["status"], "incomplete")
        self.assertEqual(payload["completed_step"], 1)


if __name__ == "__main__":
    unittest.main()
