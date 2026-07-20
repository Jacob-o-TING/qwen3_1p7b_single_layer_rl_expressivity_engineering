from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "apply_verl_optimizer_schedule_horizon_patch.py"
SPEC = importlib.util.spec_from_file_location("apply_verl_optimizer_schedule_horizon_patch", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ApplyVerlOptimizerScheduleHorizonPatchTests(unittest.TestCase):
    def test_exact_patch_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "verl" / "trainer" / "ppo" / "ray_trainer.py"
            target.parent.mkdir(parents=True)
            target.write_text(f"before\n{MODULE.OLD}after\n", encoding="utf-8")

            self.assertEqual(MODULE.apply_patch(root), "applied")
            first = target.read_text(encoding="utf-8")
            self.assertIn(MODULE.MARKER, first)
            self.assertEqual(MODULE.apply_patch(root), "already_applied")
            self.assertEqual(target.read_text(encoding="utf-8"), first)


if __name__ == "__main__":
    unittest.main()
