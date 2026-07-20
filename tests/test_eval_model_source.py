from __future__ import annotations

import unittest
from pathlib import Path

from qwen_single_layer_rl.eval.run_evalscope import _validate_model_source


class EvalModelSourceTests(unittest.TestCase):
    def test_checkpoint_mode_requires_checkpoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "checkpoint-dir is required"):
            _validate_model_source(checkpoint_dir=None, base_model_only=False)

    def test_base_mode_rejects_checkpoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            _validate_model_source(
                checkpoint_dir=Path("checkpoint"),
                base_model_only=True,
            )

    def test_valid_sources(self) -> None:
        _validate_model_source(
            checkpoint_dir=Path("checkpoint"),
            base_model_only=False,
        )
        _validate_model_source(checkpoint_dir=None, base_model_only=True)


if __name__ == "__main__":
    unittest.main()
