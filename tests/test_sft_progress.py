from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from qwen_single_layer_rl.sft.progress import format_summary, summarize_run


class SFTProgressTests(unittest.TestCase):
    def test_summary_reports_rolling_trend_eta_and_validations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "run_manifest.json").write_text(
                json.dumps({"variant": "shs", "total_steps": 10}),
                encoding="utf-8",
            )
            records = [
                {
                    "event": "step",
                    "global_step": step,
                    "loss": 1.0 - step * 0.05,
                    "step_seconds": 2.0,
                }
                for step in range(1, 6)
            ]
            records.append({"event": "validation", "global_step": 5, "validation_loss": 0.7})
            (run_dir / "metrics.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            summary = summarize_run(run_dir, window=2)
            assert summary is not None
            self.assertEqual(summary["latest_step"], 5)
            self.assertEqual(summary["eta_train_seconds"], 10.0)
            self.assertLess(summary["loss_mean_delta"], 0.0)
            rendered = format_summary(summary)
            self.assertIn("step=5/10", rendered)
            self.assertIn("eta_train=00:00:10", rendered)
            self.assertIn("validations=5:0.700000", rendered)


if __name__ == "__main__":
    unittest.main()
