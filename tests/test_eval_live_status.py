from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from qwen_single_layer_rl.eval.live_status import (
    build_model_summary,
    collect_live_status,
    format_live_status,
)


class EvalLiveStatusTests(unittest.TestCase):
    def test_collects_partial_accuracy_and_generated_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell = root / "vllm_main"
            cell.mkdir()
            receipts = [
                {"event": "engine_loaded", "actual_backend": "vllm", "engine_load_seconds": 2.5},
                {"event": "generation_completed", "generated_tokens": 11},
                {"event": "generation_completed", "generated_tokens": 13},
            ]
            (cell / "generation_receipts.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in receipts), encoding="utf-8"
            )
            review = cell / "main" / "stamp" / "reviews" / "model" / "paper_gsm8k_main.jsonl"
            review.parent.mkdir(parents=True)
            review.write_text(
                json.dumps({"sample_score": {"score": {"value": {"acc": 1.0}}}}) + "\n"
                + json.dumps({"sample_score": {"score": {"value": {"acc": 0.0}}}}) + "\n",
                encoding="utf-8",
            )
            status = collect_live_status(root)
        self.assertEqual(status["cells"]["vllm_main"]["generated_tokens"], 24)
        self.assertEqual(status["reviewed"], 2)
        self.assertEqual(status["partial_accuracy"], 0.5)
        self.assertIn("generated_tokens=24", format_live_status(status))

    def test_summary_keeps_amc_greedy_separate_from_primary_average(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = {
                "main/stamp/reviews/model/paper_math500_main.jsonl": 1.0,
                "main/stamp/reviews/model/paper_gsm8k_main.jsonl": 0.5,
                "main/stamp/reviews/model/paper_olympiadbench_main.jsonl": 0.25,
                "amc_average_at_32/stamp/reviews/model/paper_amc23_main.jsonl": 0.75,
                "amc_greedy/stamp/reviews/model/paper_amc23_main.jsonl": 0.0,
            }
            for relative, value in rows.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    json.dumps({"sample_score": {"score": {"value": {"acc": value}}}}) + "\n",
                    encoding="utf-8",
                )
            summary = build_model_summary(root)
        self.assertEqual(summary["four_benchmark_math_average"], 0.625)
        self.assertEqual(summary["benchmarks"]["amc_greedy_pass_at_1"]["accuracy"], 0.0)
        self.assertTrue(summary["amc_greedy_excluded_from_four_benchmark_average"])


if __name__ == "__main__":
    unittest.main()
