from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from qwen_single_layer_rl.sft.status_dashboard import format_dashboard


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class SftStatusDashboardTests(unittest.TestCase):
    def test_dashboard_prioritizes_scores_and_current_partial_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            shs = root / "layer10_whole_layer_shs"
            _write_json(shs / "run_manifest.json", {"variant": "shs", "total_steps": 2})
            (shs / "metrics.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"event": "step", "global_step": 1, "loss": 0.7, "step_seconds": 1.0}),
                        json.dumps({"event": "step", "global_step": 2, "loss": 0.6, "step_seconds": 0.9}),
                        json.dumps({"event": "validation", "global_step": 2, "validation_loss": 0.65}),
                    ]
                ),
                encoding="utf-8",
            )
            _write_json(shs / "train_result.json", {"wall_seconds": 120})
            eval_dir = root / "evaluations" / "layer10_whole_layer_shs"
            _write_json(
                eval_dir / "main" / "x" / "reports" / "model" / "paper_math500.json",
                {"score": 0.59, "num": 500},
            )
            diagnostic = root / "diagnostics" / "amc_greedy_modal_path_shs_sft50k_v1"
            _write_json(
                diagnostic / "diagnostic_complete.json",
                {"variant": "shs", "decode": {"do_sample": False}},
            )
            _write_json(
                diagnostic / "reports" / "model" / "paper_amc23.json",
                {"score": 0.275, "num": 40},
            )
            review = eval_dir / "main" / "x" / "reviews" / "model" / "paper_gsm8k_main.jsonl"
            review.parent.mkdir(parents=True, exist_ok=True)
            review.write_text(
                json.dumps({"sample_score": {"score": {"value": {"acc": 1.0}}}}) + "\n",
                encoding="utf-8",
            )

            dashboard = format_dashboard(root)

            self.assertIn("[SHS]", dashboard)
            self.assertIn("Training: DONE (2/2)", dashboard)
            self.assertIn("MATH-500 59.00% [n=500]", dashboard)
            self.assertIn("AMC greedy 27.50% [n=40]", dashboard)
            self.assertIn("GSM8K 1/1 correct (100.00%)", dashboard)
            self.assertIn("generated 1/1319 | PARTIAL", dashboard)
            self.assertIn("CURRENT PHASE: SHS evaluating GSM8K", dashboard)
            self.assertNotIn("checkpoints/", dashboard)

    def test_dashboard_shows_incomplete_diagnostic_as_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            diagnostic = root / "diagnostics" / "amc_greedy_modal_path_oft_sft50k_v1"
            review = diagnostic / "main" / "x" / "reviews" / "model" / "paper_amc23_main.jsonl"
            review.parent.mkdir(parents=True, exist_ok=True)
            review.write_text(
                "\n".join(
                    [
                        json.dumps({"sample_score": {"score": {"value": {"acc": 1.0}}}}),
                        json.dumps({"sample_score": {"score": {"value": {"acc": 0.0}}}}),
                    ]
                ),
                encoding="utf-8",
            )

            dashboard = format_dashboard(root)

            self.assertIn("OFT: 1/2 = 50.00%", dashboard)
            self.assertIn("2/40 generated | PARTIAL", dashboard)

    def test_dashboard_prefers_explicit_main_cache_over_newer_aborted_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            triglu = root / "layer10_whole_layer_triglu_side_ffn"
            _write_json(triglu / "run_manifest.json", {"variant": "triglu", "total_steps": 1})
            (triglu / "metrics.jsonl").write_text(
                json.dumps(
                    {"event": "step", "global_step": 1, "loss": 0.6, "step_seconds": 1.0}
                ),
                encoding="utf-8",
            )
            eval_dir = root / "evaluations" / "layer10_whole_layer_triglu_side_ffn"
            cache_dir = eval_dir / "main" / "old_cache"
            _write_json(
                eval_dir / "evaluation_manifest.json",
                {"main_use_cache": str(cache_dir)},
            )
            cached_review = (
                cache_dir
                / "reviews"
                / "model"
                / "paper_math500_main.jsonl"
            )
            cached_review.parent.mkdir(parents=True, exist_ok=True)
            cached_review.write_text(
                "\n".join(
                    json.dumps({"sample_score": {"score": {"value": {"acc": 1.0}}}})
                    for _ in range(3)
                ),
                encoding="utf-8",
            )
            aborted_review = (
                eval_dir
                / "main"
                / "newer_aborted"
                / "reviews"
                / "model"
                / "paper_math500_main.jsonl"
            )
            aborted_review.parent.mkdir(parents=True, exist_ok=True)
            aborted_review.write_text(
                json.dumps({"sample_score": {"score": {"value": {"acc": 0.0}}}}),
                encoding="utf-8",
            )

            dashboard = format_dashboard(root)

            self.assertIn("MATH-500 3/3 correct (100.00%)", dashboard)
            self.assertIn("generated 3/500 | PARTIAL", dashboard)


if __name__ == "__main__":
    unittest.main()
