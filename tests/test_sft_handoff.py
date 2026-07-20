from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from qwen_single_layer_rl.sft.handoff import (
    EXPECTED_REPORTS,
    evaluation_is_complete,
    record_completed_evaluation,
    resolve_best_evaluation_cache,
    resolve_completed_checkpoint,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class SFTHandoffTests(unittest.TestCase):
    def test_resolve_requires_exact_final_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            checkpoint = run_dir / "checkpoints" / "step_00000012"
            checkpoint.mkdir(parents=True)
            for name in ("trainable_state.pt", "trainer_state.pt"):
                (checkpoint / name).write_bytes(b"checkpoint")
            _write_json(run_dir / "train_result.json", {"global_step": 12, "benchmark": False})
            _write_json(run_dir / "run_manifest.json", {"total_steps": 12})
            _write_json(run_dir / "checkpoints" / "latest.json", {"checkpoint": checkpoint.name, "global_step": 12})
            _write_json(checkpoint / "manifest.json", {"total_steps": 12})
            self.assertEqual(resolve_completed_checkpoint(run_dir), checkpoint.resolve())

            _write_json(run_dir / "train_result.json", {"global_step": 11, "benchmark": False})
            with self.assertRaises(ValueError):
                resolve_completed_checkpoint(run_dir)

    def test_eval_receipt_hashes_all_paper_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            config = root / "config.yaml"
            config.write_text("seed: 1\n", encoding="utf-8")
            eval_dir = root / "eval"
            _write_json(
                eval_dir / "evaluation_manifest.json",
                {"checkpoint_dir": str(checkpoint.resolve()), "limit": None},
            )
            for dataset in EXPECTED_REPORTS:
                _write_json(eval_dir / "phase" / "reports" / "model" / f"{dataset}.json", {"num": 1})
            record_completed_evaluation(eval_dir, checkpoint, config)
            self.assertTrue(evaluation_is_complete(eval_dir, checkpoint))

            report = eval_dir / "phase" / "reports" / "model" / "paper_math500.json"
            report.write_text("changed", encoding="utf-8")
            self.assertFalse(evaluation_is_complete(eval_dir, checkpoint))

    def test_limited_eval_cannot_be_marked_as_production_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            config = root / "config.yaml"
            config.write_text("seed: 1\n", encoding="utf-8")
            eval_dir = root / "eval"
            _write_json(
                eval_dir / "evaluation_manifest.json",
                {"checkpoint_dir": str(checkpoint.resolve()), "limit": 1},
            )
            for dataset in EXPECTED_REPORTS:
                _write_json(eval_dir / "phase" / "reports" / "model" / f"{dataset}.json", {"num": 1})
            with self.assertRaises(ValueError):
                record_completed_evaluation(eval_dir, checkpoint, config)

    def test_cache_resolution_prefers_coverage_over_newer_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            eval_dir = Path(temporary) / "eval"
            old_cache = eval_dir / "main" / "20260711_210000"
            new_cache = eval_dir / "main" / "20260711_220000"
            for cache, count in ((old_cache, 4), (new_cache, 2)):
                review = cache / "reviews" / "model" / "paper_math500_main.jsonl"
                review.parent.mkdir(parents=True)
                review.write_text(
                    "\n".join(json.dumps({"index": index}) for index in range(count)),
                    encoding="utf-8",
                )
            _write_json(
                eval_dir / "evaluation_manifest.json",
                {"main_use_cache": str(new_cache)},
            )

            self.assertEqual(resolve_best_evaluation_cache(eval_dir, "main"), old_cache.resolve())
            self.assertIsNone(resolve_best_evaluation_cache(eval_dir, "amc"))


if __name__ == "__main__":
    unittest.main()
