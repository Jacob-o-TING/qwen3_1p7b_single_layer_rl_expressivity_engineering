from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from qwen_single_layer_rl.data.prepare_validation import prepare_validation


def _write(path: Path, problems: list[str]) -> None:
    path.write_text(
        "".join(
            json.dumps({"problem": problem, "solution": "solution"}) + "\n"
            for problem in problems
        ),
        encoding="utf-8",
    )


class PrepareValidationTests(unittest.TestCase):
    def test_excludes_train_benchmark_and_candidate_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.jsonl"
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            benchmarks = root / "benchmarks.jsonl"
            train_problem = "A unique training question whose exact text must never enter validation."
            benchmark_problem = "A pinned benchmark question whose exact text must never enter validation."
            valid_one = "A clean validation question selected from the preferred candidate source."
            valid_two = "A second clean validation question selected from the fallback candidate source."
            _write(train, [train_problem])
            _write(first, [train_problem, benchmark_problem, valid_one, valid_one])
            _write(second, [valid_two])
            benchmarks.write_text(
                json.dumps({"problem": benchmark_problem}) + "\n",
                encoding="utf-8",
            )

            selected, manifest = prepare_validation(
                train,
                [first, second],
                benchmarks,
                target_size=2,
            )

        self.assertEqual([record["problem"] for record in selected], [valid_one, valid_two])
        self.assertEqual(manifest["source_reports"][0]["train_overlap_skipped"], 1)
        self.assertEqual(manifest["source_reports"][0]["benchmark_exact_skipped"], 1)
        self.assertEqual(manifest["source_reports"][0]["duplicate_candidate_skipped"], 1)


if __name__ == "__main__":
    unittest.main()
