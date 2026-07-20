from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from qwen_single_layer_rl.data.build_benchmark_contract import BENCHMARKS, build_contract


class BenchmarkContractTests(unittest.TestCase):
    def test_build_contract_checks_counts_and_extracts_questions(self) -> None:
        original = dict(BENCHMARKS)
        try:
            BENCHMARKS.clear()
            BENCHMARKS.update({"questions": ("questions/test.jsonl", 2)})
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                path = root / "questions" / "test.jsonl"
                path.parent.mkdir()
                path.write_text(
                    json.dumps({"question": "First question?"})
                    + "\n"
                    + json.dumps({"problem": "Second problem."})
                    + "\n",
                    encoding="utf-8",
                )
                records, manifest = build_contract(root, "deadbeef")
            self.assertEqual([record["problem"] for record in records], ["First question?", "Second problem."])
            self.assertEqual(manifest["problem_count"], 2)
            self.assertEqual(manifest["qwen_eval_source_revision"], "deadbeef")
        finally:
            BENCHMARKS.clear()
            BENCHMARKS.update(original)


if __name__ == "__main__":
    unittest.main()
