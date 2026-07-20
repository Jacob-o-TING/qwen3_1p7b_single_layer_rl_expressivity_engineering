from __future__ import annotations

import unittest

from qwen_single_layer_rl.eval.response_length_diagnostics import (
    _score_fields,
    build_diagnostics,
    has_boxed_answer,
    summarize_rows,
)


class ResponseLengthDiagnosticsTests(unittest.TestCase):
    def test_evalscope_nested_score_fields_are_parsed(self) -> None:
        review = {
            "sample_score": {
                "score": {
                    "value": {"acc": 1.0},
                    "extracted_prediction": "42",
                }
            }
        }
        self.assertEqual(_score_fields(review), ("42", 1.0))

    def test_box_parser_handles_nested_and_rejects_empty_or_unclosed(self) -> None:
        self.assertTrue(has_boxed_answer(r"answer: \boxed{\frac{1}{2}}"))
        self.assertFalse(has_boxed_answer(r"answer: \boxed{}"))
        self.assertFalse(has_boxed_answer(r"answer: \boxed{42"))

    def test_summary_constructs_cap_extraction_table(self) -> None:
        rows = [
            self._row("A", 3072, True, True, 1.0),
            self._row("B", 3072, True, False, 0.0),
            self._row("C", 200, False, True, 1.0),
            self._row("D", 100, False, False, 0.0),
        ]
        summary = summarize_rows(rows)
        self.assertEqual(summary["cells"], {"A": 1, "B": 1, "C": 1, "D": 1})
        self.assertEqual(summary["cap_hit_rate"], 0.5)
        self.assertEqual(summary["p_missing_given_cap_hit"], 0.5)
        self.assertEqual(summary["accuracy_cap_hit"], 0.5)

    def test_manual_samples_are_deterministic_and_only_from_b_and_d(self) -> None:
        rows = [
            self._row("B", 3072, True, False, 0.0, index=index)
            for index in range(5)
        ] + [self._row("D", 100, False, False, 0.0, index=10)]
        first = build_diagnostics(
            rows,
            sources=[],
            token_cap=3072,
            sample_seed=7,
            samples_per_cell=2,
            tokenizer_identity="test",
        )
        second = build_diagnostics(
            rows,
            sources=[],
            token_cap=3072,
            sample_seed=7,
            samples_per_cell=2,
            tokenizer_identity="test",
        )
        self.assertEqual(first["manual_inspection_samples"], second["manual_inspection_samples"])
        self.assertTrue(
            all(
                row["cell"] in {"B", "D"}
                for samples in first["manual_inspection_samples"].values()
                for row in samples
            )
        )

    @staticmethod
    def _row(
        cell: str,
        tokens: int,
        cap_hit: bool,
        extracted: bool,
        accuracy: float,
        *,
        index: int = 0,
    ) -> dict[str, object]:
        return {
            "benchmark": "bench",
            "index": index,
            "row_id": f"bench:{index}",
            "generated_tokens": tokens,
            "cap_hit_proxy": cap_hit,
            "extracted_answer_present": extracted,
            "boxed_answer_present": extracted,
            "accuracy": accuracy,
            "cell": cell,
        }


if __name__ == "__main__":
    unittest.main()
