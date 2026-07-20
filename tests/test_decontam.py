from __future__ import annotations

import unittest

from qwen_single_layer_rl.data.decontam import (
    BenchmarkProblemIndex,
    filter_decontaminated,
    hash_problem,
    normalize_problem,
)
from qwen_single_layer_rl.data.prep_numina import reservoir_sample_decontam


class DecontaminationTests(unittest.TestCase):
    def test_normalization_handles_unicode_punctuation_and_latex_spacing(self) -> None:
        left = "Ｆｉｎｄ  X, where $X = 1 \\, + 2$!"
        right = "find x where x 1 + 2"
        self.assertEqual(normalize_problem(left), normalize_problem(right))
        self.assertEqual(hash_problem(left), hash_problem(right))

    def test_exact_and_near_duplicates_are_reported_separately(self) -> None:
        benchmark = "Find all integer values of x such that x squared plus five x plus six equals zero."
        near = benchmark + " Show every algebraic step in your derivation."
        unrelated = "A farmer has twelve apples and gives three to a friend. How many remain?"
        index = BenchmarkProblemIndex([benchmark], ngram_size=4)

        kept, report = filter_decontaminated(
            [{"problem": benchmark}, {"problem": near}, {"problem": unrelated}],
            index,
        )

        self.assertEqual(kept, [{"problem": unrelated}])
        self.assertEqual(report.exact_removed_count, 1)
        self.assertEqual(report.near_removed_count, 1)
        self.assertEqual(report.removed_count, 2)

    def test_shared_short_phrase_does_not_trigger_near_match(self) -> None:
        index = BenchmarkProblemIndex(
            ["Let x be a real number and determine the unique value satisfying this cubic equation."],
            ngram_size=4,
        )
        self.assertIsNone(index.match_kind("Let x be a real number and compute its square."))

    def test_streaming_reservoir_applies_the_same_index(self) -> None:
        benchmark = "Determine every integer solution to this polynomial equation over the rational numbers."
        index = BenchmarkProblemIndex([benchmark], ngram_size=4)
        sampled, report = reservoir_sample_decontam(
            [
                {"problem": benchmark, "solution": "one"},
                {"problem": benchmark, "solution": "duplicate"},
                {"problem": benchmark + " Include all intermediate work.", "solution": "two"},
                {"problem": "How many edges does a cube have?", "solution": "twelve"},
            ],
            target_size=10,
            seed=7,
            benchmark_hashes=set(index.hashes),
            benchmark_index=index,
        )
        self.assertEqual([record["problem"] for record in sampled], ["How many edges does a cube have?"])
        self.assertEqual(report.exact_removed_count, 1)
        self.assertEqual(report.near_removed_count, 1)
        self.assertEqual(report.source_duplicate_removed_count, 1)


if __name__ == "__main__":
    unittest.main()
