import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_parallel_eval.py"
SPEC = importlib.util.spec_from_file_location("summarize_parallel_eval", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
label = MODULE.label
display_label = MODULE.display_label
training_mix_composite = MODULE.training_mix_composite
math_avg = MODULE.math_avg
format_milestone_comparison = MODULE.format_milestone_comparison
collect_milestone_scores = MODULE.collect_milestone_scores


class SummarizeParallelEvalTests(unittest.TestCase):
    def test_amc_greedy_is_separate_from_sampled_average(self) -> None:
        sampled = Path(
            "amc_average_at_32/stamp/reviews/model/paper_amc23_main.jsonl"
        )
        greedy = Path("amc_greedy/stamp/reviews/model/paper_amc23_main.jsonl")

        self.assertEqual(label(sampled), "paper_amc23")
        self.assertEqual(label(greedy), "paper_amc23_greedy")
        self.assertEqual(display_label(label(sampled)), "paper_amc23_avg_at_32")
        self.assertEqual(
            display_label(label(greedy)), "paper_amc23_greedy_pass_at_1"
        )

    def test_training_mix_composite_uses_greedy_amc(self) -> None:
        benchmarks = {
            "paper_gsm8k": {"accuracy": 1102 / 1319},
            "paper_math500": {"accuracy": 320 / 500},
            "paper_olympiadbench": {"accuracy": 172 / 675},
            "paper_amc23": {"accuracy": 280 / 1280},
            "paper_amc23_greedy": {"accuracy": 15 / 40},
        }

        self.assertAlmostEqual(training_mix_composite(benchmarks), 64.2849251)

    def test_math_avg_uses_sampled_amc_average_at_32(self) -> None:
        benchmarks = {
            "paper_gsm8k": {"accuracy": 1102 / 1319},
            "paper_math500": {"accuracy": 320 / 500},
            "paper_olympiadbench": {"accuracy": 172 / 675},
            "paper_amc23": {"accuracy": 280 / 1280},
            "paper_amc23_greedy": {"accuracy": 15 / 40},
        }
        expected = 100.0 * (
            1102 / 1319 + 320 / 500 + 172 / 675 + 280 / 1280
        ) / 4

        self.assertAlmostEqual(math_avg(benchmarks), expected)

    def test_comparison_formats_matched_and_pending_milestones(self) -> None:
        matched = format_milestone_comparison(
            20, {"triglu": 64.2849251, "baseline": 63.7890202}
        )
        pending = format_milestone_comparison(98, {"triglu": 65.0})

        self.assertEqual(
            matched,
            "step 20: TriGLU=64.2849  baseline=63.7890  delta=+0.4959 pp",
        )
        self.assertEqual(
            pending,
            "step 98: TriGLU=65.0000  baseline=pending  delta=pending",
        )

    def test_milestone_filter_excludes_historical_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in (
                "triglu_step_30",
                "baseline_step_30",
                "triglu_step_128",
                "baseline_step_158",
            ):
                (root / name).mkdir()

            scores = collect_milestone_scores(root, steps={128, 158, 196})

        self.assertEqual(set(scores), {128, 158})


if __name__ == "__main__":
    unittest.main()
