from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class GreedyAmcHookTests(unittest.TestCase):
    def test_triglu_entrypoint_has_scoped_diagnostic_hook(self) -> None:
        launcher = (ROOT / "scripts" / "launch_sft_single_node.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("layer10_whole_layer_triglu_side_ffn_sft.yaml", launcher)
        self.assertIn("launch_greedy_amc_controls_before_triglu.sh", launcher)
        self.assertIn("SFT_RUN_TRIGLU_GREEDY_BEFORE_OFT", launcher)
        self.assertIn("eval-status", launcher)

    def test_diagnostic_is_greedy_once_and_concurrent(self) -> None:
        diagnostic = (
            ROOT / "scripts" / "launch_greedy_amc_controls_before_triglu.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--datasets paper_amc23", diagnostic)
        self.assertIn("--amc-repeats 0", diagnostic)
        self.assertIn('BATCH_SIZE="${SFT_GREEDY_AMC_BATCH_SIZE:-16}"', diagnostic)
        self.assertIn('"do_sample": os.environ["DO_SAMPLE"] == "true"', diagnostic)
        self.assertIn("expected_rows=1280", diagnostic)
        self.assertIn("--amc-only --amc-repeats 32", diagnostic)
        self.assertIn("amc_greedy_modal_path_shs_sft50k_v1", diagnostic)
        self.assertIn("amc_greedy_modal_path_baseline_sft50k_v1", diagnostic)
        self.assertIn("amc_greedy_modal_path_untuned_qwen3_1p7b_base_v1", diagnostic)
        self.assertIn("--base-model-only", diagnostic)
        self.assertIn("amc_average_at_32_untuned_qwen3_1p7b_base_v1", diagnostic)
        self.assertLess(
            diagnostic.index("amc_greedy_modal_path_shs_sft50k_v1"),
            diagnostic.index("amc_greedy_modal_path_baseline_sft50k_v1"),
        )
        self.assertLess(
            diagnostic.index("amc_greedy_modal_path_baseline_sft50k_v1"),
            diagnostic.index("amc_greedy_modal_path_untuned_qwen3_1p7b_base_v1"),
        )
        self.assertLess(
            diagnostic.index("amc_greedy_modal_path_untuned_qwen3_1p7b_base_v1"),
            diagnostic.index("amc_average_at_32_untuned_qwen3_1p7b_base_v1"),
        )

    def test_post_eval_greedy_controls_follow_triglu_and_oft_eval(self) -> None:
        launcher = (ROOT / "scripts" / "launch_sft_ordered_variants.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("amc_greedy_modal_path_triglu_sft50k_v1", launcher)
        self.assertIn("amc_greedy_modal_path_oft_sft50k_v1", launcher)
        self.assertIn("post-variant", launcher)
        self.assertLess(
            launcher.index("SFT_ORDERED_EVAL_END"),
            launcher.index('post_eval_greedy_id=""'),
        )

        waiter = (ROOT / "scripts" / "wait_and_run_oft_greedy_amc.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("eval-status", waiter)
        self.assertIn("amc_greedy_modal_path_oft_sft50k_v1", waiter)
        self.assertLess(waiter.index("eval-status"), waiter.index("post-variant"))


if __name__ == "__main__":
    unittest.main()
