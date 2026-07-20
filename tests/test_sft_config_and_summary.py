from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from qwen_single_layer_rl.config import load_config
from qwen_single_layer_rl.sft.summarize_benchmarks import CASES, summarize, summarize_pair


ROOT = Path(__file__).resolve().parents[1]


class SFTConfigAndSummaryTests(unittest.TestCase):
    def test_smoke_config_caps_data_without_changing_production_config(self) -> None:
        production = load_config(ROOT / "configs" / "sft" / "layer10_whole_layer_baseline_sft.yaml")
        smoke = load_config(ROOT / "configs" / "sft" / "smoke_layer10_whole_layer_baseline_sft.yaml")

        self.assertNotIn("max_packed_sequences", production["sft"])
        self.assertEqual(production["freeze_policy"]["train_layers"], [10])
        self.assertEqual(smoke["sft"]["max_packed_sequences"], 16)
        self.assertEqual(smoke["sft"]["checkpoint_fractions"], [0.5, 1.0])

    def test_all_sft_variants_share_training_contract(self) -> None:
        expected = {
            "layer10_whole_layer_baseline_sft.yaml": "identity",
            "layer10_whole_layer_shs_sft.yaml": "qwen_swiglu_shs",
            "layer10_whole_layer_triglu_side_ffn_sft.yaml": "qwen_swiglu_triglu_side",
            "layer10_whole_layer_oft_sft.yaml": "qwen_swiglu_oft",
        }
        contracts = []
        for file_name, variant in expected.items():
            cfg = load_config(ROOT / "configs" / "sft" / file_name)
            self.assertEqual(cfg["architecture_variant"]["name"], variant)
            self.assertEqual(cfg["experiment"]["seed"], 20260707)
            self.assertEqual(cfg["dataset"]["dataloader_seed"], 20260707)
            contracts.append(
                (
                    cfg["dataset"]["sft_train"],
                    cfg["sft"]["epochs"],
                    cfg["sft"]["max_seq_length"],
                    cfg["sft"]["per_device_micro_batch_size"],
                    cfg["sft"]["gradient_accumulation_steps"],
                    tuple(cfg["sft"]["checkpoint_fractions"]),
                )
            )
        self.assertEqual(len(set(contracts)), 1)

    def test_checkpoint_steps_cover_whole_run(self) -> None:
        try:
            from qwen_single_layer_rl.sft.trainer import _checkpoint_steps
        except ImportError as exc:
            if exc.name == "torch":
                self.skipTest("torch is required for trainer checkpoint tests")
            raise
        self.assertEqual(
            _checkpoint_steps(392, [0.10, 0.25, 0.50, 0.75, 1.0]),
            [40, 98, 196, 294, 392],
        )

    def test_benchmark_summary_computes_speedups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            medians = {
                "baseline_eager": 4.0,
                "baseline_compile": 2.0,
                "shs_eager": 12.0,
                "shs_compile": 6.0,
            }
            for case in CASES:
                case_dir = root / case
                case_dir.mkdir()
                (case_dir / "benchmark_result.json").write_text(
                    json.dumps(
                        {
                            "compile_mode": "eager" if case.endswith("eager") else "default",
                            "correctness_loss": 1.0,
                            "cold_step_seconds": 10.0,
                            "step_seconds_median": medians[case],
                            "step_seconds_p90": medians[case],
                            "max_memory_allocated_gb": 1.0,
                            "timed_assistant_tokens_per_second": 100.0,
                        }
                    ),
                    encoding="utf-8",
                )
            report = summarize(root)
        self.assertEqual(report["comparisons"]["baseline_compile_speedup"], 2.0)
        self.assertEqual(report["comparisons"]["shs_compile_speedup"], 2.0)
        self.assertEqual(report["comparisons"]["shs_eager_overhead_ratio"], 3.0)

    def test_pair_summary_handles_compile_speedup_and_slowdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for case, median, cold in (("shs_eager", 4.0, 1.0), ("shs_compile", 2.0, 10.0)):
                case_dir = root / case
                case_dir.mkdir()
                (case_dir / "benchmark_result.json").write_text(
                    json.dumps(
                        {
                            "step_seconds_median": median,
                            "cold_step_seconds": cold,
                            "correctness_loss": 1.0,
                        }
                    ),
                    encoding="utf-8",
                )
            faster = summarize_pair(root, "shs")
            self.assertEqual(faster["comparisons"]["compile_speedup"], 2.0)
            self.assertEqual(faster["comparisons"]["compile_break_even_steps"], 5.0)

            compiled_path = root / "shs_compile" / "benchmark_result.json"
            compiled = json.loads(compiled_path.read_text(encoding="utf-8"))
            compiled["step_seconds_median"] = 5.0
            compiled_path.write_text(json.dumps(compiled), encoding="utf-8")
            slower = summarize_pair(root, "shs")
            self.assertIsNone(slower["comparisons"]["compile_break_even_steps"])


if __name__ == "__main__":
    unittest.main()
