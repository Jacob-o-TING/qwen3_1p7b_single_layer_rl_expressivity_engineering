from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "scripts" / "summarize_ood_eval.py"
SPEC = importlib.util.spec_from_file_location("summarize_ood_eval", SUMMARY_PATH)
assert SPEC and SPEC.loader
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


class OODStep294QueueTests(unittest.TestCase):
    def test_summary_uses_unweighted_paper_categories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            values = {
                "humaneval_plus": 0.4,
                "mbpp": 0.5,
                "live_code_bench": 0.1,
                "gpqa_diamond": 0.2,
                "mmlu_pro": 0.4,
                "ceval": 0.6,
                "ifeval": 0.3,
                "mgsm": 0.6,
            }
            for name, score in values.items():
                path = root / name / "reports" / "model" / f"{name}.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"dataset_name": name, "score": score, "num": 10}))
            (root / "PARALLEL_OOD_EVAL_COMPLETE").touch()
            summary = SUMMARY.build_summary(root)
        self.assertEqual(summary["status"], "complete")
        self.assertAlmostEqual(summary["category_scores"]["code"], 1 / 3)
        self.assertAlmostEqual(summary["category_scores"]["reasoning"], 0.3)
        self.assertAlmostEqual(summary["category_scores"]["language"], 0.5)

    def test_parser_sensitive_code_comparison_uses_corrected_lcb_and_changes_only_heplus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            values = {
                "humaneval_plus": 0.0,
                "mbpp": 0.292,
                "live_code_bench": 0.0,
            }
            for name, score in values.items():
                path = root / name / "reports" / "model" / f"{name}.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"dataset_name": name, "score": score, "num": 10}))
            he = root / "diagnostics/humanevalplus_parser_v2_reviewonly_test/summary.json"
            he.parent.mkdir(parents=True)
            he.write_text(json.dumps({"rows": 164, "parser_only_score": 36 / 164}))
            lcb = (
                root
                / "diagnostics/triglu_livecodebench_cached_reviewonly_sandbox_output_contract_test"
                / "summary.json"
            )
            lcb.parent.mkdir(parents=True)
            lcb.write_text(
                json.dumps(
                    {
                        "rows": 1055,
                        "corrected_score": 107 / 1055,
                        "source_predictions_unchanged": True,
                    }
                )
            )

            summary = SUMMARY.build_summary(root)
            comparison = summary["parser_sensitive_code_comparison"]

        self.assertAlmostEqual(comparison["humaneval_plus_pre_parser"], 0.0)
        self.assertAlmostEqual(comparison["humaneval_plus_post_parser"], 36 / 164)
        self.assertAlmostEqual(comparison["live_code_bench_corrected"], 107 / 1055)
        self.assertAlmostEqual(comparison["code_avg_pre_parser"], (0 + 0.292 + 107 / 1055) / 3)
        self.assertAlmostEqual(
            comparison["code_avg_post_parser"],
            (36 / 164 + 0.292 + 107 / 1055) / 3,
        )
        self.assertAlmostEqual(summary["benchmarks"]["humaneval_plus"]["score"], 36 / 164)
        self.assertAlmostEqual(summary["benchmarks"]["live_code_bench"]["score"], 107 / 1055)
        self.assertAlmostEqual(
            summary["category_scores"]["code"],
            comparison["code_avg_post_parser"],
        )

    def test_prompt_corrected_humaneval_is_exposed_without_overwriting_chat_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            root = project_root / "ood" / "triglu"
            values = {
                "humaneval_plus": 0.2,
                "mbpp": 0.3,
                "live_code_bench": 0.1,
            }
            for name, score in values.items():
                path = root / name / "reports" / "model" / f"{name}.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"dataset_name": name, "score": score, "num": 10}))
            lcb = (
                root
                / "diagnostics/triglu_livecodebench_cached_reviewonly_sandbox_output_contract_test"
                / "summary.json"
            )
            lcb.parent.mkdir(parents=True)
            lcb.write_text(
                json.dumps(
                    {
                        "rows": 1055,
                        "corrected_score": 0.1,
                        "source_predictions_unchanged": True,
                    }
                )
            )
            protocol = (
                project_root
                / "runs/eval_protocol"
                / SUMMARY.PROMPT_PROTOCOL_RUN
                / "models/triglu_step294/summary.json"
            )
            protocol.parent.mkdir(parents=True)
            protocol.write_text(
                json.dumps(
                    {
                        "sample_count": 164,
                        "cells": {
                            SUMMARY.PROMPT_PROTOCOL_CELL: {
                                "rows": 164,
                                "score": 100 / 164,
                            }
                        },
                    }
                )
            )
            summary = SUMMARY.build_summary(root, project_root=project_root)
            comparison = summary["parser_sensitive_code_comparison"]

        self.assertAlmostEqual(comparison["humaneval_plus_prompt_corrected"], 100 / 164)
        self.assertAlmostEqual(
            comparison["code_avg_prompt_corrected"],
            (100 / 164 + 0.3 + 0.1) / 3,
        )
        self.assertAlmostEqual(summary["benchmarks"]["humaneval_plus"]["score"], 0.2)

    def test_prompt_codeavg_falls_back_to_exact_merged_reports_without_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp)
            root = project_root / "qwen3_1p7b_ood_6x5090_baseline_step196_test"
            for name, score in {
                "humaneval_plus": 51 / 164,
                "mbpp": 0.282,
                "live_code_bench": 0.0976,
            }.items():
                path = root / name / "reports" / "model" / f"{name}.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({"dataset_name": name, "score": score, "num": 10}))
            protocol = (
                project_root
                / "runs/eval_protocol"
                / SUMMARY.PROMPT_PROTOCOL_RUN
                / "models/baseline_step196/summary.json"
            )
            protocol.parent.mkdir(parents=True)
            protocol.write_text(
                json.dumps(
                    {
                        "sample_count": 164,
                        "cells": {
                            SUMMARY.PROMPT_PROTOCOL_CELL: {
                                "rows": 164,
                                "score": 100 / 164,
                            }
                        },
                    }
                )
            )
            summary = SUMMARY.build_summary(root, project_root=project_root)
            comparison = summary["parser_sensitive_code_comparison"]

        self.assertAlmostEqual(comparison["humaneval_plus_post_parser"], 51 / 164)
        self.assertAlmostEqual(comparison["live_code_bench_corrected"], 0.0976)
        self.assertAlmostEqual(
            comparison["code_avg_prompt_corrected"],
            (100 / 164 + 0.282 + 0.0976) / 3,
        )

    def test_queue_supports_split_owner_approved_order_and_six_gpu_partition(self) -> None:
        controller = (ROOT / "scripts/run_qwen3_1p7b_ood_6x5090_step294_20260716_v1.sh").read_text()
        autostart = (ROOT / "scripts/autostart_qwen3_1p7b_ood_6x5090_step294_20260716_v1.sh").read_text()
        parallel = (ROOT / "scripts/run_parallel_ood_vllm_eval_6gpu_20260716_v1.sh").read_text()
        lcb_parallel = (
            ROOT / "scripts/run_livecodebench_parallel6_vllm_20260717_v1.sh"
        ).read_text()
        self.assertIn("requested_models=(\"$@\")", controller)
        self.assertIn("requested_models=(triglu untuned_base baseline)", controller)
        pre = autostart.index('bash "$OOD_RUNNER" triglu untuned_base')
        baseline_wait = autostart.index("baseline_math_step294", pre)
        baseline = autostart.index('bash "$OOD_RUNNER" baseline', baseline_wait)
        self.assertEqual([pre, baseline_wait, baseline], sorted([pre, baseline_wait, baseline]))
        for gpu in range(5):
            self.assertIn(f"run_rank {gpu} ", parallel)
        self.assertIn("run_livecodebench_parallel6_vllm_20260717_v1.sh", parallel)
        self.assertIn("--local-code-sandbox", parallel)
        self.assertIn("humaneval_plus mbpp", parallel)
        self.assertIn("live_code_bench", lcb_parallel)

    def test_autostart_interposes_between_triglu_math_and_baseline_training(self) -> None:
        text = (ROOT / "scripts/autostart_qwen3_1p7b_ood_6x5090_step294_20260716_v1.sh").read_text()
        controller = (ROOT / "scripts/run_triglu_priority_to294_then_baseline196_20260715_v1.sh").read_text()
        self.assertIn("triglu_math_step294", text)
        self.assertIn("fail_safe_exit", text)
        self.assertIn("trap fail_safe_exit EXIT HUP INT TERM", text)
        self.assertIn("PRE_BASELINE_OOD_COMPLETE", text)
        self.assertIn("TRIGLU_294_PRE_BASELINE_OOD_READY", text)
        self.assertNotIn("kill -STOP", text)
        self.assertIn("baseline_math_step294", text)
        self.assertIn('while [[ ! -f "$GRPO_ROOT/WAVE_COMPLETE" ]]', text)
        barrier = controller.index("wait_for_pre_baseline_ood")
        baseline = controller.index("prepare_third_stage_transition baseline", barrier)
        self.assertLess(barrier, baseline)
        self.assertIn('while [[ ! -f "$OOD_ROOT/PRE_BASELINE_OOD_COMPLETE" ]]', controller)

    def test_monitor_marks_historical_livecodebench_zero_as_invalid(self) -> None:
        monitor = (
            ROOT
            / "scripts/monitor_triglu_priority_to294_then_baseline196_20260715_v1.sh"
        ).read_text()
        summarizer = (ROOT / "scripts/summarize_ood_eval.py").read_text()
        self.assertIn("INVALID (sandbox output-contract artifact)", monitor)
        self.assertIn("scripts/summarize_ood_eval.py", monitor)
        self.assertIn("corrected LiveCodeBench", summarizer)
        self.assertIn('current_model=$(awk -F=', monitor)
        self.assertIn("current eval: $current_model / $active_cell", monitor)

    def test_ood_launch_fails_fast_when_ifeval_dependency_is_missing(self) -> None:
        parallel = (ROOT / "scripts/run_parallel_ood_vllm_eval_6gpu_20260716_v1.sh").read_text()
        self.assertIn("import langdetect", parallel)


if __name__ == "__main__":
    unittest.main()
