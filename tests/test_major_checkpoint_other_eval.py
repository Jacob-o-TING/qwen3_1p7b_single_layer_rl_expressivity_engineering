from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_qwen3_1p7b_other_eval_majorsteps_6x5090_20260718_v1.sh"
MONITOR = ROOT / "scripts" / "monitor_qwen3_1p7b_other_eval_majorsteps_6x5090_20260718_v1.sh"
UNIFIED_MONITOR = ROOT / "scripts" / "monitor_triglu_priority_to294_then_baseline196_20260715_v1.sh"
CONFIG = ROOT / "configs" / "eval" / "qwen3_1p7b_other_eval_majorsteps_6x5090_20260718_v1.yaml"
DASHBOARD = ROOT / "scripts" / "summarize_qwen_eval_dashboard.py"
SUMMARIZER_PATH = ROOT / "scripts" / "summarize_ood_eval.py"
SPEC = importlib.util.spec_from_file_location("summarize_ood_eval", SUMMARIZER_PATH)
assert SPEC and SPEC.loader
SUMMARIZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARIZER)


class MajorCheckpointOtherEvalTests(unittest.TestCase):
    def test_approved_ledger_is_exact(self) -> None:
        text = CONFIG.read_text(encoding="utf-8")
        self.assertIn("variants: [triglu, baseline]", text)
        self.assertIn("global_steps: [158, 196, 226, 256, 294]", text)
        self.assertIn("show_ood_8_average: false", text)
        self.assertIn("show_ood_category_average: false", text)
        self.assertNotIn("global_steps: [158, 196, 294]", text)

    def test_runner_executes_only_eight_missing_cells_in_paired_order(self) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        ordered = (
            "run_cell triglu 158",
            "run_cell baseline 158",
            "run_cell triglu 196",
            "run_cell triglu 226",
            "run_cell baseline 226",
            "run_cell triglu 256",
            "run_cell baseline 256",
            "run_cell baseline 294",
        )
        offsets = [text.index(item) for item in ordered]
        self.assertEqual(offsets, sorted(offsets))
        self.assertNotIn("run_cell baseline 196", text)
        self.assertNotIn("run_cell triglu 294", text)
        self.assertIn("import_cell triglu_step294", text)
        self.assertIn("import_cell baseline_step196", text)

    def test_runner_preserves_staged_six_gpu_and_primary_protocol(self) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        self.assertIn("for gpu in 0 1 2 3 4 5", text)
        self.assertIn("--dataset-shard-count 6", text)
        self.assertIn("--humanevalplus-parser-v2", text)
        self.assertIn("evalscope_raw_instruction_nochat", text)
        self.assertIn("run_livecodebench_parallel6_vllm_20260717_v1.sh", text)
        self.assertIn("minimum_free_gb_before_cell: 40", CONFIG.read_text(encoding="utf-8"))

    def test_strict_unset_mode_does_not_expand_dependent_locals_early(self) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        forbidden = (
            'path="$PRIORITY_ROOT/exports/${variant}_step_${step}"',
            'label="${variant}_step${step}" cell=',
            'label="$2" receipt="$OUT/receipts/${label}',
            'corrected="$3" link="$OUT/cells/${label',
        )
        for fragment in forbidden:
            for line in text.splitlines():
                if line.lstrip().startswith("local "):
                    self.assertNotIn(fragment, line)

    def test_local_prompt_protocol_summary_is_preferred(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local = root / "primary_humanevalplus"
            local.mkdir()
            (local / "summary.json").write_text(
                json.dumps(
                    {
                        "sample_count": 164,
                        "cells": {
                            "evalscope_raw_instruction_nochat": {
                                "rows": 164,
                                "score": 0.625,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            score, path = SUMMARIZER._prompt_protocol_summary(root)
        self.assertEqual(score, 0.625)
        self.assertIsNotNone(path)
        self.assertEqual(Path(str(path)).parts[-2:], ("primary_humanevalplus", "summary.json"))

    def test_monitor_groups_architectures_by_step_without_hard_averages(self) -> None:
        text = MONITOR.read_text(encoding="utf-8")
        self.assertIn("steps = (158, 196, 226, 256, 294)", text)
        self.assertIn('variants = ("triglu", "baseline")', text)
        self.assertIn("PRIMARY corrected-protocol category comparison", text)
        self.assertIn("HERITAGE chat-protocol code view", text)
        self.assertIn(
            "architecture  HumanEval+  MBPP       LCB        GPQA-Diamond  GPQA-Freeform",
            text,
        )
        self.assertIn("GPQA-Freeform queue:", text)
        self.assertIn('freeform_cells = freeform_summary.get("cells", {})', text)
        self.assertIn('get("accuracy_strict")', text)
        self.assertIn("never averaged into it", text)
        self.assertIn("ACROSS-CHECKPOINT descriptive summary", text)
        self.assertIn("statistics.pstdev", text)
        self.assertIn("TriGLU", text)
        self.assertIn("baseline", text)
        self.assertIn("BENCHMARK_DISPLAY=GPQA-Diamond", text)
        self.assertNotIn("OOD-8", text)
        self.assertNotIn("OOD-category", text)
        dashboard = DASHBOARD.read_text(encoding="utf-8")
        self.assertNotIn('print("  model             HumanEval+       CodeAvg     reasoning    language     OOD-8', dashboard)
        self.assertIn("monitor_qwen3_1p7b_other_eval_majorsteps_6x5090_20260718_v1.sh", UNIFIED_MONITOR.read_text())


if __name__ == "__main__":
    unittest.main()
