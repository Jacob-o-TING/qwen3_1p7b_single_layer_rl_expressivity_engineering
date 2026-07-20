from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/eval/qwen3_1p7b_gpqa_diamond_freeform126_manual_agent_audit_majorsteps_6x5090_20260718_v1.yaml"
RUNNER = ROOT / "scripts/run_qwen3_1p7b_gpqa_diamond_freeform126_manual_agent_audit_6x5090_20260718_v1.sh"
MONITOR = ROOT / "scripts/monitor_qwen3_1p7b_gpqa_diamond_freeform126_manual_agent_audit_6x5090_20260718_v1.sh"
BUILDER_PATH = ROOT / "scripts/build_gpqa_diamond_freeform126_manual_audit_package.py"
MERGER_PATH = ROOT / "scripts/merge_gpqa_diamond_freeform126_manual_audit.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = load_module("gpqa_manual_builder", BUILDER_PATH)
MERGER = load_module("gpqa_manual_merger", MERGER_PATH)


class GPQAManualAgentAuditTests(unittest.TestCase):
    def test_approved_identity_and_matcher_free_contract(self) -> None:
        config = CONFIG.read_text(encoding="utf-8")
        self.assertIn("manual_agent_audit_6x5090_steps158_196_226_256_294", config)
        self.assertIn("screen_name: qwen_gpqa_free126_manualaudit_gen6_20260718_v1", config)
        self.assertIn("expected_rows: 1260", config)
        self.assertIn("question_chunks: 126", config)
        self.assertIn("enforce_eager: true", config)
        self.assertNotIn("matcher:", config)
        self.assertNotIn("Qwen3-4B", config)

    def test_controller_runs_exact_ten_cell_order_and_stops_for_audit(self) -> None:
        text = RUNNER.read_text(encoding="utf-8")
        ordered = [
            f"run_generation_cell {variant} {step}"
            for step in (158, 196, 226, 256, 294)
            for variant in ("triglu", "baseline")
        ]
        offsets = [text.index(value) for value in ordered]
        self.assertEqual(offsets, sorted(offsets))
        self.assertIn("for gpu in 0 1 2 3 4 5", text)
        self.assertIn('VLLM_CACHE_ROOT="$OUT/compile_cache/vllm/rank_${gpu}"', text)
        self.assertIn('TORCHINDUCTOR_CACHE_DIR="$OUT/compile_cache/torchinductor/rank_${gpu}"', text)
        self.assertIn("GPQA_PREPARE_DATASET", text)
        self.assertIn("prepare_gpqa_diamond_freeform126_assets.py", text)
        self.assertIn(" dataset | tee", text)
        self.assertNotIn("run_matching", text)
        self.assertNotIn("Qwen3-4B", text)
        self.assertIn("AWAITING_MANUAL_AUDIT", text)
        self.assertIn("AUDIT_PACKAGE_READY", text)

    def test_question_centric_rows_close_exactly(self) -> None:
        cells = [
            {"label": f"{variant}_step{step}", "variant": variant, "global_step": step}
            for step in (158, 196, 226, 256, 294)
            for variant in ("triglu", "baseline")
        ]
        config = {"cells": cells}
        ledger = [
            {"question_id": f"q{index:03d}", "question": f"Question {index}", "reference_answer": f"A{index}"}
            for index in range(126)
        ]
        generation = {
            cell["label"]: [
                {
                    "question_id": row["question_id"],
                    "raw_response": f"response {cell['label']} {row['question_id']}",
                    "parsed_answer": row["reference_answer"],
                    "generated_tokens": 5,
                    "finish_reason": "stop",
                    "cap_hit": False,
                    "request_seed": index,
                    "generation_receipt_sha256": "a" * 64,
                }
                for index, row in enumerate(ledger)
            ]
            for cell in cells
        }
        rows = BUILDER.build_audit_rows(config, ledger, generation)
        self.assertEqual(len(rows), 1260)
        self.assertEqual(len({row["audit_row_id"] for row in rows}), 1260)
        self.assertEqual([row["question_index"] for row in rows[:10]], [0] * 10)
        self.assertEqual([row["variant"] for row in rows[:4]], ["triglu", "baseline", "triglu", "baseline"])

    def test_verdict_schema_preserves_uncertain(self) -> None:
        row = {
            "audit_row_id": "row-1",
            "response_read_completely": True,
            "semantic_correctness": "uncertain",
            "answer_extraction": "uncertain",
            "failure_category": "reference_or_question_ambiguity",
            "confidence": "low",
            "evidence_note": "Reference equivalence is not established.",
        }
        MERGER.validate_verdict(row, {"row-1"})
        invalid = dict(row, semantic_correctness="probably")
        with self.assertRaisesRegex(ValueError, "semantic_correctness"):
            MERGER.validate_verdict(invalid, {"row-1"})

    def test_monitor_never_substitutes_matcher_score(self) -> None:
        text = MONITOR.read_text(encoding="utf-8")
        self.assertIn("generation:", text)
        self.assertIn("correctness remains pending until complete row-by-row audit", text)
        self.assertIn("across checkpoints (population std)", text)
        self.assertNotIn("Qwen3-4B", text)


if __name__ == "__main__":
    unittest.main()
