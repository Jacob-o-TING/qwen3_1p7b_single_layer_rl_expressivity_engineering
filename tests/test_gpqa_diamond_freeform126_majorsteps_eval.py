from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qwen_single_layer_rl.eval.gpqa_freeform import (
    build_matcher_ledger,
    extract_answer_tag,
    merge_exact_shards,
    normalize_dataset_rows,
    parse_matcher_decision,
    select_shard,
    summarize_matches,
    write_jsonl_atomic,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/eval/qwen3_1p7b_gpqa_diamond_freeform126_majorsteps_6x5090_20260718_v1.yaml"
RUNNER = ROOT / "scripts/run_qwen3_1p7b_gpqa_diamond_freeform126_majorsteps_6x5090_20260718_v1.sh"
MONITOR = ROOT / "scripts/monitor_qwen3_1p7b_gpqa_diamond_freeform126_majorsteps_6x5090_20260718_v1.sh"
ASSETS = ROOT / "scripts/prepare_gpqa_diamond_freeform126_assets.py"


def source_rows() -> list[dict[str, str]]:
    return [
        {"question_id": f"q-{index:03d}", "question": f"Question {index}?", "answer": f"Answer {index}"}
        for index in range(126)
    ]


class GPQAFreeformProtocolTests(unittest.TestCase):
    def test_dataset_ledger_is_exact_and_balanced(self) -> None:
        rows = normalize_dataset_rows(reversed(source_rows()))
        self.assertEqual(len(rows), 126)
        self.assertEqual([row["question_id"] for row in rows], sorted(row["question_id"] for row in rows))
        self.assertEqual([len(select_shard(rows, rank=rank, shard_count=6)) for rank in range(6)], [21] * 6)

    def test_dataset_rejects_duplicate_and_choice_scaffold(self) -> None:
        duplicate = source_rows()
        duplicate[-1]["question_id"] = duplicate[0]["question_id"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            normalize_dataset_rows(duplicate)
        choices = source_rows()
        choices[0]["question"] = "Answer Choices: A or B"
        with self.assertRaisesRegex(ValueError, "choice scaffold"):
            normalize_dataset_rows(choices)

    def test_answer_and_matcher_tags_are_fail_closed(self) -> None:
        self.assertEqual(extract_answer_tag("work <answer> final </answer>"), "final")
        self.assertIsNone(extract_answer_tag("no tag"))
        self.assertEqual(parse_matcher_decision("<answer>1</answer>"), 1)
        self.assertEqual(parse_matcher_decision("<answer>0</answer>"), 0)
        self.assertIsNone(parse_matcher_decision("reasoning <answer>1</answer>"))
        self.assertIsNone(parse_matcher_decision("<answer>1</answer><answer>0</answer>"))

    def test_generation_merge_and_matcher_ledger_have_exact_coverage(self) -> None:
        ledger = normalize_dataset_rows(source_rows())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shards = []
            for rank in range(6):
                path = root / f"generation_{rank}.jsonl"
                rows = [
                    {**row, "raw_response": "<answer>x</answer>", "parsed_answer": "x"}
                    for row in select_shard(ledger, rank=rank, shard_count=6)
                ]
                write_jsonl_atomic(path, rows)
                shards.append(path)
            merged = merge_exact_shards(
                shards,
                expected_rows=126,
                expected_ids=[row["question_id"] for row in ledger],
                identity_key="question_id",
            )
            cells = []
            for step in (158, 196, 226, 256, 294):
                for variant in ("triglu", "baseline"):
                    path = root / f"{variant}_{step}.jsonl"
                    write_jsonl_atomic(path, merged)
                    cells.append((variant, step, path))
            matcher = build_matcher_ledger(cells)
        self.assertEqual(len(matcher), 1260)
        self.assertEqual([len(select_shard(matcher, rank=rank, shard_count=6)) for rank in range(6)], [210] * 6)

    def test_summary_closes_matcher_failures_inside_denominator(self) -> None:
        rows = []
        for step in (158, 196, 226, 256, 294):
            for variant in ("triglu", "baseline"):
                for index in range(126):
                    decision = None if index == 0 else int(index % 2 == 0)
                    rows.append(
                        {
                            "global_step": step,
                            "variant": variant,
                            "matcher_decision": decision,
                            "generation_cap_hit": False,
                            "parsed_candidate_answer": "x",
                        }
                    )
        summary = summarize_matches(rows)
        self.assertEqual(summary["rows"], 1260)
        self.assertEqual(summary["cells"]["triglu_step158"]["matcher_failures"], 1)
        self.assertEqual(summary["cells"]["triglu_step158"]["total"], 126)

    def test_approved_config_and_execution_order_are_immutable(self) -> None:
        config = CONFIG.read_text(encoding="utf-8")
        self.assertIn("global_steps: [158, 196, 226, 256, 294]", config)
        self.assertIn("rows_per_replica: 21", config)
        self.assertIn("rows_per_replica: 210", config)
        self.assertIn("max_tokens: 3072", config)
        self.assertIn("public_fallback_repo_id: nikhilchandak/GPQA-diamond-free", config)
        runner = RUNNER.read_text(encoding="utf-8")
        ordered = [
            f"run_generation_cell {variant} {step}"
            for step in (158, 196, 226, 256, 294)
            for variant in ("triglu", "baseline")
        ]
        offsets = [runner.index(value) for value in ordered]
        self.assertEqual(offsets, sorted(offsets))
        self.assertIn("for gpu in 0 1 2 3 4 5", runner)
        self.assertIn("WAITING_ACTIVE_OTHER_OOD_WAVE", runner)
        self.assertIn("OTHER_OOD_SCREEN_DRAINING_AFTER_WAVE_COMPLETE", runner)
        self.assertIn('OTHER_EVAL_OUT/WAVE_COMPLETE', runner)
        self.assertIn("FREEFORM_GPU_BUSY_REFUSING_TO_CONTEND", runner)
        self.assertIn("GPQA_PREPARE_ASSETS", runner)

    def test_monitor_matches_existing_dashboard_grammar(self) -> None:
        monitor = MONITOR.read_text(encoding="utf-8")
        for phrase in (
            "controller:",
            "phase:",
            "current cell:",
            "overall:",
            "recent speed:",
            "ETA:",
            "PRIMARY GPQA-Diamond-Freeform-126 paired comparison:",
            "GPU utilization / memory:",
            "disk free:",
        ):
            self.assertIn(phrase, monitor)
        self.assertIn("correct/126", monitor)
        self.assertIn("delta(TriGLU-baseline)", monitor)
        self.assertNotIn("hard average", monitor.lower())

    def test_asset_script_never_serializes_hf_token(self) -> None:
        text = ASSETS.read_text(encoding="utf-8")
        self.assertIn('os.environ.get("HF_TOKEN")', text)
        self.assertIn('"credentials_recorded": False', text)
        self.assertIn("maintainer_public_filtered_csv_fallback", text)
        self.assertIn('row["Record ID"]', text)


if __name__ == "__main__":
    unittest.main()
