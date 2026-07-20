from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from qwen_single_layer_rl.data.build_selection_provenance import (
    trace_selection,
    validation_source_splits,
)
from qwen_single_layer_rl.data.prep_numina import normalize_numina_record


class SelectionProvenanceTests(unittest.TestCase):
    def test_traces_first_source_occurrence_and_output_order(self) -> None:
        source = [
            {"source": "a", "problem": "First", "solution": "Answer is 1."},
            {"source": "b", "problem": "Second", "solution": "Answer is 2."},
            {"source": "a", "problem": "First", "solution": "Different duplicate."},
        ]
        selected = [normalize_numina_record(source[1]), normalize_numina_record(source[0])]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.jsonl"
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected),
                encoding="utf-8",
            )
            rows, stats = trace_selection(
                role="train",
                materialized_path=path,
                source_split="train",
                source_records=source,
                expected_source_rows=3,
                progress_every=0,
            )

        self.assertEqual([row["materialized_index"] for row in rows], [0, 1])
        self.assertEqual([row["source_index"] for row in rows], [1, 0])
        self.assertEqual(stats["matched_count"], 2)
        self.assertTrue(stats["all_normalized_records_match"])

    def test_rejects_changed_source_content_for_same_problem(self) -> None:
        source = [{"source": "a", "problem": "Same", "solution": "New solution"}]
        selected = normalize_numina_record(
            {"source": "a", "problem": "Same", "solution": "Original solution"}
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.jsonl"
            path.write_text(json.dumps(selected) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from materialized"):
                trace_selection(
                    role="train",
                    materialized_path=path,
                    source_split="train",
                    source_records=source,
                    expected_source_rows=1,
                    progress_every=0,
                )

    def test_validation_manifest_assigns_each_candidate_to_its_online_split(self) -> None:
        manifest = {
            "selected_sources": [
                {"candidate_path": "/remote/data/original/val.jsonl"},
                {"candidate_path": "/remote/data/original/train.jsonl"},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "validation_manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            assignments, mapping = validation_source_splits(
                validation_count=2,
                default_split="test",
                selection_manifest=path,
                candidate_mappings=["original/val.jsonl=test", "original/train.jsonl=train"],
            )

        self.assertEqual(assignments, ["test", "train"])
        self.assertEqual(mapping["original/val.jsonl"], "test")


if __name__ == "__main__":
    unittest.main()
