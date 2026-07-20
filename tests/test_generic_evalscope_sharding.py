from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from qwen_single_layer_rl.eval.generic_evalscope_sharding import (
    canonical_sample_identity,
    formatted_sample_sha256,
    shard_for_identity,
)


ROOT = Path(__file__).resolve().parents[1]
MERGE_PATH = ROOT / "scripts" / "merge_generic_evalscope_shards.py"
SPEC = importlib.util.spec_from_file_location("merge_generic_evalscope_shards", MERGE_PATH)
assert SPEC and SPEC.loader
MERGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MERGE)


class GenericEvalScopeShardingTests(unittest.TestCase):
    def test_identity_is_deterministic_and_namespaced(self) -> None:
        sample = {"input": "question", "target": "answer"}
        first = canonical_sample_identity("mmlu_pro", "math", 7, sample)
        repeated = canonical_sample_identity("mmlu_pro", "math", 7, dict(reversed(sample.items())))
        rank_local_format = canonical_sample_identity(
            "mmlu_pro", "math", 7, {"input": "question", "target": "B"}
        )
        other = canonical_sample_identity("gpqa_diamond", "math", 7, sample)
        self.assertEqual(first, repeated)
        self.assertEqual(first, rank_local_format)
        self.assertNotEqual(first, other)
        self.assertEqual(len(first), 64)
        self.assertNotEqual(
            formatted_sample_sha256(sample),
            formatted_sample_sha256({"input": "question", "target": "B"}),
        )

    def test_merge_requires_exact_disjoint_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = 0
            weighted = 0.0
            for shard_index in range(6):
                shard = root / "shards" / f"shard_{shard_index:02d}"
                report_dir = shard / "main" / "timestamp" / "reports" / "model"
                report_dir.mkdir(parents=True)
                identities = []
                candidate = 0
                while len(identities) < shard_index + 1:
                    identity = hashlib.sha256(str(candidate).encode("ascii")).hexdigest()
                    candidate += 1
                    if shard_for_identity(identity, 6) == shard_index:
                        identities.append(identity)
                with (shard / "dataset_shard_receipt.jsonl").open("w", encoding="utf-8") as handle:
                    for sample_index, identity in enumerate(identities):
                        handle.write(
                            json.dumps(
                                {
                                    "dataset": "mmlu_pro",
                                    "formatted_sample_sha256": hashlib.sha256(
                                        f"payload-{sample_index}".encode("ascii")
                                    ).hexdigest(),
                                    "identity_sha256": identity,
                                    "sample_index": sample_index,
                                    "shard_count": 6,
                                    "shard_index": shard_index,
                                    "subset": "test",
                                }
                            )
                            + "\n"
                        )
                score = shard_index / 10
                (report_dir / "mmlu_pro.json").write_text(
                    json.dumps(
                        {
                            "dataset_name": "mmlu_pro",
                            "num": len(identities),
                            "score": score,
                        }
                    ),
                    encoding="utf-8",
                )
                (shard / "RANK_COMPLETE").touch()
                expected += len(identities)
                weighted += score * len(identities)

            summary = MERGE.merge_shards(
                root,
                dataset="mmlu_pro",
                shard_count=6,
                expected_identities=expected,
            )

        self.assertEqual(summary["identity_unique"], expected)
        self.assertEqual(summary["identity_duplicates"], 0)
        self.assertAlmostEqual(summary["score"], weighted / expected)

    def test_baseline_step196_runner_stages_each_benchmark_across_six_gpus(self) -> None:
        text = (
            ROOT / "scripts" / "run_baseline_step196_staged6_ood_20260717_v1.sh"
        ).read_text()
        self.assertIn('for gpu in 0 1 2 3 4 5', text)
        self.assertIn("--dataset-shard-count 6", text)
        self.assertIn("--humanevalplus-parser-v2", text)
        self.assertIn("run_stage 2 reasoning_mmlupro_staged6 mmlu_pro 12032", text)
        self.assertLess(text.index("run_stage 1 "), text.index("run_stage 2 "))
        self.assertLess(text.index("run_stage 2 "), text.index("run_stage 3 "))

    def test_ood_monitor_includes_baseline_step196_partial_results(self) -> None:
        text = (
            ROOT / "scripts" / "monitor_qwen3_1p7b_ood_6x5090_step294_20260716_v1.sh"
        ).read_text()
        self.assertIn('echo "--- baseline_step196 staged detail ---"', text)
        self.assertIn("summarize_qwen_eval_dashboard.py", text)
        self.assertIn("not final until exact merge", text)
        self.assertIn("final=", text)
        self.assertIn("partial=", text)


if __name__ == "__main__":
    unittest.main()
