from __future__ import annotations

import importlib.util
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from qwen_single_layer_rl.eval.live_code_bench_sharding import (
    canonical_lcb_record_identity,
    shard_for_identity,
)


ROOT = Path(__file__).resolve().parents[1]
MERGE_PATH = ROOT / "scripts" / "merge_livecodebench_shards.py"
SPEC = importlib.util.spec_from_file_location("merge_livecodebench_shards", MERGE_PATH)
assert SPEC and SPEC.loader
MERGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MERGE)


class LiveCodeBenchShardingTests(unittest.TestCase):
    def test_identity_and_assignment_are_deterministic(self) -> None:
        record = {
            "contest_date": "2025-01-02T00:00:00",
            "contest_id": "contest-1",
            "platform": "codeforces",
            "question_content": "Solve it.",
            "question_id": "problem-a",
        }
        identity = canonical_lcb_record_identity(record)
        self.assertEqual(identity, canonical_lcb_record_identity(dict(reversed(record.items()))))
        self.assertEqual(len(identity), 64)
        self.assertEqual(shard_for_identity(identity, 6), shard_for_identity(identity, 6))

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
                    for identity in identities:
                        handle.write(
                            json.dumps(
                                {
                                    "identity_sha256": identity,
                                    "shard_count": 6,
                                    "shard_index": shard_index,
                                }
                            )
                            + "\n"
                        )
                score = shard_index / 10
                (report_dir / "live_code_bench.json").write_text(
                    json.dumps(
                        {
                            "dataset_name": "live_code_bench",
                            "num": len(identities),
                            "score": score,
                        }
                    ),
                    encoding="utf-8",
                )
                (shard / "RANK_COMPLETE").touch()
                expected += len(identities)
                weighted += score * len(identities)

            summary = MERGE.merge_shards(root, shard_count=6, expected_samples=expected)
            self.assertEqual(summary["identity_unique"], expected)
            self.assertEqual(summary["identity_duplicates"], 0)
            self.assertAlmostEqual(summary["score"], weighted / expected)

    def test_runner_uses_all_six_gpus_for_disjoint_lcb_shards(self) -> None:
        text = (ROOT / "scripts" / "run_livecodebench_parallel6_vllm_20260717_v1.sh").read_text()
        self.assertIn("--dataset-subsets release_latest", text)
        self.assertIn("--dataset-shard-count 6", text)
        self.assertIn('for gpu in 0 1 2 3 4 5', text)
        self.assertIn("--local-code-sandbox", text)
        self.assertIn("--max-tokens 3072", text)


if __name__ == "__main__":
    unittest.main()
