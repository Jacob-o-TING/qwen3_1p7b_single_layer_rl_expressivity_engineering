from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from qwen_single_layer_rl.eval.replica_sharding import (
    RequestIdentity,
    merge_rank_jsonl,
    owner_rank,
    shard_identities,
    topology_preflight,
)


class EvalReplicaShardingTests(unittest.TestCase):
    def test_sharding_is_stable_and_independent_of_input_order(self) -> None:
        identities = [RequestIdentity("amc", item, sample) for item in range(5) for sample in range(4)]
        forward = shard_identities(identities, 7)
        reverse = shard_identities(reversed(identities), 7)
        self.assertEqual(forward, reverse)
        for rank, rows in forward.items():
            self.assertTrue(all(owner_rank(identity, 7) == rank for identity in rows))

    def test_topology_preflight_has_no_fixed_replica_limit(self) -> None:
        identities = [RequestIdentity("math", item, 0) for item in range(19)]
        result = topology_preflight(
            visible_gpu_ids=[str(index) for index in range(9)],
            requested_replicas=9,
            tensor_parallel_size=1,
            identities=identities,
            output_root=Path("outputs"),
        )
        self.assertEqual(result["requested_replicas"], 9)
        self.assertEqual(sum(result["shard_counts"].values()), 19)

    def test_merge_is_incremental_and_idempotent(self) -> None:
        identities = [RequestIdentity("gsm8k", item, 0) for item in range(8)]
        shards = shard_identities(identities, 3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for rank, rows in shards.items():
                path = root / f"rank_{rank}.jsonl"
                path.write_text(
                    "".join(
                        json.dumps({"identity": asdict(identity), "value": identity.item_id}) + "\n"
                        for identity in reversed(rows)
                    ),
                    encoding="utf-8",
                )
                paths[rank] = path
            merged = root / "merged.jsonl"
            first = merge_rank_jsonl(
                rank_paths=paths, merged_path=merged, expected=identities, replicas=3
            )
            second = merge_rank_jsonl(
                rank_paths=paths, merged_path=merged, expected=identities, replicas=3
            )
        self.assertEqual(first["status"], "complete")
        self.assertEqual(first["newly_merged"], 8)
        self.assertEqual(second["newly_merged"], 0)
        self.assertEqual(second["idempotent_skips"], 8)

    def test_wrong_owner_and_duplicate_rows_are_fatal(self) -> None:
        identity = RequestIdentity("math", 1, 0)
        owner = owner_rank(identity, 2)
        wrong = 1 - owner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wrong_path = root / "wrong.jsonl"
            row = json.dumps({"identity": asdict(identity)}) + "\n"
            wrong_path.write_text(row, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Wrong output owner"):
                merge_rank_jsonl(
                    rank_paths={wrong: wrong_path},
                    merged_path=root / "merged.jsonl",
                    expected=[identity],
                    replicas=2,
                )
            duplicate_path = root / "duplicate.jsonl"
            duplicate_path.write_text(row + row, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate identity"):
                merge_rank_jsonl(
                    rank_paths={owner: duplicate_path},
                    merged_path=root / "merged2.jsonl",
                    expected=[identity],
                    replicas=2,
                )


if __name__ == "__main__":
    unittest.main()
