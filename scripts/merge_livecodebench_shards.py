from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from qwen_single_layer_rl.eval.live_code_bench_sharding import shard_for_identity


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"non-object JSONL row in {path}")
        rows.append(value)
    return rows


def _find_report(shard: Path) -> Path:
    reports = sorted(shard.glob("main/*/reports/*/live_code_bench.json"))
    if len(reports) != 1:
        raise ValueError(f"expected one LiveCodeBench report under {shard}, found {len(reports)}")
    return reports[0]


def merge_shards(root: Path, *, shard_count: int, expected_samples: int) -> dict[str, Any]:
    identities: list[str] = []
    shard_metrics: list[dict[str, Any]] = []
    for shard_index in range(shard_count):
        shard = root / "shards" / f"shard_{shard_index:02d}"
        if not (shard / "RANK_COMPLETE").exists():
            raise ValueError(f"shard {shard_index} is incomplete")
        receipts = _read_jsonl(shard / "dataset_shard_receipt.jsonl")
        report_path = _find_report(shard)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        count = int(report["num"])
        score = float(report["score"])
        if len(receipts) != count:
            raise ValueError(
                f"shard {shard_index} receipt/report count mismatch: {len(receipts)} != {count}"
            )
        for receipt in receipts:
            identity = str(receipt["identity_sha256"])
            if int(receipt["shard_index"]) != shard_index:
                raise ValueError(f"shard receipt index mismatch for {identity}")
            if int(receipt["shard_count"]) != shard_count:
                raise ValueError(f"shard receipt count mismatch for {identity}")
            if shard_for_identity(identity, shard_count) != shard_index:
                raise ValueError(f"identity assigned to the wrong shard: {identity}")
            identities.append(identity)
        shard_metrics.append(
            {
                "report": str(report_path.relative_to(root)),
                "samples": count,
                "score": score,
                "shard_index": shard_index,
            }
        )

    duplicates = sorted(identity for identity, count in Counter(identities).items() if count != 1)
    if duplicates:
        raise ValueError(f"duplicate shard identities: {duplicates[:5]}")
    if len(identities) != expected_samples:
        raise ValueError(f"expected {expected_samples} identities, found {len(identities)}")

    total = sum(item["samples"] for item in shard_metrics)
    score = sum(item["score"] * item["samples"] for item in shard_metrics) / total
    summary = {
        "benchmark": "live_code_bench",
        "dataset_subset": "release_latest",
        "expected_samples": expected_samples,
        "identity_duplicates": 0,
        "identity_unique": len(identities),
        "protocol": "vllm_greedy_pass_at_1_max_tokens_3072_6way_sha256_shard",
        "samples": total,
        "score": score,
        "shard_count": shard_count,
        "shards": shard_metrics,
        "status": "complete",
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "merge_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (root / "merged_identity_index.jsonl").open("w", encoding="utf-8") as handle:
        for identity in sorted(identities):
            handle.write(json.dumps({"identity_sha256": identity}, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--shard-count", type=int, default=6)
    parser.add_argument("--expected-samples", type=int, default=1055)
    args = parser.parse_args()
    summary = merge_shards(
        args.root.resolve(),
        shard_count=args.shard_count,
        expected_samples=args.expected_samples,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
