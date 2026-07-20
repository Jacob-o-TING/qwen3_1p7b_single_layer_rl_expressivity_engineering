from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from qwen_single_layer_rl.eval.generic_evalscope_sharding import shard_for_identity


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row in {path}")
            rows.append(value)
    return rows


def _find_report(shard: Path, dataset: str) -> tuple[Path, dict[str, Any]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(shard.glob("main/*/reports/*/*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if report.get("dataset_name") == dataset:
            matches.append((path, report))
    if len(matches) != 1:
        raise ValueError(
            f"expected one {dataset} report under {shard}, found {len(matches)}"
        )
    return matches[0]


def merge_shards(
    root: Path,
    *,
    dataset: str,
    shard_count: int,
    expected_identities: int,
) -> dict[str, Any]:
    identities: list[str] = []
    shard_metrics: list[dict[str, Any]] = []
    for shard_index in range(shard_count):
        shard = root / "shards" / f"shard_{shard_index:02d}"
        if not (shard / "RANK_COMPLETE").exists():
            raise ValueError(f"shard {shard_index} is incomplete")
        receipts = _read_jsonl(shard / "dataset_shard_receipt.jsonl")
        report_path, report = _find_report(shard, dataset)
        for receipt in receipts:
            identity = str(receipt["identity_sha256"])
            payload_digest = str(receipt.get("formatted_sample_sha256", ""))
            if len(payload_digest) != 64:
                raise ValueError(f"missing formatted payload digest for {identity}")
            if receipt.get("dataset") != dataset:
                raise ValueError(f"dataset mismatch for {identity}")
            if int(receipt["shard_index"]) != shard_index:
                raise ValueError(f"shard index mismatch for {identity}")
            if int(receipt["shard_count"]) != shard_count:
                raise ValueError(f"shard count mismatch for {identity}")
            if shard_for_identity(identity, shard_count) != shard_index:
                raise ValueError(f"identity assigned to the wrong shard: {identity}")
            identities.append(identity)
        shard_metrics.append(
            {
                "identity_count": len(receipts),
                "report": str(report_path.relative_to(root)),
                "report_samples": int(report["num"]),
                "score": float(report["score"]),
                "shard_index": shard_index,
            }
        )

    duplicates = sorted(
        identity for identity, count in Counter(identities).items() if count != 1
    )
    if duplicates:
        raise ValueError(f"duplicate shard identities: {duplicates[:5]}")
    if len(identities) != expected_identities:
        raise ValueError(
            f"expected {expected_identities} identities, found {len(identities)}"
        )
    report_samples = sum(item["report_samples"] for item in shard_metrics)
    score = (
        sum(item["score"] * item["report_samples"] for item in shard_metrics)
        / report_samples
    )
    summary = {
        "dataset": dataset,
        "expected_identities": expected_identities,
        "identity_duplicates": 0,
        "identity_unique": len(identities),
        "protocol": "vllm_greedy_pass_at_1_max_tokens_3072_6way_coordinate_shard_payload_digest",
        "report_samples": report_samples,
        "score": score,
        "shard_count": shard_count,
        "shards": shard_metrics,
        "status": "complete",
    }
    (root / "merge_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (root / "merged_identity_index.jsonl").open("w", encoding="utf-8") as handle:
        for identity in sorted(identities):
            handle.write(json.dumps({"identity_sha256": identity}) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--shard-count", type=int, default=6)
    parser.add_argument("--expected-identities", type=int, required=True)
    args = parser.parse_args()
    summary = merge_shards(
        args.root.resolve(),
        dataset=args.dataset,
        shard_count=args.shard_count,
        expected_identities=args.expected_identities,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
