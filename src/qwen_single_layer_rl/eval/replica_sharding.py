from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True, order=True)
class RequestIdentity:
    benchmark: str
    item_id: int
    sample_id: int

    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "RequestIdentity":
        return cls(
            benchmark=str(value["benchmark"]),
            item_id=int(value["item_id"]),
            sample_id=int(value["sample_id"]),
        )


def owner_rank(identity: RequestIdentity, replicas: int) -> int:
    if replicas <= 0:
        raise ValueError("replicas must be positive")
    digest = hashlib.sha256(identity.key().encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % replicas


def shard_identities(
    identities: Iterable[RequestIdentity], replicas: int
) -> dict[int, list[RequestIdentity]]:
    shards = {rank: [] for rank in range(replicas)}
    seen: set[RequestIdentity] = set()
    for identity in sorted(identities):
        if identity in seen:
            raise ValueError(f"Duplicate expected identity: {identity.key()}")
        seen.add(identity)
        shards[owner_rank(identity, replicas)].append(identity)
    return shards


def topology_preflight(
    *,
    visible_gpu_ids: list[str],
    requested_replicas: int,
    tensor_parallel_size: int,
    identities: Iterable[RequestIdentity],
    output_root: Path,
) -> dict[str, Any]:
    if requested_replicas <= 0:
        raise ValueError("requested_replicas must be positive")
    if tensor_parallel_size <= 0:
        raise ValueError("tensor_parallel_size must be positive")
    required_gpus = requested_replicas * tensor_parallel_size
    if required_gpus > len(visible_gpu_ids):
        raise ValueError(
            f"Topology requires {required_gpus} visible GPUs but found {len(visible_gpu_ids)}"
        )
    identity_list = list(identities)
    shards = shard_identities(identity_list, requested_replicas)
    return {
        "visible_gpu_ids": visible_gpu_ids,
        "requested_replicas": requested_replicas,
        "tensor_parallel_size": tensor_parallel_size,
        "required_gpus": required_gpus,
        "global_expected_count": len(identity_list),
        "shard_counts": {str(rank): len(rows) for rank, rows in shards.items()},
        "output_ownership": {
            str(rank): str(output_root / f"rank_{rank:05d}") for rank in range(requested_replicas)
        },
    }


def write_rank_receipt(
    *,
    path: Path,
    rank: int,
    replicas: int,
    expected: list[RequestIdentity],
    completed: list[RequestIdentity],
) -> None:
    if any(owner_rank(identity, replicas) != rank for identity in completed):
        raise ValueError(f"Rank {rank} receipt contains an identity owned by another rank")
    completed_counts = Counter(completed)
    duplicates = sorted(identity.key() for identity, count in completed_counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"Rank {rank} receipt contains duplicate identities: {duplicates}")
    payload = {
        "rank": rank,
        "replicas": replicas,
        "expected_count": len(expected),
        "completed_count": len(completed),
        "missing": [asdict(identity) for identity in sorted(set(expected) - set(completed))],
        "status": "complete" if set(expected) == set(completed) else "partial",
    }
    _atomic_json(path, payload)


def _row_identity(row: dict[str, Any]) -> RequestIdentity:
    identity = row.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("Every replica row must contain an identity object")
    return RequestIdentity.from_mapping(identity)


def _canonical_row(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def merge_rank_jsonl(
    *,
    rank_paths: dict[int, Path],
    merged_path: Path,
    expected: Iterable[RequestIdentity],
    replicas: int,
) -> dict[str, Any]:
    expected_set = set(expected)
    merged: dict[RequestIdentity, dict[str, Any]] = {}
    for row in _read_jsonl(merged_path):
        identity = _row_identity(row)
        if identity in merged:
            raise ValueError(f"Existing merge contains duplicate identity: {identity.key()}")
        merged[identity] = row

    idempotent_skips = 0
    newly_merged = 0
    for rank, path in sorted(rank_paths.items()):
        rank_seen: set[RequestIdentity] = set()
        for row in _read_jsonl(path):
            identity = _row_identity(row)
            if identity in rank_seen:
                raise ValueError(f"Rank {rank} output contains duplicate identity: {identity.key()}")
            rank_seen.add(identity)
            actual_owner = owner_rank(identity, replicas)
            if actual_owner != rank:
                raise ValueError(
                    f"Wrong output owner for {identity.key()}: rank={rank}, expected={actual_owner}"
                )
            if identity not in expected_set:
                raise ValueError(f"Unexpected identity in rank {rank} output: {identity.key()}")
            if identity in merged:
                if _canonical_row(merged[identity]) != _canonical_row(row):
                    raise ValueError(f"Conflicting row for identity: {identity.key()}")
                idempotent_skips += 1
                continue
            merged[identity] = row
            newly_merged += 1

    missing = sorted(expected_set - set(merged))
    unexpected = sorted(set(merged) - expected_set)
    ordered_rows = [merged[identity] for identity in sorted(merged)]
    _atomic_jsonl(merged_path, ordered_rows)
    return {
        "expected_count": len(expected_set),
        "merged_count": len(merged),
        "newly_merged": newly_merged,
        "idempotent_skips": idempotent_skips,
        "missing": [asdict(identity) for identity in missing],
        "unexpected": [asdict(identity) for identity in unexpected],
        "status": "complete" if not missing and not unexpected else "partial",
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
