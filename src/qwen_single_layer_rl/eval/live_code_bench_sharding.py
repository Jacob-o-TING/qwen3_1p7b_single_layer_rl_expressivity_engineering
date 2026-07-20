from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_lcb_record_identity(record: dict[str, Any]) -> str:
    payload = {
        "contest_date": record.get("contest_date"),
        "contest_id": record.get("contest_id"),
        "platform": record.get("platform"),
        "question_content": record.get("question_content"),
        "question_id": record.get("question_id"),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def shard_for_identity(identity: str, shard_count: int) -> int:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    return int(identity[:16], 16) % shard_count


def install_evalscope_live_code_bench_sharding(
    *, shard_index: int, shard_count: int, receipt_path: Path
) -> None:
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")

    from evalscope.benchmarks.live_code_bench.live_code_bench_adapter import (
        LiveCodeBenchAdapter,
    )

    receipt_path = receipt_path.resolve()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text("", encoding="utf-8")
    requested = (shard_index, shard_count, str(receipt_path))
    installed = getattr(LiveCodeBenchAdapter, "_qwen_lcb_shard", None)
    if installed is not None:
        if installed != requested:
            raise RuntimeError(
                f"LiveCodeBench sharding already installed as {installed}, requested {requested}"
            )
        return

    original_record_to_sample = LiveCodeBenchAdapter.record_to_sample
    original_sample_filter = LiveCodeBenchAdapter.sample_filter

    def record_to_sample(instance: Any, record: dict[str, Any]) -> Any:
        identity = canonical_lcb_record_identity(record)
        sample = original_record_to_sample(instance, record)
        sample.metadata = dict(sample.metadata or {})
        sample.metadata["qwen_lcb_identity_sha256"] = identity
        sample.metadata["qwen_lcb_shard_index"] = shard_index
        sample.metadata["qwen_lcb_shard_count"] = shard_count
        return sample

    def sample_filter(instance: Any, sample: Any) -> bool:
        if not original_sample_filter(instance, sample):
            return False
        identity = sample.metadata.get("qwen_lcb_identity_sha256")
        if not isinstance(identity, str) or len(identity) != 64:
            raise RuntimeError("LiveCodeBench sample is missing its canonical identity")
        accepted = shard_for_identity(identity, shard_count) == shard_index
        if accepted:
            receipt = {
                "identity_sha256": identity,
                "shard_count": shard_count,
                "shard_index": shard_index,
            }
            with receipt_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(receipt, sort_keys=True) + "\n")
        return accepted

    LiveCodeBenchAdapter.record_to_sample = record_to_sample
    LiveCodeBenchAdapter.sample_filter = sample_filter
    LiveCodeBenchAdapter._qwen_lcb_shard = requested
