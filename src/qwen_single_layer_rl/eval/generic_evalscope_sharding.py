from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_sample_identity(
    dataset_name: str,
    subset_name: str,
    sample_index: int,
    sample: Any,
) -> str:
    # Ownership must not depend on rank-local formatting RNG. Keep the
    # formatted payload as a separate integrity digest in the receipt.
    del sample
    payload = {
        "dataset": dataset_name,
        "sample_index": sample_index,
        "subset": subset_name,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def formatted_sample_sha256(sample: Any) -> str:
    if hasattr(sample, "model_dump"):
        sample_payload = sample.model_dump(mode="json")
    else:
        sample_payload = sample
    encoded = json.dumps(
        sample_payload,
        default=str,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def shard_for_identity(identity: str, shard_count: int) -> int:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    return int(identity[:16], 16) % shard_count


def install_evalscope_generic_sharding(
    *,
    dataset_name: str,
    shard_index: int,
    shard_count: int,
    receipt_path: Path,
) -> None:
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")

    from evalscope.api.benchmark.adapters.default_data_adapter import DefaultDataAdapter
    from evalscope.api.dataset import DatasetDict, MemoryDataset

    receipt_path = receipt_path.resolve()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    requested = (dataset_name, shard_index, shard_count, str(receipt_path))
    installed = getattr(DefaultDataAdapter, "_qwen_generic_shard", None)
    if installed is not None:
        if installed != requested:
            raise RuntimeError(
                f"generic EvalScope sharding already installed as {installed}, "
                f"requested {requested}"
            )
        return

    original_load_dataset = DefaultDataAdapter.load_dataset

    def load_dataset(instance: Any) -> Any:
        full_dataset = original_load_dataset(instance)
        sharded_subsets: dict[str, Any] = {}
        receipts: list[dict[str, Any]] = []
        for subset_name, subset in full_dataset.items():
            accepted = []
            for sample_index, sample in enumerate(subset):
                identity = canonical_sample_identity(
                    dataset_name,
                    str(subset_name),
                    sample_index,
                    sample,
                )
                owner = shard_for_identity(identity, shard_count)
                if owner != shard_index:
                    continue
                accepted.append(sample)
                receipts.append(
                    {
                        "dataset": dataset_name,
                        "formatted_sample_sha256": formatted_sample_sha256(sample),
                        "identity_sha256": identity,
                        "sample_index": sample_index,
                        "shard_count": shard_count,
                        "shard_index": shard_index,
                        "subset": str(subset_name),
                    }
                )
            sharded_subsets[str(subset_name)] = MemoryDataset(
                samples=accepted,
                name=subset.name,
                location=subset.location,
                shuffled=subset.shuffled,
            )
        sharded = DatasetDict(sharded_subsets)
        instance.test_dataset = sharded
        with receipt_path.open("w", encoding="utf-8") as handle:
            for receipt in receipts:
                handle.write(json.dumps(receipt, sort_keys=True) + "\n")
        return sharded

    DefaultDataAdapter.load_dataset = load_dataset
    DefaultDataAdapter._qwen_generic_shard = requested
