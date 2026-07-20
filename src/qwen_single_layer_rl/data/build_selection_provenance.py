from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .decontam import hash_problem
from .prep_numina import (
    iter_hf_dataset,
    iter_jsonl,
    normalize_numina_record,
)


LEDGER_COLUMNS = (
    "role",
    "materialized_index",
    "source_split",
    "source_index",
    "problem_sha256",
    "normalized_record_sha256",
    "source_category",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_record_sha256(record: dict[str, Any]) -> str:
    payload = json.dumps(
        record,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_selection_targets(
    role: str,
    materialized_path: Path,
    source_splits: list[str],
) -> list[dict[str, Any]]:
    selected = list(iter_jsonl(materialized_path))
    if len(selected) != len(source_splits):
        raise ValueError(
            f"Source split assignments for {role} had {len(source_splits)} rows; "
            f"materialized data had {len(selected)}"
        )
    targets: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for index, (record, source_split) in enumerate(zip(selected, source_splits, strict=True)):
        problem_hash = hash_problem(str(record.get("problem") or ""))
        if problem_hash in seen_hashes:
            raise ValueError(f"Duplicate materialized problem hash for {role}: {problem_hash}")
        seen_hashes.add(problem_hash)
        targets.append(
            {
                "role": role,
                "materialized_index": index,
                "source_split": source_split,
                "problem_sha256": problem_hash,
                "expected_record_sha256": canonical_record_sha256(record),
            }
        )
    return targets


def trace_source_split(
    *,
    source_split: str,
    targets: list[dict[str, Any]],
    source_records: Iterable[dict[str, Any]],
    expected_source_rows: int | None = None,
    progress_every: int = 50_000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_by_hash: dict[str, dict[str, Any]] = {}
    for target in targets:
        problem_hash = str(target["problem_sha256"])
        if problem_hash in target_by_hash:
            raise ValueError(f"Multiple selected rows map to {source_split} hash {problem_hash}")
        target_by_hash[problem_hash] = target

    matched: dict[str, dict[str, Any]] = {}
    source_problem_order = hashlib.sha256()
    source_record_order = hashlib.sha256()
    source_count = 0
    for source_index, raw in enumerate(source_records):
        source_count += 1
        normalized = normalize_numina_record(raw)
        problem_hash = hash_problem(normalized["problem"])
        record_hash = canonical_record_sha256(normalized)
        source_problem_order.update(bytes.fromhex(problem_hash))
        source_record_order.update(bytes.fromhex(record_hash))
        if problem_hash in target_by_hash and problem_hash not in matched:
            target = target_by_hash[problem_hash]
            if record_hash != target["expected_record_sha256"]:
                raise ValueError(
                    f"Normalized source record differs from materialized {target['role']} row "
                    f"{target['materialized_index']} at {source_split}:{source_index}"
                )
            matched[problem_hash] = {
                "role": target["role"],
                "materialized_index": target["materialized_index"],
                "source_split": source_split,
                "source_index": source_index,
                "problem_sha256": problem_hash,
                "normalized_record_sha256": record_hash,
                "source_category": str(normalized.get("source") or ""),
            }
        if progress_every > 0 and source_count % progress_every == 0:
            print(
                f"provenance progress: split={source_split} seen={source_count} "
                f"matched={len(matched)}/{len(target_by_hash)}",
                flush=True,
            )

    if expected_source_rows is not None and source_count != expected_source_rows:
        raise ValueError(
            f"Source split {source_split} had {source_count} rows; expected {expected_source_rows}"
        )
    missing = sorted(set(target_by_hash) - set(matched))
    if missing:
        raise ValueError(
            f"Could not trace {len(missing)} rows in {source_split}; first missing hash: {missing[0]}"
        )
    stats = {
        "source_split": source_split,
        "source_row_count": source_count,
        "target_count": len(target_by_hash),
        "matched_count": len(matched),
        "source_problem_order_sha256": source_problem_order.hexdigest(),
        "source_normalized_record_order_sha256": source_record_order.hexdigest(),
        "source_index_base": 0,
        "duplicate_resolution": "first normalized-problem occurrence",
        "all_normalized_records_match": True,
    }
    return list(matched.values()), stats


def trace_selection(
    *,
    role: str,
    materialized_path: Path,
    source_split: str,
    source_records: Iterable[dict[str, Any]],
    expected_source_rows: int | None = None,
    progress_every: int = 50_000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = list(iter_jsonl(materialized_path))
    targets = load_selection_targets(role, materialized_path, [source_split] * len(selected))
    ledger_rows, source_stats = trace_source_split(
        source_split=source_split,
        targets=targets,
        source_records=source_records,
        expected_source_rows=expected_source_rows,
        progress_every=progress_every,
    )
    ledger_rows.sort(key=lambda row: int(row["materialized_index"]))
    stats = {
        "role": role,
        "materialized_path": str(materialized_path.resolve()),
        "materialized_sha256": file_sha256(materialized_path),
        "materialized_count": len(selected),
        "matched_count": len(ledger_rows),
        **source_stats,
    }
    return ledger_rows, stats


def write_ledger(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=LEDGER_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _load_modelscope_splits(
    dataset_id: str,
    cache_dir: Path | None,
) -> Any:
    try:
        from modelscope import MsDataset  # type: ignore
    except (ModuleNotFoundError, RuntimeError) as exc:
        raise SystemExit(
            "Missing ModelScope dataset dependencies. Use the isolated EvalScope environment."
        ) from exc

    kwargs: dict[str, Any] = {"use_streaming": True}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    # One load returns both splits. Repeated MsDataset.load calls in ModelScope
    # 1.33 can recursively re-patch HfFileSystem in the same process.
    return MsDataset.load(dataset_id, **kwargs)


def validation_source_splits(
    *,
    validation_count: int,
    default_split: str,
    selection_manifest: Path | None,
    candidate_mappings: list[str],
) -> tuple[list[str], dict[str, str]]:
    if selection_manifest is None:
        return [default_split] * validation_count, {}
    mapping: dict[str, str] = {}
    for item in candidate_mappings:
        try:
            candidate_path, split = item.rsplit("=", 1)
        except ValueError as exc:
            raise ValueError(f"Invalid candidate mapping {item!r}; expected PATH=SPLIT") from exc
        mapping[candidate_path.replace("\\", "/")] = split
    manifest = json.loads(selection_manifest.read_text(encoding="utf-8"))
    sources = manifest.get("selected_sources", [])
    if len(sources) != validation_count:
        raise ValueError(
            f"Validation selection manifest had {len(sources)} sources; expected {validation_count}"
        )
    assignments: list[str] = []
    for selected in sources:
        candidate = str(selected["candidate_path"]).replace("\\", "/")
        matches = [split for suffix, split in mapping.items() if candidate.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(
                f"Candidate {candidate!r} matched {len(matches)} source-split mappings; expected one"
            )
        assignments.append(matches[0])
    return assignments, mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--validation-jsonl", type=Path, required=True)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--source-hub", choices=("modelscope", "huggingface"), default="modelscope")
    parser.add_argument("--train-source-split", default="train")
    parser.add_argument("--validation-source-split", default="test")
    parser.add_argument("--validation-selection-manifest", type=Path)
    parser.add_argument(
        "--candidate-source-split",
        action="append",
        default=[],
        metavar="PATH_SUFFIX=SPLIT",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=50_000)
    args = parser.parse_args()

    contract = json.loads(args.source_contract.read_text(encoding="utf-8"))
    dataset_id = str(contract["dataset_id"])
    expected_rows = contract.get("expected_split_rows", {})
    train_count = sum(1 for _ in iter_jsonl(args.train_jsonl))
    validation_count = sum(1 for _ in iter_jsonl(args.validation_jsonl))
    validation_splits, candidate_mapping = validation_source_splits(
        validation_count=validation_count,
        default_split=args.validation_source_split,
        selection_manifest=args.validation_selection_manifest,
        candidate_mappings=args.candidate_source_split,
    )
    all_targets = load_selection_targets(
        "train", args.train_jsonl, [args.train_source_split] * train_count
    ) + load_selection_targets("validation", args.validation_jsonl, validation_splits)
    modelscope_splits = (
        _load_modelscope_splits(dataset_id, args.cache_dir)
        if args.source_hub == "modelscope"
        else None
    )
    source_stats: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for source_split in sorted({str(target["source_split"]) for target in all_targets}):
        split_targets = [
            target for target in all_targets if target["source_split"] == source_split
        ]
        rows, stats = trace_source_split(
            source_split=source_split,
            targets=split_targets,
            source_records=(
                modelscope_splits[source_split]
                if modelscope_splits is not None
                else iter_hf_dataset(dataset_id, source_split, True, args.cache_dir)
            ),
            expected_source_rows=expected_rows.get(source_split),
            progress_every=args.progress_every,
        )
        all_rows.extend(rows)
        source_stats[source_split] = stats

    role_order = {"train": 0, "validation": 1}
    all_rows.sort(
        key=lambda row: (role_order[str(row["role"])], int(row["materialized_index"]))
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = args.out_dir / "selected_rows.tsv"
    write_ledger(ledger_path, all_rows)
    manifest = {
        "schema_version": 1,
        "source_contract": contract,
        "source_contract_path": str(args.source_contract.resolve()),
        "source_contract_sha256": file_sha256(args.source_contract),
        "source_hub_used": args.source_hub,
        "selection_semantics": (
            "Each row maps a materialized output index to the zero-based global index in its "
            "online split. Source deduplication retained the first normalized-problem occurrence."
        ),
        "validation_selection_manifest": (
            str(args.validation_selection_manifest.resolve())
            if args.validation_selection_manifest
            else None
        ),
        "candidate_source_split_mapping": candidate_mapping,
        "materialized": {
            "train": {
                "path": str(args.train_jsonl.resolve()),
                "sha256": file_sha256(args.train_jsonl),
                "row_count": train_count,
            },
            "validation": {
                "path": str(args.validation_jsonl.resolve()),
                "sha256": file_sha256(args.validation_jsonl),
                "row_count": validation_count,
            },
        },
        "source_splits": source_stats,
        "ledger": {
            "path": ledger_path.name,
            "sha256": file_sha256(ledger_path),
            "row_count": len(all_rows),
            "columns": list(LEDGER_COLUMNS),
        },
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
