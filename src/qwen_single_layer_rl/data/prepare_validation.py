from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .decontam import BenchmarkProblemIndex, hash_problem
from .prep_numina import (
    iter_jsonl,
    normalize_numina_record,
    read_benchmark_problems,
    write_jsonl,
    write_parquet,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_validation(
    train_path: Path,
    candidate_paths: list[Path],
    benchmark_problems_path: Path,
    target_size: int,
) -> tuple[list[dict], dict]:
    train_hashes = {
        hash_problem(str(record.get("problem") or record.get("question") or record.get("prompt") or ""))
        for record in iter_jsonl(train_path)
    }
    benchmark_index = BenchmarkProblemIndex(read_benchmark_problems(benchmark_problems_path))
    selected: list[dict] = []
    selected_hashes: set[str] = set()
    selected_sources: list[dict] = []
    source_reports: list[dict] = []

    for candidate_path in candidate_paths:
        report = {
            "path": str(candidate_path.resolve()),
            "sha256": _sha256(candidate_path),
            "seen": 0,
            "selected": 0,
            "train_overlap_skipped": 0,
            "benchmark_exact_skipped": 0,
            "benchmark_near_skipped": 0,
            "duplicate_candidate_skipped": 0,
        }
        for source_index, raw in enumerate(iter_jsonl(candidate_path)):
            report["seen"] += 1
            record = normalize_numina_record(raw)
            problem_hash = hash_problem(record["problem"])
            if problem_hash in train_hashes:
                report["train_overlap_skipped"] += 1
                continue
            if problem_hash in selected_hashes:
                report["duplicate_candidate_skipped"] += 1
                continue
            match_kind = benchmark_index.match_kind(record["problem"])
            if match_kind == "exact":
                report["benchmark_exact_skipped"] += 1
                continue
            if match_kind == "near":
                report["benchmark_near_skipped"] += 1
                continue
            selected.append(record)
            selected_hashes.add(problem_hash)
            selected_sources.append(
                {
                    "candidate_path": str(candidate_path.resolve()),
                    "source_index": source_index,
                    "problem_sha256": problem_hash,
                }
            )
            report["selected"] += 1
            if len(selected) == target_size:
                break
        source_reports.append(report)
        if len(selected) == target_size:
            break

    if len(selected) != target_size:
        raise RuntimeError(f"Only found {len(selected)} valid validation rows; expected {target_size}")
    manifest = {
        "train_path": str(train_path.resolve()),
        "train_sha256": _sha256(train_path),
        "benchmark_problems_path": str(benchmark_problems_path.resolve()),
        "benchmark_problems_sha256": _sha256(benchmark_problems_path),
        "target_size": target_size,
        "written_count": len(selected),
        "source_reports": source_reports,
        "selected_sources": selected_sources,
    }
    return selected, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--candidate-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--benchmark-problems", type=Path, required=True)
    parser.add_argument("--target-size", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--write-parquet", action="store_true")
    args = parser.parse_args()

    selected, manifest = prepare_validation(
        args.train_jsonl,
        args.candidate_jsonl,
        args.benchmark_problems,
        args.target_size,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    validation_path = args.out_dir / "val.jsonl"
    write_jsonl(validation_path, selected)
    if args.write_parquet:
        write_parquet(args.out_dir / "val.parquet", selected)
    manifest["validation_sha256"] = _sha256(validation_path)
    (args.out_dir / "validation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
