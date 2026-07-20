from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .decontam import BenchmarkProblemIndex, hash_problem
from .prep_numina import iter_jsonl, read_benchmark_problems


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit(path: Path, benchmark_index: BenchmarkProblemIndex) -> tuple[dict, set[str]]:
    hashes: set[str] = set()
    row_count = 0
    exact_matches = 0
    near_matches = 0
    for record in iter_jsonl(path):
        row_count += 1
        problem = str(record.get("problem") or record.get("question") or record.get("prompt") or "")
        hashes.add(hash_problem(problem))
        match_kind = benchmark_index.match_kind(problem)
        exact_matches += int(match_kind == "exact")
        near_matches += int(match_kind == "near")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "row_count": row_count,
        "unique_problem_hashes": len(hashes),
        "duplicate_problem_rows": row_count - len(hashes),
        "benchmark_exact_matches": exact_matches,
        "benchmark_near_matches": near_matches,
    }, hashes


def verify(
    train_path: Path,
    val_path: Path,
    benchmark_problems_path: Path,
) -> dict:
    benchmark_index = BenchmarkProblemIndex(read_benchmark_problems(benchmark_problems_path))
    train, train_hashes = _audit(train_path, benchmark_index)
    val, val_hashes = _audit(val_path, benchmark_index)
    report = {
        "benchmark_problems_path": str(benchmark_problems_path.resolve()),
        "benchmark_problem_count": len(benchmark_index.problems),
        "train": train,
        "validation": val,
        "train_validation_exact_overlap": len(train_hashes & val_hashes),
    }
    report["passed"] = (
        train["benchmark_exact_matches"] == 0
        and train["benchmark_near_matches"] == 0
        and train["duplicate_problem_rows"] == 0
        and val["benchmark_exact_matches"] == 0
        and val["benchmark_near_matches"] == 0
        and val["duplicate_problem_rows"] == 0
        and report["train_validation_exact_overlap"] == 0
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--benchmark-problems", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = verify(args.train_jsonl, args.val_jsonl, args.benchmark_problems)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("Materialized data integrity verification failed")


if __name__ == "__main__":
    main()
