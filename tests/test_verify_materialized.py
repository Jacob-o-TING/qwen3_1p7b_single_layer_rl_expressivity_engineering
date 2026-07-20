from __future__ import annotations

import json
from pathlib import Path

from qwen_single_layer_rl.data.verify_materialized import verify


def _write_jsonl(path: Path, problems: list[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for problem in problems:
            handle.write(json.dumps({"problem": problem}) + "\n")


def test_verify_requires_unique_disjoint_materialized_splits(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmarks.jsonl"
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    _write_jsonl(benchmark_path, ["A benchmark-only problem"])

    _write_jsonl(train_path, ["Train problem one", "Train problem two"])
    _write_jsonl(val_path, ["Validation problem"])
    clean = verify(train_path, val_path, benchmark_path)
    assert clean["passed"] is True

    _write_jsonl(train_path, ["Train problem one", "Train problem one"])
    duplicate = verify(train_path, val_path, benchmark_path)
    assert duplicate["train"]["duplicate_problem_rows"] == 1
    assert duplicate["passed"] is False

    _write_jsonl(train_path, ["Shared problem"])
    _write_jsonl(val_path, ["Shared problem"])
    overlap = verify(train_path, val_path, benchmark_path)
    assert overlap["train_validation_exact_overlap"] == 1
    assert overlap["passed"] is False
