from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from qwen_single_layer_rl.data.decontam import hash_problem


DEFAULT_SYSTEM_SUFFIX = "Let's think step by step and output the final answer within \\boxed{}."


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def problem_order_sha256(problems: list[str]) -> str:
    digest = hashlib.sha256()
    for problem in problems:
        digest.update(problem.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    try:
        import pandas as pd  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing pandas/pyarrow; install requirements before materializing veRL data.") from exc

    return pd.read_parquet(path).to_dict(orient="records")


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pandas as pd  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing pandas/pyarrow; install requirements before materializing veRL data.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def to_verl_row(row: dict[str, Any], index: int, source_name: str, prompt_suffix: str) -> dict[str, Any]:
    problem = str(row.get("problem") or row.get("prompt") or row.get("question") or "")
    answer = str(row.get("answer") or "")
    content = problem.strip()
    if prompt_suffix:
        content = f"{content}\n\n{prompt_suffix}"

    return {
        "data_source": source_name,
        "prompt": [{"role": "user", "content": content}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "index": index,
            "source": str(row.get("source") or source_name),
            "problem_sha256": hash_problem(problem),
        },
    }


def materialize_verl_parquet(
    input_path: Path,
    output_path: Path,
    *,
    source_name: str = "numina_math_cot",
    prompt_suffix: str = DEFAULT_SYSTEM_SUFFIX,
) -> dict[str, Any]:
    rows = _read_rows(input_path)
    problems = [str(row.get("problem") or row.get("prompt") or row.get("question") or "") for row in rows]
    verl_rows = [to_verl_row(row, index, source_name, prompt_suffix) for index, row in enumerate(rows)]
    _write_parquet(output_path, verl_rows)
    return {
        "input_path": str(input_path),
        "input_sha256": file_sha256(input_path),
        "output_path": str(output_path),
        "output_sha256": file_sha256(output_path),
        "row_count": len(rows),
        "problem_order_sha256": problem_order_sha256(problems),
        "source_name": source_name,
        "prompt_suffix": prompt_suffix,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-name", default="numina_math_cot")
    parser.add_argument("--prompt-suffix", default=DEFAULT_SYSTEM_SUFFIX)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "train": materialize_verl_parquet(
            args.train,
            args.out_dir / "train.parquet",
            source_name=args.source_name,
            prompt_suffix=args.prompt_suffix,
        ),
        "val": materialize_verl_parquet(
            args.val,
            args.out_dir / "val.parquet",
            source_name=args.source_name,
            prompt_suffix=args.prompt_suffix,
        ),
        "format": "verl_rlhf_parquet",
        "order_contract": "Rows preserve canonical input order; veRL sampler shuffle is controlled separately by data.seed.",
    }
    (args.out_dir / "verl_data_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
