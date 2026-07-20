from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .decontam import NORMALIZATION_VERSION, hash_problem


BENCHMARKS = {
    "math500": ("math500/test.jsonl", 500),
    "gsm8k": ("gsm8k/test.jsonl", 1319),
    "olympiadbench": ("olympiadbench/test.jsonl", 675),
    "amc23": ("amc23/test.jsonl", 40),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_contract(
    root: Path,
    source_revision: str,
    math500_revision: str = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
) -> tuple[list[dict], dict]:
    output: list[dict] = []
    sources: dict[str, dict] = {}
    for benchmark, (relative_path, expected_count) in BENCHMARKS.items():
        path = root / relative_path
        records = _read_jsonl(path)
        if len(records) != expected_count:
            raise ValueError(f"{benchmark} expected {expected_count} rows, found {len(records)}")
        for source_index, record in enumerate(records):
            problem = str(record.get("problem") or record.get("question") or record.get("prompt") or "")
            if not problem.strip():
                raise ValueError(f"{benchmark} row {source_index} has no problem text")
            output.append(
                {
                    "benchmark": benchmark,
                    "source_index": source_index,
                    "problem": problem,
                    "normalized_sha256": hash_problem(problem),
                }
            )
        sources[benchmark] = {
            "relative_path": relative_path,
            "row_count": len(records),
            "sha256": _sha256(path),
            "repository": (
                "https://huggingface.co/datasets/HuggingFaceH4/MATH-500"
                if benchmark == "math500"
                else "https://github.com/QwenLM/Qwen2.5-Math"
            ),
            "revision": math500_revision if benchmark == "math500" else source_revision,
        }
    unique_hashes = sorted({record["normalized_sha256"] for record in output})
    manifest = {
        "qwen_eval_source_revision": source_revision,
        "math500_source_revision": math500_revision,
        "normalization_version": NORMALIZATION_VERSION,
        "sources": sources,
        "problem_count": len(output),
        "unique_normalized_hash_count": len(unique_hashes),
    }
    return output, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-eval-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument(
        "--math500-revision",
        default="6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    records, manifest = build_contract(
        args.qwen_eval_root,
        args.source_revision,
        args.math500_revision,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    problems_path = args.out_dir / "benchmark_problems.jsonl"
    hashes_path = args.out_dir / "benchmark_problem_hashes.txt"
    with problems_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    hashes = sorted({record["normalized_sha256"] for record in records})
    hashes_path.write_text("".join(f"{value}\n" for value in hashes), encoding="ascii")
    manifest.update(
        {
            "benchmark_problems_sha256": _sha256(problems_path),
            "benchmark_hashes_sha256": _sha256(hashes_path),
        }
    )
    (args.out_dir / "benchmark_contract_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
