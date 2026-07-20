from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, Iterator

from qwen_single_layer_rl.rewards.math_reward import extract_answer

from .decontam import (
    BenchmarkProblemIndex,
    DecontamReport,
    filter_decontaminated,
    filter_exact_hashes,
    hash_problem,
)


def normalize_numina_record(record: dict) -> dict:
    problem = str(record.get("problem") or record.get("prompt") or record.get("question") or "")
    solution = str(record.get("solution") or record.get("answer") or "")
    messages = record.get("messages")
    return {
        "source": record.get("source", ""),
        "problem": problem,
        "solution": solution,
        "answer": extract_answer(solution),
        "messages": messages if isinstance(messages, list) else [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": solution},
        ],
    }


def sample_records(records: list[dict], target_size: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    if len(records) <= target_size:
        return list(records)
    indices = sorted(rng.sample(range(len(records)), target_size))
    return [records[i] for i in indices]


def reservoir_sample_decontam(
    records: Iterable[dict],
    target_size: int,
    seed: int,
    benchmark_hashes: set[str],
    benchmark_index: BenchmarkProblemIndex | None = None,
    max_source_records: int | None = None,
    progress_every: int = 50000,
) -> tuple[list[dict], DecontamReport]:
    rng = random.Random(seed)
    reservoir: list[dict] = []
    total = 0
    kept = 0
    removed = 0
    exact_removed = 0
    near_removed = 0
    duplicate_removed = 0
    seen_problem_hashes: set[str] = set()

    for raw in records:
        total += 1
        if max_source_records is not None and total > max_source_records:
            break
        record = normalize_numina_record(raw)
        problem_hash = hash_problem(record["problem"])
        if problem_hash in seen_problem_hashes:
            duplicate_removed += 1
            removed += 1
        else:
            seen_problem_hashes.add(problem_hash)
            match_kind = (
                benchmark_index.match_kind(record["problem"])
                if benchmark_index
                else "exact" if problem_hash in benchmark_hashes else None
            )
            if match_kind == "exact":
                exact_removed += 1
                removed += 1
            elif match_kind == "near":
                near_removed += 1
                removed += 1
            else:
                kept += 1
                if len(reservoir) < target_size:
                    reservoir.append(record)
                else:
                    replacement = rng.randrange(kept)
                    if replacement < target_size:
                        reservoir[replacement] = record
        if progress_every > 0 and total % progress_every == 0:
            print(
                f"prep_numina progress: seen={total} kept={kept} removed={removed} "
                f"sampled={len(reservoir)}",
                flush=True,
            )

    return reservoir, DecontamReport(
        input_count=total if max_source_records is None else min(total, max_source_records),
        kept_count=kept,
        removed_count=removed,
        benchmark_hash_count=len(benchmark_hashes),
        exact_removed_count=exact_removed,
        near_removed_count=near_removed,
        benchmark_problem_count=len(benchmark_index.problems) if benchmark_index else 0,
        source_duplicate_removed_count=duplicate_removed,
    )


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def iter_hf_dataset(
    dataset_name: str,
    split: str,
    streaming: bool,
    cache_dir: Path | None,
) -> Iterable[dict]:
    try:
        from datasets import load_dataset  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency `datasets`. Install requirements.txt on the GPU/AutoDL machine first."
        ) from exc

    kwargs = {"split": split, "streaming": streaming}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    dataset = load_dataset(dataset_name, **kwargs)
    return dataset


def iter_modelscope_dataset(
    dataset_name: str,
    split: str,
    streaming: bool,
    cache_dir: Path | None,
) -> Iterable[dict]:
    try:
        from modelscope import MsDataset  # type: ignore
    except (ModuleNotFoundError, RuntimeError) as exc:
        raise SystemExit(
            "Missing ModelScope dataset dependencies. Use the isolated EvalScope environment."
        ) from exc

    kwargs = {
        "split": split,
        "use_streaming": streaming,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    return MsDataset.load(dataset_name, **kwargs)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def write_parquet(path: Path, records: list[dict]) -> None:
    try:
        import pandas as pd  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency `pandas`/`pyarrow` for parquet output. Install requirements.txt first."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in records:
        row = dict(record)
        row["messages_json"] = json.dumps(row.pop("messages", []), ensure_ascii=True)
        rows.append(row)
    pd.DataFrame(rows).to_parquet(path, index=False)


def read_hashes(path: Path | None) -> set[str]:
    if path and path.exists():
        return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    return set()


def read_benchmark_problems(path: Path | None) -> list[str]:
    if not path or not path.exists():
        return []
    problems: list[str] = []
    for record in iter_jsonl(path):
        problem = str(record.get("problem") or record.get("question") or record.get("prompt") or "")
        if problem.strip():
            problems.append(problem)
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-jsonl", type=Path)
    source.add_argument("--hf-dataset", default=None)
    source.add_argument("--modelscope-dataset", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-source-records", type=int)
    parser.add_argument("--progress-every", type=int, default=50000)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--benchmark-hashes", type=Path)
    parser.add_argument("--benchmark-problems", type=Path)
    parser.add_argument("--ngram-size", type=int, default=8)
    parser.add_argument("--ngram-jaccard-threshold", type=float, default=0.50)
    parser.add_argument("--ngram-containment-threshold", type=float, default=0.80)
    parser.add_argument("--write-parquet", action="store_true")
    parser.add_argument("--val-input-jsonl", type=Path)
    parser.add_argument("--hf-val-split", default="test")
    parser.add_argument("--val-size", type=int, default=100)
    args = parser.parse_args()

    hashes = read_hashes(args.benchmark_hashes)
    benchmark_problems = read_benchmark_problems(args.benchmark_problems)
    benchmark_index = None
    if benchmark_problems:
        benchmark_index = BenchmarkProblemIndex(
            benchmark_problems,
            ngram_size=args.ngram_size,
            jaccard_threshold=args.ngram_jaccard_threshold,
            containment_threshold=args.ngram_containment_threshold,
        )
        hashes.update(benchmark_index.hashes)
    if args.input_jsonl:
        records = [normalize_numina_record(record) for record in load_jsonl(args.input_jsonl)]
        if benchmark_index:
            filtered, report = filter_decontaminated(records, benchmark_index)
        else:
            filtered, report = filter_exact_hashes(records, hashes)
        sampled = sample_records(filtered, args.target_size, args.seed)
        source_description = str(args.input_jsonl)
    else:
        if args.modelscope_dataset:
            source_records = iter_modelscope_dataset(
                args.modelscope_dataset, args.split, args.streaming, args.cache_dir
            )
            source_name = args.modelscope_dataset
            source_hub = "modelscope"
        else:
            source_records = iter_hf_dataset(args.hf_dataset, args.split, args.streaming, args.cache_dir)
            source_name = args.hf_dataset
            source_hub = "huggingface"
        sampled, report = reservoir_sample_decontam(
            source_records,
            target_size=args.target_size,
            seed=args.seed,
            benchmark_hashes=hashes,
            benchmark_index=benchmark_index,
            max_source_records=args.max_source_records,
            progress_every=args.progress_every,
        )
        source_description = f"{source_hub}:{source_name}:{args.split}"

    write_jsonl(args.out_dir / "train.jsonl", sampled)
    if args.write_parquet:
        write_parquet(args.out_dir / "train.parquet", sampled)

    val_records: list[dict] = []
    if args.val_input_jsonl:
        val_records = [normalize_numina_record(record) for record in load_jsonl(args.val_input_jsonl)]
        val_records = val_records[: args.val_size]
    elif (args.hf_dataset or args.modelscope_dataset) and args.hf_val_split and args.val_size > 0:
        try:
            if args.modelscope_dataset:
                val_iter = iter_modelscope_dataset(
                    args.modelscope_dataset, args.hf_val_split, args.streaming, args.cache_dir
                )
            else:
                val_iter = iter_hf_dataset(
                    args.hf_dataset, args.hf_val_split, args.streaming, args.cache_dir
                )
            val_records = [normalize_numina_record(record) for _, record in zip(range(args.val_size), val_iter)]
        except Exception as exc:  # Keep train prep usable if a dataset has no validation split.
            print(f"Skipping validation split {args.hf_val_split!r}: {exc}")

    if val_records:
        write_jsonl(args.out_dir / "val.jsonl", val_records)
        if args.write_parquet:
            write_parquet(args.out_dir / "val.parquet", val_records)

    (args.out_dir / "prep_manifest.json").write_text(
        json.dumps(
            {
                "source": source_description,
                "split": args.split,
                "streaming": args.streaming,
                "target_size": args.target_size,
                "seed": args.seed,
                "decontam_report": report.__dict__,
                "decontam_policy": {
                    "benchmark_problems": str(args.benchmark_problems) if args.benchmark_problems else None,
                    "benchmark_hashes": str(args.benchmark_hashes) if args.benchmark_hashes else None,
                    "ngram_size": args.ngram_size,
                    "ngram_jaccard_threshold": args.ngram_jaccard_threshold,
                    "ngram_containment_threshold": args.ngram_containment_threshold,
                },
                "written_count": len(sampled),
                "val_written_count": len(val_records),
                "write_parquet": args.write_parquet,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
