from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from qwen_single_layer_rl.eval.local_code_sandbox import PrivilegeDroppedLocalSandboxBackend


PROMPT_TEMPLATE = (
    "### Question:\n{question_content}\n\n{format_prompt} "
    "### Answer: (use the provided format with backticks)\n\n"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_evaluation_samples(arrow_root: Path) -> dict[str, str]:
    from datasets import Dataset
    from evalscope.benchmarks.live_code_bench.load_utils import transform

    paths = sorted(arrow_root.glob("release_latest-*/0.0.0/master/*-test-*.arrow"))
    if len(paths) != 6:
        raise ValueError(f"expected six release_latest Arrow shards, found {len(paths)}")
    samples: dict[str, str] = {}
    for path in paths:
        for source in Dataset.from_file(str(path)):
            row = transform(source)
            prompt = PROMPT_TEMPLATE.format(
                question_content=row["question_content"],
                format_prompt=row["format_prompt"],
            )
            if prompt in samples:
                raise ValueError("duplicate rendered LiveCodeBench prompt")
            samples[prompt] = str(row["evaluation_sample"])
    return samples


def load_predictions(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    paths = sorted(root.glob("shards/shard_*/main/*/predictions/*/live_code_bench_*.jsonl"))
    if len(paths) != 6:
        raise ValueError(f"expected six prediction files, found {len(paths)}")
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    for path in paths:
        shard = next(part for part in path.parts if part.startswith("shard_"))
        sources.append({"path": str(path), "sha256": _sha256(path)})
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            row = json.loads(line)
            messages = row.get("messages") or []
            if not messages:
                raise ValueError(f"prediction has no prompt messages: {path}:{line_no}")
            assistant = next(
                (
                    message.get("content")
                    for message in reversed(messages)
                    if message.get("role") == "assistant"
                    and isinstance(message.get("content"), str)
                ),
                None,
            )
            if assistant is None:
                raise ValueError(f"prediction has no assistant content: {path}:{line_no}")
            rows.append(
                {
                    "index": row.get("index"),
                    "line_no": line_no,
                    # EvalScope persists model_output as an OpenAI-style response
                    # object. The canonical generated text is the assistant message.
                    "model_output": assistant,
                    "prompt": str(messages[0].get("content") or ""),
                    "shard": shard,
                    "source": str(path),
                }
            )
    return rows, sources


class _Adapter:
    def __init__(self, backend: PrivilegeDroppedLocalSandboxBackend) -> None:
        self.backend = backend

    def execute_code_in_sandbox(
        self, code: str | list[str], timeout: int = 60, language: str = "python"
    ) -> dict[str, Any]:
        return self.backend.execute(code, timeout, language)


def rescore_one(
    row: dict[str, Any],
    evaluation_sample: str,
    adapter: _Adapter,
    timeout: int,
) -> dict[str, Any]:
    from evalscope.benchmarks.live_code_bench.extract_utils import extract_code_generation
    from evalscope.benchmarks.live_code_bench.sandbox_evaluate_utils import evaluate_in_sandbox

    code = extract_code_generation(row["model_output"])
    passed, details = evaluate_in_sandbox(
        adapter,
        code,
        evaluation_sample,
        timeout=timeout,
        debug=False,
    )
    return {
        "code_chars": len(code),
        "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
        "details": details,
        "index": row["index"],
        "line_no": row["line_no"],
        "passed": bool(passed),
        "prompt_sha256": hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest(),
        "shard": row["shard"],
        "source": row["source"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--arrow-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sandbox-runner", type=Path, required=True)
    parser.add_argument("--sandbox-python", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=1055)
    parser.add_argument("--historical-score", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=6)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    samples = load_evaluation_samples(args.arrow_root.resolve())
    rows, source_predictions = load_predictions(args.prediction_root.resolve())
    if len(samples) != args.expected_rows or len(rows) != args.expected_rows:
        raise ValueError(
            f"coverage mismatch: dataset={len(samples)} predictions={len(rows)} "
            f"expected={args.expected_rows}"
        )
    missing = [row for row in rows if row["prompt"] not in samples]
    if missing:
        raise ValueError(f"{len(missing)} prediction prompts do not match the frozen dataset")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    backend = PrivilegeDroppedLocalSandboxBackend(
        args.sandbox_runner,
        args.sandbox_python,
        args.scratch_root,
    )
    backend.start()
    adapter = _Adapter(backend)
    completed = 0
    lock = threading.Lock()
    started = time.perf_counter()
    receipts: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                rescore_one,
                row,
                samples[row["prompt"]],
                adapter,
                args.timeout,
            ): row
            for row in rows
        }
        for future in as_completed(futures):
            receipt = future.result()
            receipts.append(receipt)
            with lock:
                completed += 1
                if completed % 25 == 0 or completed == len(rows):
                    elapsed = time.perf_counter() - started
                    print(
                        f"LCB_RESCORE_PROGRESS completed={completed}/{len(rows)} "
                        f"elapsed_s={elapsed:.1f}",
                        flush=True,
                    )
    backend.stop()

    receipts.sort(key=lambda row: (row["shard"], int(row["line_no"])))
    receipt_path = output_dir / "review_receipts.jsonl"
    with receipt_path.open("w", encoding="utf-8") as handle:
        for row in receipts:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    passed = sum(bool(row["passed"]) for row in receipts)
    per_shard: dict[str, dict[str, int | float]] = {}
    for shard in sorted({str(row["shard"]) for row in receipts}):
        subset = [row for row in receipts if row["shard"] == shard]
        shard_passed = sum(bool(row["passed"]) for row in subset)
        per_shard[shard] = {
            "passed": shard_passed,
            "rows": len(subset),
            "score": shard_passed / len(subset),
        }
    summary = {
        "corrected_score": passed / len(receipts),
        "historical_score": args.historical_score,
        "historical_score_valid": False,
        "passed": passed,
        "protocol": "cached_predictions_local_sandbox_output_contract_v2",
        "review_receipts_sha256": _sha256(receipt_path),
        "rows": len(receipts),
        "source_predictions": source_predictions,
        "source_predictions_unchanged": True,
        "per_shard": per_shard,
        "workers": args.workers,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
