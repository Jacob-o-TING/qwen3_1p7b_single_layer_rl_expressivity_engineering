from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from qwen_single_layer_rl.eval.humanevalplus_parser import (
    parse_humanevalplus_prediction,
)
from qwen_single_layer_rl.eval.local_code_sandbox import (
    PrivilegeDroppedLocalSandboxBackend,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assistant_content(row: dict[str, object]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("prediction row has no messages list")
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            content = message.get("content")
            if isinstance(content, str):
                return content
    raise ValueError("prediction row has no assistant content")


def _parser_output(text: str, mode: str) -> tuple[str, dict[str, object]]:
    if mode == "improved":
        parsed, receipt = parse_humanevalplus_prediction(text)
        return parsed, receipt.to_dict()
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return text, {
        "strategy": "raw_unchanged_control",
        "changed": False,
        "raw_sha256": digest,
        "parsed_sha256": digest,
        "removed_prefix_chars": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_jsonl", type=Path)
    parser.add_argument("raw_report_json", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sandbox-runner", type=Path, required=True)
    parser.add_argument("--sandbox-python", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--parser-mode", choices=("improved", "raw"), default="improved")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output_dir / "review_receipts.jsonl"
    summary_path = args.output_dir / "summary.json"
    if receipt_path.exists() or summary_path.exists():
        raise SystemExit(f"refusing to overwrite existing parser-only review: {args.output_dir}")

    raw_report = json.loads(args.raw_report_json.read_text(encoding="utf-8"))
    backend = PrivilegeDroppedLocalSandboxBackend(
        args.sandbox_runner,
        args.sandbox_python,
        args.scratch_root,
    )
    backend.start()
    started = time.perf_counter()
    passed = 0
    rows = 0
    strategies: Counter[str] = Counter()

    try:
        with args.prediction_jsonl.open(encoding="utf-8") as source, receipt_path.open(
            "x", encoding="utf-8"
        ) as target:
            for line in source:
                row = json.loads(line)
                metadata = row.get("metadata")
                if not isinstance(metadata, dict):
                    raise ValueError(f"prediction row {row.get('index')} has no metadata")
                raw = _assistant_content(row)
                parsed, parser_receipt = _parser_output(raw, args.parser_mode)
                program = (
                    f"{metadata['prompt']}{parsed}\n{metadata['test']}\n"
                    f"check({metadata['entry_point']})"
                )
                result = backend.execute(program, args.timeout, "python")
                is_pass = result.get("status") == "success"
                passed += int(is_pass)
                rows += 1
                strategies[str(parser_receipt["strategy"])] += 1
                target.write(
                    json.dumps(
                        {
                            "index": row.get("index"),
                            "task_id": metadata.get("task_id"),
                            "passed": is_pass,
                            "execution_status": result.get("status"),
                            "execution_exit_code": result.get("exit_code"),
                            "execution_stderr_tail": str(result.get("stderr", ""))[-2000:],
                            **parser_receipt,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
    finally:
        backend.stop()

    summary = {
        "status": "complete_parser_only_cached_predictions_no_generation",
        "scientific_boundary": {
            "model_loaded": False,
            "checkpoint_mutated": False,
            "generation_invoked": False,
            "raw_report_preserved": True,
        },
        "source_prediction_jsonl": str(args.prediction_jsonl.resolve()),
        "source_prediction_sha256": _sha256(args.prediction_jsonl),
        "raw_report_json": str(args.raw_report_json.resolve()),
        "raw_report_sha256": _sha256(args.raw_report_json),
        "raw_report_score": raw_report.get("score"),
        "parser_mode": args.parser_mode,
        "rows": rows,
        "passed": passed,
        "parser_only_score": passed / rows if rows else None,
        "strategies": dict(sorted(strategies.items())),
        "timeout_seconds_per_sample": args.timeout,
        "elapsed_seconds": time.perf_counter() - started,
        "review_receipts": str(receipt_path.resolve()),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
