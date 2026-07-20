from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path

from qwen_single_layer_rl.eval.humanevalplus_parser import (
    parse_humanevalplus_prediction,
)


def _assistant_content(row: dict[str, object]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            content = message.get("content")
            return content if isinstance(content, str) else ""
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output_dir / "parser_receipts.jsonl"
    strategies: Counter[str] = Counter()
    raw_syntax_ok = 0
    parsed_syntax_ok = 0
    rows = 0

    with args.prediction_jsonl.open(encoding="utf-8") as source, receipt_path.open(
        "w", encoding="utf-8"
    ) as target:
        for line in source:
            row = json.loads(line)
            raw = _assistant_content(row)
            parsed, receipt = parse_humanevalplus_prediction(raw)
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            prompt = metadata.get("prompt", "")
            test = metadata.get("test", "")
            entry_point = metadata.get("entry_point", "")
            raw_program = f"{prompt}{raw}\n{test}\ncheck({entry_point})"
            parsed_program = f"{prompt}{parsed}\n{test}\ncheck({entry_point})"

            raw_ok = parsed_ok = False
            try:
                ast.parse(raw_program)
                raw_ok = True
            except SyntaxError:
                pass
            try:
                ast.parse(parsed_program)
                parsed_ok = True
            except SyntaxError:
                pass

            rows += 1
            raw_syntax_ok += raw_ok
            parsed_syntax_ok += parsed_ok
            strategies[receipt.strategy] += 1
            target.write(
                json.dumps(
                    {
                        "index": row.get("index"),
                        "task_id": metadata.get("task_id"),
                        "raw_syntax_ok": raw_ok,
                        "parsed_syntax_ok": parsed_ok,
                        **receipt.to_dict(),
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    summary = {
        "status": "parser_audit_only_no_generation_no_execution",
        "source_prediction_jsonl": str(args.prediction_jsonl.resolve()),
        "rows": rows,
        "raw_combined_program_syntax_ok": raw_syntax_ok,
        "parsed_combined_program_syntax_ok": parsed_syntax_ok,
        "strategies": dict(sorted(strategies.items())),
        "receipt_jsonl": str(receipt_path.resolve()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
