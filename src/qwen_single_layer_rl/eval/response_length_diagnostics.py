from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable


LENGTH_BUCKETS = (256, 512, 1024, 1536, 2048, 2560, 3072)


def has_boxed_answer(text: str) -> bool:
    """Return whether text contains a non-empty, brace-balanced \boxed{...}."""
    marker = "\\boxed{"
    start = 0
    while True:
        index = text.find(marker, start)
        if index < 0:
            return False
        depth = 1
        cursor = index + len(marker)
        content_start = cursor
        while cursor < len(text) and depth:
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth == 0 and text[content_start : cursor - 1].strip():
            return True
        start = index + len(marker)


def _assistant_content(record: dict[str, Any]) -> str:
    for message in reversed(record.get("messages") or []):
        if message.get("role") == "assistant":
            return str(message.get("content") or "")
    return ""


def _score_fields(review: dict[str, Any]) -> tuple[Any, float]:
    score = (review.get("sample_score") or {}).get("score") or {}
    extracted = score.get("extracted_prediction")
    value = (score.get("value") or {}).get("acc", 0.0)
    return extracted, float(value or 0.0)


def _is_extracted(value: Any) -> bool:
    return value is not None and bool(str(value).strip())


def _percentile(values: list[int], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bucket_label(token_count: int) -> str:
    lower = 0
    for upper in LENGTH_BUCKETS:
        if token_count <= upper:
            return f"{lower}-{upper}"
        lower = upper + 1
    return f">{LENGTH_BUCKETS[-1]}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc


def _prediction_path(review_path: Path) -> Path:
    parts = list(review_path.parts)
    try:
        parts[parts.index("reviews")] = "predictions"
    except ValueError as exc:
        raise ValueError(f"Review path has no reviews directory: {review_path}") from exc
    return Path(*parts)


def collect_rows(
    evaluation_root: Path,
    *,
    encode: Callable[[str], list[int]],
    token_cap: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    review_paths = sorted(evaluation_root.glob("**/reviews/**/*.jsonl"))
    if not review_paths:
        raise FileNotFoundError(f"No EvalScope review JSONL files under {evaluation_root}")

    for review_path in review_paths:
        prediction_path = _prediction_path(review_path)
        if not prediction_path.is_file():
            raise FileNotFoundError(f"Missing matching prediction file: {prediction_path}")
        benchmark = review_path.stem.removesuffix("_main")
        predictions = {int(row["index"]): row for row in _read_jsonl(prediction_path)}
        sources.extend(
            [
                {"path": str(review_path), "sha256": _sha256(review_path)},
                {"path": str(prediction_path), "sha256": _sha256(prediction_path)},
            ]
        )
        for review in _read_jsonl(review_path):
            index = int(review["index"])
            prediction = predictions.get(index)
            if prediction is None:
                raise ValueError(f"No prediction for {benchmark} index={index}")
            response = _assistant_content(review) or _assistant_content(prediction)
            token_count = len(encode(response))
            extracted, accuracy = _score_fields(review)
            choices = (prediction.get("model_output") or {}).get("choices") or []
            stop_reason = choices[0].get("stop_reason") if choices else None
            metadata = (review.get("sample_score") or {}).get("sample_metadata") or {}
            unique_id = metadata.get("unique_id") or metadata.get("id") or index
            cap_hit = token_count >= token_cap
            extracted_present = _is_extracted(extracted)
            cell = ("A" if extracted_present else "B") if cap_hit else (
                "C" if extracted_present else "D"
            )
            rows.append(
                {
                    "benchmark": benchmark,
                    "index": index,
                    "row_id": f"{benchmark}:{unique_id}:{index}",
                    "sample_id": (review.get("sample_score") or {}).get("sample_id"),
                    "group_id": (review.get("sample_score") or {}).get("group_id"),
                    "generated_tokens": token_count,
                    "token_count_provenance": "retokenized_text",
                    "cap_hit_proxy": cap_hit,
                    "reported_stop_reason": stop_reason,
                    "extracted_answer_present": extracted_present,
                    "boxed_answer_present": has_boxed_answer(response),
                    "accuracy": accuracy,
                    "cell": cell,
                }
            )
    return rows, sources


def _rate(numerator: int | float, denominator: int) -> float | None:
    return float(numerator) / denominator if denominator else None


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    token_counts = [int(row["generated_tokens"]) for row in rows]
    cells = Counter(str(row["cell"]) for row in rows)
    cap_rows = [row for row in rows if row["cap_hit_proxy"]]
    noncap_rows = [row for row in rows if not row["cap_hit_proxy"]]
    missing_rows = [row for row in rows if not row["extracted_answer_present"]]

    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        bucket = _bucket_label(int(row["generated_tokens"]))
        item = buckets.setdefault(bucket, {"count": 0, "correct": 0.0})
        item["count"] += 1
        item["correct"] += float(row["accuracy"])
    for item in buckets.values():
        item["accuracy"] = _rate(item.pop("correct"), item["count"])

    def accuracy(items: list[dict[str, Any]]) -> float | None:
        return _rate(sum(float(row["accuracy"]) for row in items), len(items))

    def missing_box_rate(items: list[dict[str, Any]]) -> float | None:
        return _rate(sum(not row["boxed_answer_present"] for row in items), len(items))

    return {
        "count": len(rows),
        "cells": {name: cells.get(name, 0) for name in "ABCD"},
        "cap_hit_rate": _rate(len(cap_rows), len(rows)),
        "missing_extraction_rate": _rate(len(missing_rows), len(rows)),
        "p_missing_given_cap_hit": _rate(cells.get("B", 0), len(cap_rows)),
        "p_cap_hit_given_missing": _rate(cells.get("B", 0), len(missing_rows)),
        "accuracy_cap_hit": accuracy(cap_rows),
        "accuracy_non_cap": accuracy(noncap_rows),
        "missing_box_rate_cap_hit": missing_box_rate(cap_rows),
        "missing_box_rate_non_cap": missing_box_rate(noncap_rows),
        "generated_token_percentiles": {
            "p50": _percentile(token_counts, 0.50),
            "p90": _percentile(token_counts, 0.90),
            "p95": _percentile(token_counts, 0.95),
            "p99": _percentile(token_counts, 0.99),
            "max": max(token_counts) if token_counts else None,
        },
        "accuracy_by_length_bucket": buckets,
    }


def build_diagnostics(
    rows: list[dict[str, Any]],
    *,
    sources: list[dict[str, str]],
    token_cap: int,
    sample_seed: int,
    samples_per_cell: int,
    tokenizer_identity: str,
) -> dict[str, Any]:
    benchmarks = sorted({str(row["benchmark"]) for row in rows})
    summaries = {
        benchmark: summarize_rows([row for row in rows if row["benchmark"] == benchmark])
        for benchmark in benchmarks
    }
    rng = random.Random(sample_seed)
    manual_samples: dict[str, list[dict[str, Any]]] = {}
    for benchmark in benchmarks:
        for cell in ("B", "D"):
            candidates = [
                row for row in rows if row["benchmark"] == benchmark and row["cell"] == cell
            ]
            rng.shuffle(candidates)
            manual_samples[f"{benchmark}:{cell}"] = candidates[:samples_per_cell]
    return {
        "schema_version": 1,
        "analysis": "cap_hit_vs_answer_extraction",
        "token_cap": token_cap,
        "cap_hit_definition": "retokenized generated text token count >= token_cap",
        "finish_reason_limitation": (
            "The current adapter reports stop for all decoded outputs; cap status is a text "
            "retokenization proxy, not an exact generation finish reason."
        ),
        "tokenizer_identity": tokenizer_identity,
        "sample_seed": sample_seed,
        "samples_per_cell": samples_per_cell,
        "source_artifacts": sources,
        "overall": summarize_rows(rows),
        "benchmarks": summaries,
        "manual_inspection_samples": manual_samples,
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def render_markdown(diagnostics: dict[str, Any]) -> str:
    lines = [
        "# Response Length Diagnostics",
        "",
        "Cap status is reconstructed from retokenized response text and is not an exact finish reason.",
        "",
        "| Benchmark | N | A | B | C | D | Cap hit | Missing extraction | Accuracy cap | Accuracy non-cap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for benchmark, summary in diagnostics["benchmarks"].items():
        cells = summary["cells"]
        lines.append(
            f"| {benchmark} | {summary['count']} | {cells['A']} | {cells['B']} | "
            f"{cells['C']} | {cells['D']} | {_pct(summary['cap_hit_rate'])} | "
            f"{_pct(summary['missing_extraction_rate'])} | "
            f"{_pct(summary['accuracy_cap_hit'])} | {_pct(summary['accuracy_non_cap'])} |"
        )
    lines.extend(["", "## Token Length Percentiles", ""])
    lines.extend(
        f"- `{benchmark}`: "
        + ", ".join(f"{key}={value:.1f}" for key, value in summary["generated_token_percentiles"].items())
        for benchmark, summary in diagnostics["benchmarks"].items()
    )
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "Cell A is cap-hit with an extracted answer; B is cap-hit without one; C is non-cap "
            "with an extracted answer; D is non-cap without one. Current cap-hit labels use "
            "`retokenized_text` provenance.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only EvalScope response-length audit")
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--token-cap", type=int, default=3072)
    parser.add_argument("--sample-seed", type=int, default=20260711)
    parser.add_argument("--samples-per-cell", type=int, default=10)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    evaluation_root = args.evaluation_root.resolve()
    manifest_path = evaluation_root / "evaluation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_path = args.model_path or Path(str(manifest["model_path"]))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    rows, sources = collect_rows(
        evaluation_root,
        encode=lambda text: tokenizer.encode(text, add_special_tokens=False),
        token_cap=args.token_cap,
    )
    diagnostics = build_diagnostics(
        rows,
        sources=sources,
        token_cap=args.token_cap,
        sample_seed=args.sample_seed,
        samples_per_cell=args.samples_per_cell,
        tokenizer_identity=str(model_path.resolve()),
    )
    output_dir = (args.output_dir or evaluation_root / "diagnostics").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "response_length_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "response_length_diagnostics.md").write_text(
        render_markdown(diagnostics), encoding="utf-8"
    )
    (output_dir / "response_length_rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    print(render_markdown(diagnostics))


if __name__ == "__main__":
    main()
