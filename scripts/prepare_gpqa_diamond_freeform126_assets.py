from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import yaml

from qwen_single_layer_rl.eval.gpqa_freeform import (
    canonical_json_sha256,
    file_sha256,
    normalize_dataset_rows,
    write_json_atomic,
    write_jsonl_atomic,
)


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("free-form eval config must be a mapping")
    return value


def write_json_immutable(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"refusing to overwrite mismatched pinned asset receipt: {path}")
        return
    write_json_atomic(path, payload)


def write_jsonl_immutable(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        existing = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if existing != rows:
            raise RuntimeError(f"refusing to overwrite mismatched pinned asset ledger: {path}")
        return
    write_jsonl_atomic(path, rows)


def prepare_dataset(config: dict[str, Any], root: Path) -> None:
    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download

    spec = config["dataset"]
    output = root / spec["asset_root"]
    output.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    source_mode = "canonical_gated_dataset"
    source_repo = spec["repo_id"]
    source_file = None
    requested_revision = spec["revision"]
    canonical_error = None
    try:
        info = HfApi(token=token).dataset_info(spec["repo_id"], revision=spec["revision"])
        revision = str(info.sha)
        kwargs: dict[str, Any] = {
            "path": spec["repo_id"],
            "split": spec["split"],
            "revision": revision,
            "token": token,
        }
        if spec.get("config"):
            kwargs["name"] = spec["config"]
        source_rows = [dict(row) for row in load_dataset(**kwargs)]
    except Exception as exc:
        if not spec.get("public_fallback_repo_id"):
            raise
        source_mode = "maintainer_public_filtered_csv_fallback"
        source_repo = spec["public_fallback_repo_id"]
        source_file = spec["public_fallback_filename"]
        requested_revision = spec["public_fallback_revision"]
        canonical_error = type(exc).__name__
        info = HfApi(token=token).dataset_info(
            source_repo,
            revision=spec["public_fallback_revision"],
        )
        revision = str(info.sha)
        csv_path = Path(
            hf_hub_download(
                repo_id=source_repo,
                repo_type="dataset",
                filename=source_file,
                revision=revision,
                token=token,
            )
        )
        with csv_path.open("r", encoding="utf-8-sig", newline="") as source:
            raw_rows = list(csv.DictReader(source))
        required = {"Question", "Answer", "Record ID"}
        if not raw_rows or not required.issubset(raw_rows[0]):
            raise RuntimeError(f"public GPQA fallback schema drift: {sorted(raw_rows[0] if raw_rows else [])}")
        source_rows = [
            {
                "question_id": row["Record ID"],
                "question": row["Question"],
                "answer": row["Answer"],
            }
            for row in raw_rows
        ]
    rows = normalize_dataset_rows(source_rows)
    normalized = output / "gpqa_diamond_freeform126.normalized.jsonl"
    ledger = output / "gpqa_diamond_freeform126.ledger.jsonl"
    write_jsonl_immutable(normalized, rows)
    write_jsonl_immutable(ledger, rows)
    manifest = {
        "status": "PINNED_DATASET_READY",
        "repo_id": source_repo,
        "source_mode": source_mode,
        "source_file": source_file,
        "canonical_repo_id": spec["repo_id"],
        "canonical_access_error": canonical_error,
        "requested_revision": requested_revision,
        "resolved_revision": revision,
        "config": spec.get("config"),
        "split": spec["split"],
        "rows": len(rows),
        "unique_question_ids": len({row["question_id"] for row in rows}),
        "assignment": "ordered_index % 6",
        "rows_per_generation_rank": [
            sum(int(row["generation_rank"]) == rank for row in rows) for rank in range(6)
        ],
        "ordered_question_ids_sha256": canonical_json_sha256([row["question_id"] for row in rows]),
        "normalized_path": str(normalized.resolve()),
        "normalized_sha256": file_sha256(normalized),
        "ledger_path": str(ledger.resolve()),
        "ledger_sha256": file_sha256(ledger),
        "credentials_recorded": False,
        "public_fallback_not_claimed_byte_identical_to_gated_repo": source_mode != "canonical_gated_dataset",
    }
    write_json_immutable(output / "dataset_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def prepare_matcher(config: dict[str, Any], root: Path) -> None:
    from huggingface_hub import HfApi, snapshot_download

    spec = config["matcher"]
    output = Path(spec["model_path"])
    if not output.is_absolute():
        output = root / output
    token = os.environ.get("HF_TOKEN")
    info = HfApi(token=token).model_info(spec["repo_id"], revision=spec["revision"])
    revision = str(info.sha)
    snapshot_download(
        repo_id=spec["repo_id"],
        revision=revision,
        local_dir=output,
        token=token,
    )
    required = ("config.json", "tokenizer_config.json")
    missing = [name for name in required if not (output / name).is_file()]
    if missing or not list(output.glob("*.safetensors")):
        raise RuntimeError(f"matcher snapshot is incomplete: missing={missing}")
    tracked = [output / name for name in required]
    tracked.extend(
        output / name
        for name in ("tokenizer.json", "tokenizer.model", "generation_config.json", "special_tokens_map.json")
        if (output / name).is_file()
    )
    tracked.extend(sorted(output.glob("*.safetensors.index.json")))
    receipt = {
        "status": "PINNED_MATCHER_READY",
        "repo_id": spec["repo_id"],
        "requested_revision": spec["revision"],
        "resolved_revision": revision,
        "model_path": str(output.resolve()),
        "dtype": spec["dtype"],
        "tracked_files": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
            for path in tracked
        ],
        "weight_files": [
            {"name": path.name, "bytes": path.stat().st_size} for path in sorted(output.glob("*.safetensors"))
        ],
        "credentials_recorded": False,
    }
    write_json_immutable(root / spec["receipt"], receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("action", choices=("dataset", "matcher", "all"))
    args = parser.parse_args()
    root = args.root.resolve()
    config = load_config(args.config)
    if args.action in ("dataset", "all"):
        prepare_dataset(config, root)
    if args.action in ("matcher", "all"):
        prepare_matcher(config, root)


if __name__ == "__main__":
    main()
