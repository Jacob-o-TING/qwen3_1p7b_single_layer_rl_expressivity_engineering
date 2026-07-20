from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .distributed import gradient_accumulation_for_effective_batch


EXPECTED_REPORTS = {
    "paper_math500",
    "paper_gsm8k",
    "paper_olympiadbench",
    "paper_amc23",
}

CACHE_PHASE_DIRS = {
    "main": "main",
    "amc": "amc_average_at_32",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cached_unique_rows(cache_dir: Path) -> int:
    files = sorted(cache_dir.glob("reviews/*/*.jsonl"))
    if not files:
        files = sorted(cache_dir.glob("predictions/*/*.jsonl"))
    identities: set[tuple[str, object]] = set()
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                identities.add((path.stem, row.get("index", line_number)))
    return len(identities)


def resolve_best_evaluation_cache(eval_dir: Path, phase: str) -> Path | None:
    if phase not in CACHE_PHASE_DIRS:
        raise ValueError(f"Unknown evaluation cache phase: {phase}")
    eval_dir = eval_dir.resolve()
    phase_root = eval_dir / CACHE_PHASE_DIRS[phase]
    candidates = [path for path in phase_root.iterdir() if path.is_dir()] if phase_root.is_dir() else []
    manifest_path = eval_dir / "evaluation_manifest.json"
    preferred: Path | None = None
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        value = manifest.get(f"{phase}_use_cache")
        if value:
            candidate = Path(str(value)).resolve()
            if candidate.is_dir():
                preferred = candidate
                if candidate not in candidates:
                    candidates.append(candidate)
    scored = [
        (_cached_unique_rows(candidate), candidate == preferred, candidate.stat().st_mtime, candidate)
        for candidate in candidates
    ]
    scored = [entry for entry in scored if entry[0] > 0]
    return max(scored, key=lambda entry: entry[:3])[-1] if scored else None


def resolve_completed_checkpoint(run_dir: Path) -> Path:
    run_dir = run_dir.resolve()
    result = _read_json(run_dir / "train_result.json")
    manifest = _read_json(run_dir / "run_manifest.json")
    latest = _read_json(run_dir / "checkpoints" / "latest.json")
    if bool(result.get("benchmark")):
        raise ValueError(f"Benchmark run cannot be handed to final evaluation: {run_dir}")
    result_step = int(result.get("global_step", -1))
    total_steps = int(manifest.get("total_steps", -1))
    latest_step = int(latest.get("global_step", -1))
    if total_steps <= 0 or result_step != total_steps or latest_step != total_steps:
        raise ValueError(
            "Training is not exactly complete: "
            f"result_step={result_step}, latest_step={latest_step}, total_steps={total_steps}"
        )
    checkpoint_dir = (run_dir / "checkpoints" / str(latest["checkpoint"])).resolve()
    expected_name = f"step_{total_steps:08d}"
    if checkpoint_dir.name != expected_name:
        raise ValueError(f"Final checkpoint name is {checkpoint_dir.name}, expected {expected_name}")
    for name in ("trainable_state.pt", "trainer_state.pt", "manifest.json"):
        path = checkpoint_dir / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Final checkpoint file is missing or empty: {path}")
    checkpoint_manifest = _read_json(checkpoint_dir / "manifest.json")
    if int(checkpoint_manifest.get("total_steps", -1)) != total_steps:
        raise ValueError("Final checkpoint manifest does not match the completed run")
    return checkpoint_dir


def evaluation_is_complete(eval_dir: Path, checkpoint_dir: Path) -> bool:
    try:
        receipt = _read_json(eval_dir.resolve() / "evaluation_complete.json")
        if receipt.get("checkpoint_dir") != str(checkpoint_dir.resolve()):
            return False
        reports = receipt.get("reports")
        if not isinstance(reports, list) or len(reports) < len(EXPECTED_REPORTS):
            return False
        report_names = {str(item.get("dataset")) for item in reports if isinstance(item, dict)}
        if report_names != EXPECTED_REPORTS:
            return False
        for item in reports:
            path = Path(str(item["path"]))
            if not path.is_file() or _sha256(path) != item.get("sha256"):
                return False
        return True
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def record_completed_evaluation(
    eval_dir: Path,
    checkpoint_dir: Path,
    config_path: Path,
    *,
    allow_limited: bool = False,
) -> Path:
    eval_dir = eval_dir.resolve()
    checkpoint_dir = checkpoint_dir.resolve()
    manifest_path = eval_dir / "evaluation_manifest.json"
    manifest = _read_json(manifest_path)
    if Path(str(manifest.get("checkpoint_dir", ""))).resolve() != checkpoint_dir:
        raise ValueError("Evaluation manifest checkpoint does not match the completed training run")
    if manifest.get("limit") is not None and not allow_limited:
        raise ValueError("A limited preflight evaluation cannot receive a production completion receipt")
    report_candidates = sorted(eval_dir.glob("**/reports/*/*.json"))
    latest_reports: dict[str, Path] = {}
    for path in report_candidates:
        if path.stem in EXPECTED_REPORTS:
            latest_reports[path.stem] = path
    if set(latest_reports) != EXPECTED_REPORTS:
        raise ValueError(
            f"Expected reports {sorted(EXPECTED_REPORTS)}, found {sorted(latest_reports)}"
        )
    receipt = {
        "schema_version": 1,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_dir": str(checkpoint_dir),
        "config_path": str(config_path.resolve()),
        "config_sha256": _sha256(config_path.resolve()),
        "evaluation_manifest": str(manifest_path),
        "evaluation_manifest_sha256": _sha256(manifest_path),
        "limit": manifest.get("limit"),
        "reports": [
            {
                "dataset": dataset,
                "path": str(latest_reports[dataset].resolve()),
                "sha256": _sha256(latest_reports[dataset]),
            }
            for dataset in sorted(latest_reports)
        ],
    }
    receipt_path = eval_dir / "evaluation_complete.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return receipt_path


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    accumulation = subparsers.add_parser("accumulation")
    accumulation.add_argument("--target", type=int, required=True)
    accumulation.add_argument("--world-size", type=int, required=True)
    accumulation.add_argument("--micro-batch-size", type=int, default=1)

    resolve = subparsers.add_parser("resolve-checkpoint")
    resolve.add_argument("--run-dir", type=Path, required=True)

    status = subparsers.add_parser("eval-status")
    status.add_argument("--eval-dir", type=Path, required=True)
    status.add_argument("--checkpoint-dir", type=Path, required=True)

    record = subparsers.add_parser("record-eval")
    record.add_argument("--eval-dir", type=Path, required=True)
    record.add_argument("--checkpoint-dir", type=Path, required=True)
    record.add_argument("--config", type=Path, required=True)
    record.add_argument("--allow-limited", action="store_true")

    cache = subparsers.add_parser("resolve-eval-cache")
    cache.add_argument("--eval-dir", type=Path, required=True)
    cache.add_argument("--phase", choices=sorted(CACHE_PHASE_DIRS), required=True)

    args = parser.parse_args()
    if args.command == "accumulation":
        print(
            gradient_accumulation_for_effective_batch(
                args.target,
                world_size=args.world_size,
                micro_batch_size=args.micro_batch_size,
            )
        )
    elif args.command == "resolve-checkpoint":
        print(resolve_completed_checkpoint(args.run_dir))
    elif args.command == "eval-status":
        if evaluation_is_complete(args.eval_dir, args.checkpoint_dir):
            print("complete")
        else:
            print("incomplete")
            raise SystemExit(1)
    elif args.command == "record-eval":
        print(
            record_completed_evaluation(
                args.eval_dir,
                args.checkpoint_dir,
                args.config,
                allow_limited=args.allow_limited,
            )
        )
    elif args.command == "resolve-eval-cache":
        resolved = resolve_best_evaluation_cache(args.eval_dir, args.phase)
        if resolved is None:
            raise SystemExit(1)
        print(resolved)


if __name__ == "__main__":
    main()
