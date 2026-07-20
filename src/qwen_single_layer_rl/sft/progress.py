from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_run(run_dir: Path, *, window: int = 100) -> dict[str, Any] | None:
    metrics_path = run_dir / "metrics.jsonl"
    manifest_path = run_dir / "run_manifest.json"
    if not metrics_path.is_file() or not manifest_path.is_file():
        return None
    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line]
    steps = [record for record in records if record.get("event") == "step"]
    if not steps:
        return None
    validations = [record for record in records if record.get("event") == "validation"]
    manifest = _read_json(manifest_path)
    first = steps[:window]
    recent = steps[-window:]
    first_loss = statistics.mean(float(record["loss"]) for record in first)
    recent_loss = statistics.mean(float(record["loss"]) for record in recent)
    median_seconds = statistics.median(float(record["step_seconds"]) for record in recent)
    latest_step = int(steps[-1]["global_step"])
    total_steps = int(manifest["total_steps"])
    return {
        "variant": manifest.get("variant", run_dir.name),
        "run_dir": str(run_dir),
        "latest_step": latest_step,
        "total_steps": total_steps,
        "progress": latest_step / total_steps,
        "latest_loss": float(steps[-1]["loss"]),
        "first_window_count": len(first),
        "first_window_loss_mean": first_loss,
        "recent_window_count": len(recent),
        "recent_window_loss_mean": recent_loss,
        "loss_mean_delta": recent_loss - first_loss,
        "recent_step_seconds_median": median_seconds,
        "eta_train_seconds": max(0, total_steps - latest_step) * median_seconds,
        "validations": [
            {
                "step": int(record["global_step"]),
                "loss": float(record["validation_loss"]),
            }
            for record in validations
        ],
    }


def _duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_summary(summary: dict[str, Any]) -> str:
    validations = ",".join(
        f"{item['step']}:{item['loss']:.6f}" for item in summary["validations"]
    ) or "none"
    return (
        f"SFT_TREND variant={summary['variant']} "
        f"step={summary['latest_step']}/{summary['total_steps']} "
        f"progress={summary['progress']:.2%} latest_loss={summary['latest_loss']:.6f} "
        f"first{summary['first_window_count']}_mean={summary['first_window_loss_mean']:.6f} "
        f"last{summary['recent_window_count']}_mean={summary['recent_window_loss_mean']:.6f} "
        f"delta={summary['loss_mean_delta']:+.6f} "
        f"median_step_s={summary['recent_step_seconds_median']:.3f} "
        f"eta_train={_duration(summary['eta_train_seconds'])} validations={validations}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--window", type=int, default=100)
    args = parser.parse_args()
    if args.window <= 0:
        raise SystemExit("--window must be positive")
    summaries = [
        summary
        for metrics_path in sorted(args.run_root.glob("*/metrics.jsonl"))
        if (summary := summarize_run(metrics_path.parent, window=args.window)) is not None
    ]
    if not summaries:
        print("SFT_TREND no step metrics available")
        return
    for summary in summaries:
        print(format_summary(summary))


if __name__ == "__main__":
    main()
