#!/usr/bin/env python3
"""Run preregistered RTX 5090 KV-balanced SHS throughput profiles."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ENGINE_PATTERNS = {
    "available_kv_cache_gib": re.compile(r"Available KV cache memory:\s*([0-9.]+) GiB"),
    "kv_cache_tokens": re.compile(r"GPU KV cache size:\s*([0-9,]+) tokens"),
    "full_context_concurrency": re.compile(r"Maximum concurrency for 4,096 tokens per request:\s*([0-9.]+)x"),
    "peak_activation_gib": re.compile(r"([0-9.]+) GiB for peak activation"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def root_for(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["project_root"]) / config["run"]["output_root"]


def build_profile_config(config: dict[str, Any], profile_name: str, root: Path) -> dict[str, Any]:
    template = load_yaml(Path(config["paths"]["staged_template"]))
    profile = config["profiles"][profile_name]
    generated = copy.deepcopy(template)
    generated["run"]["id"] = config["run"]["id"]
    generated["run"]["screen"] = config["run"]["screen"]
    generated["run"]["output_root"] = str(
        Path(config["run"]["output_root"]) / "profiles" / profile["label"]
    )
    generated["run"]["log"] = str(Path(config["run"]["log"]).with_suffix(f".{profile['label']}.log"))
    generated["identity"]["base_source_commit"] = config["identity"]["base_source_commit"]
    generated["engine"]["max_num_batched_tokens"] = int(profile["max_num_batched_tokens"])
    generated["engine"]["gpu_memory_utilization"] = float(profile["gpu_memory_utilization"])
    generated["balanced_profile"] = {
        "name": profile_name,
        "label": profile["label"],
        "parent_run_id": config["run"]["id"],
    }
    return generated


def parse_engine_profile(worker_log: Path) -> dict[str, float | int | None]:
    text = worker_log.read_text(encoding="utf-8", errors="replace")
    result: dict[str, float | int | None] = {}
    for name, pattern in ENGINE_PATTERNS.items():
        matches = pattern.findall(text)
        if not matches:
            result[name] = None
            continue
        value = matches[-1]
        result[name] = int(value.replace(",", "")) if name == "kv_cache_tokens" else float(value)
    return result


def profile_receipt(profile_root: Path) -> dict[str, Any]:
    manifest_path = profile_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    engine_profiles = []
    for gpu_index in (0, 1):
        worker_log = profile_root / f"gpu{gpu_index}" / "worker.log"
        engine_profiles.append(
            {"gpu_index": gpu_index, **(parse_engine_profile(worker_log) if worker_log.exists() else {})}
        )
    oom = False
    for path in profile_root.glob("gpu*/cells/*.json"):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        oom = oom or row.get("status") == "oom"
    return {
        "manifest_path": str(manifest_path),
        "status": None if manifest is None else manifest.get("status"),
        "decision": None if manifest is None else manifest.get("decision"),
        "aggregate": None if manifest is None else manifest.get("aggregate"),
        "engine_profiles": engine_profiles,
        "oom": oom,
    }


def conditional_decision(receipt: dict[str, Any], trigger: dict[str, Any]) -> dict[str, Any]:
    profiles = receipt["engine_profiles"]
    required = ("kv_cache_tokens", "full_context_concurrency", "peak_activation_gib", "available_kv_cache_gib")
    missing = [
        f"gpu{row['gpu_index']}:{key}"
        for row in profiles
        for key in required
        if row.get(key) is None
    ]
    if missing:
        return {"action": "fail_stop", "reasons": ["missing_engine_profile"], "missing": missing}
    reasons = []
    if receipt["oom"] and trigger["run_on_primary_oom"]:
        reasons.append("primary_oom")
    if min(int(row["kv_cache_tokens"]) for row in profiles) < int(trigger["minimum_kv_cache_tokens"]):
        reasons.append("kv_cache_tokens_below_threshold")
    if min(float(row["full_context_concurrency"]) for row in profiles) < float(trigger["minimum_full_context_concurrency"]):
        reasons.append("full_context_concurrency_below_threshold")
    return {"action": "run_conditional" if reasons else "accept_primary", "reasons": reasons, "missing": []}


def find_cell(manifest: dict[str, Any], cell: str, pressure: int) -> dict[str, Any] | None:
    for row in manifest.get("aggregate", {}).get("cells", []):
        if row["cell"] == cell and int(row["pressure_per_gpu"]) == pressure:
            return row
    return None


def sampling_stability(control_root: Path, selected_root: Path) -> dict[str, Any]:
    results = {}
    for pressure in (16, 32, 64):
        rows = []
        for root in (control_root, selected_root):
            requests = {}
            for gpu_index in (0, 1):
                path = root / f"gpu{gpu_index}" / "cells" / f"production_cap3072_p{pressure}.json"
                cell = json.loads(path.read_text(encoding="utf-8"))
                requests.update({row["request_id"]: row for row in cell["requests"]})
            rows.append(requests)
        control, selected = rows
        identities = sorted(set(control) & set(selected))
        absolute_length_deltas = [
            abs(control[identity]["generated_tokens"] - selected[identity]["generated_tokens"])
            for identity in identities
        ]
        results[str(pressure)] = {
            "request_count": len(identities),
            "seed_equal_count": sum(control[identity]["seed"] == selected[identity]["seed"] for identity in identities),
            "token_trace_equal_count": sum(
                control[identity]["token_ids_sha256"] == selected[identity]["token_ids_sha256"]
                for identity in identities
            ),
            "length_equal_count": sum(
                control[identity]["generated_tokens"] == selected[identity]["generated_tokens"]
                for identity in identities
            ),
            "control_generated_tokens": sum(control[identity]["generated_tokens"] for identity in identities),
            "balanced_generated_tokens": sum(selected[identity]["generated_tokens"] for identity in identities),
            "mean_absolute_length_delta": sum(absolute_length_deltas) / len(absolute_length_deltas),
            "max_absolute_length_delta": max(absolute_length_deltas),
        }
    return results


def build_comparison(
    control: dict[str, Any],
    selected_manifest: dict[str, Any],
    profile_receipt_row: dict[str, Any],
    control_root: Path,
    selected_root: Path,
) -> dict[str, Any]:
    control_matched = find_cell(control, "matched_800_1024", 64)
    control_production = find_cell(control, "production_cap3072", 64)
    selected_matched = find_cell(selected_manifest, "matched_800_1024", 64)
    selected_production = find_cell(selected_manifest, "production_cap3072", 64)
    if not all((control_matched, control_production, selected_matched, selected_production)):
        raise RuntimeError("missing pressure-64 comparison cell")
    selected_p16_matched = find_cell(selected_manifest, "matched_800_1024", 16)
    selected_p16_production = find_cell(selected_manifest, "production_cap3072", 16)
    return {
        "matched_pressure64": {
            "control_pair_tokens_per_second": control_matched["pair_generated_tokens_per_second"],
            "balanced_pair_tokens_per_second": selected_matched["pair_generated_tokens_per_second"],
            "balanced_over_control": selected_matched["pair_generated_tokens_per_second"] / control_matched["pair_generated_tokens_per_second"],
            "balanced_mean_per_gpu": selected_matched["mean_per_gpu_generated_tokens_per_second"],
            "rtx_pro6000_anchor_per_gpu": 2262.8,
            "balanced_over_rtx_pro6000": selected_matched["mean_per_gpu_generated_tokens_per_second"] / 2262.8,
        },
        "production_pressure64": {
            "control_pair_tokens_per_second": control_production["pair_generated_tokens_per_second"],
            "balanced_pair_tokens_per_second": selected_production["pair_generated_tokens_per_second"],
            "balanced_over_control": selected_production["pair_generated_tokens_per_second"] / control_production["pair_generated_tokens_per_second"],
            "output_length_mean": selected_production["output_length_mean"],
            "output_length_p50": selected_production["output_length_p50"],
            "output_length_p90": selected_production["output_length_p90"],
            "cap_hit_fraction": selected_production["cap_hit_fraction"],
            "pair_start_skew_seconds": selected_production["start_skew_seconds"],
            "pair_overlap_fraction": selected_production["concurrent_overlap_fraction"],
            "peak_memory_mib_by_gpu": selected_production["peak_memory_mib_by_gpu"],
            "request_latency_p50_seconds": selected_production["request_latency_p50_seconds"],
            "request_latency_p90_seconds": selected_production["request_latency_p90_seconds"],
        },
        "pressure_scaling_efficiency": {
            "matched_p64_over_p16": selected_matched["pair_generated_tokens_per_second"] / selected_p16_matched["pair_generated_tokens_per_second"],
            "production_p64_over_p16": selected_production["pair_generated_tokens_per_second"] / selected_p16_production["pair_generated_tokens_per_second"],
            "ideal_pressure_ratio": 4.0,
        },
        "engine_profile": profile_receipt_row["engine_profiles"],
        "sampling_stability": sampling_stability(control_root, selected_root),
    }


def run_profile(
    args: argparse.Namespace,
    config: dict[str, Any],
    profile_name: str,
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    generated = build_profile_config(config, profile_name, root)
    profile = config["profiles"][profile_name]
    config_path = root / "generated_configs" / f"{profile['label']}.yaml"
    write_yaml(config_path, generated)
    profile_root = Path(config["paths"]["project_root"]) / generated["run"]["output_root"]
    command = [
        config["paths"]["python"],
        config["paths"]["staged_runner"],
        "--config",
        str(config_path),
    ]
    log_path = root / f"{profile['label']}_launcher.log"
    with log_path.open("a", encoding="utf-8") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    receipt = profile_receipt(profile_root)
    receipt["launcher_return_code"] = completed.returncode
    receipt["profile"] = profile
    atomic_write_json(root / f"{profile['label']}_receipt.json", receipt)
    return generated, receipt


def write_report(root: Path, manifest: dict[str, Any]) -> None:
    comparison = manifest.get("comparison")
    lines = [
        "# SHS 2xRTX 5090 KV-Balanced Throughput",
        "",
        f"Status: **{manifest['status']}**",
        "",
        f"Selected profile: `{manifest.get('selected_profile')}`",
        f"Conditional action: `{manifest['conditional_decision']['action']}`",
    ]
    if comparison:
        matched = comparison["matched_pressure64"]
        production = comparison["production_pressure64"]
        lines.extend(
            [
                "",
                "| Metric | Control 131072 | Balanced | Ratio |",
                "|---|---:|---:|---:|",
                f"| Matched p64 pair tok/s | {matched['control_pair_tokens_per_second']:.1f} | {matched['balanced_pair_tokens_per_second']:.1f} | {matched['balanced_over_control']:.2f}x |",
                f"| Production p64 pair tok/s | {production['control_pair_tokens_per_second']:.1f} | {production['balanced_pair_tokens_per_second']:.1f} | {production['balanced_over_control']:.2f}x |",
                "",
                f"Selected staged status: `{manifest['profile_receipts'][manifest['selected_profile']]['status']}`",
                "The control remains immutable. Ratios compare memory/scheduler configurations, not hardware generations.",
            ]
        )
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(args: argparse.Namespace) -> int:
    config = load_yaml(args.config)
    root = root_for(config)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and not args.summarize_only:
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("status") in {"passed", "degraded"}:
            raise RuntimeError(f"refusing to overwrite completed run: {manifest_path}")
        attempt = root / "attempts" / f"failed_{int(time.time())}"
        attempt.mkdir(parents=True, exist_ok=False)
        manifest_path.replace(attempt / "manifest.json")
        if (root / "report.md").exists():
            (root / "report.md").replace(attempt / "report.md")

    identity = config["identity"]
    control_path = Path(config["paths"]["immutable_control_manifest"])
    actual = {
        "checkpoint_sha256": sha256(
            Path(config["paths"]["project_root"])
            / "runs/sft_ordered_20260711_sft50k_v1/layer10_whole_layer_shs/checkpoints/step_00003916/trainable_state.pt"
        ),
        "export_model_sha256": sha256(
            Path(config["paths"]["project_root"])
            / "runs/rtx5090_pair_bringup_20260712_v1/gate_b/fullmodel_parity/deployment_export/model.safetensors"
        ),
        "immutable_control_manifest_sha256": sha256(control_path),
    }
    mismatches = {key: [identity[key], value] for key, value in actual.items() if identity[key] != value}
    if mismatches:
        raise RuntimeError(f"identity mismatch: {mismatches}")

    prereg = {
        "run": config["run"],
        "identity": {
            **identity,
            **actual,
            "deployed_source_hashes": {
                "balanced_runner": sha256(Path(__file__).resolve()),
                "balanced_config": sha256(args.config.resolve()),
                "staged_runner": sha256(Path(config["paths"]["staged_runner"])),
                "staged_template": sha256(Path(config["paths"]["staged_template"])),
            },
        },
        "profiles": config["profiles"],
        "conditional_trigger": config["conditional_trigger"],
        "claims": config["claims"],
        "timestamp_utc": utc_now(),
        "status": "preregistered",
    }
    atomic_write_json(root / "preregistered_manifest.json", prereg)
    for name in config["profiles"]:
        generated = build_profile_config(config, name, root)
        write_yaml(root / "generated_configs" / f"{config['profiles'][name]['label']}.yaml", generated)
    if args.prepare_only:
        print(json.dumps(prereg, indent=2, sort_keys=True))
        return 0

    primary_root = root / "profiles" / config["profiles"]["primary"]["label"]
    if args.summarize_only:
        primary = profile_receipt(primary_root)
    else:
        _, primary = run_profile(args, config, "primary", root)
    decision = conditional_decision(primary, config["conditional_trigger"])
    receipts = {"primary": primary}
    selected_name = "primary"
    if decision["action"] == "run_conditional":
        conditional_root = root / "profiles" / config["profiles"]["conditional"]["label"]
        if args.summarize_only:
            conditional = profile_receipt(conditional_root)
        else:
            _, conditional = run_profile(args, config, "conditional", root)
        receipts["conditional"] = conditional
        selected_name = "conditional"

    selected = receipts[selected_name]
    selected_manifest_path = Path(selected["manifest_path"])
    selected_manifest = json.loads(selected_manifest_path.read_text(encoding="utf-8")) if selected_manifest_path.exists() else None
    control = json.loads(control_path.read_text(encoding="utf-8"))
    comparison = None
    selected_root = selected_manifest_path.parent
    if selected_manifest is not None and selected.get("status") in {"passed", "degraded"}:
        comparison = build_comparison(control, selected_manifest, selected, control_path.parent, selected_root)
    if comparison is None or decision["action"] == "fail_stop":
        status = "failed"
    else:
        status = "passed" if selected.get("status") == "passed" else "degraded"
    manifest = {
        **prereg,
        "status": status,
        "completed_at_utc": utc_now(),
        "conditional_decision": decision,
        "profile_receipts": receipts,
        "selected_profile": selected_name,
        "comparison": comparison,
        "production_grpo_authorized": False,
    }
    atomic_write_json(manifest_path, manifest)
    write_report(root, manifest)
    print((root / "report.md").read_text(encoding="utf-8"), flush=True)
    return 0 if status == "passed" else 5


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--config", required=True, type=Path)
    result.add_argument("--prepare-only", action="store_true")
    result.add_argument("--summarize-only", action="store_true")
    return result


if __name__ == "__main__":
    raise SystemExit(main(parser().parse_args()))
