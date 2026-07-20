#!/usr/bin/env python3
"""Run a staged two-replica SHS production-length vLLM throughput gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float | int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    return ordered[round((len(ordered) - 1) * fraction)]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def output_root(config: dict[str, Any]) -> Path:
    root = Path(config["paths"]["project_root"])
    return root / config["run"]["output_root"]


def query_gpu_memory(physical_index: int) -> dict[str, int]:
    command = [
        "nvidia-smi",
        f"--id={physical_index}",
        "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    fields = subprocess.check_output(command, text=True).strip().split(",")
    used, total, utilization, temperature = (int(field.strip()) for field in fields)
    return {
        "memory_used_mib": used,
        "memory_total_mib": total,
        "utilization_percent": utilization,
        "temperature_c": temperature,
    }


def metric_value(metrics: Any, name: str) -> float | None:
    value = getattr(metrics, name, None)
    return None if value is None else float(value)


def select_prompts(config: dict[str, Any], destination: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    existing = json.loads(destination.read_text(encoding="utf-8")) if destination.exists() else None
    if existing is not None:
        return existing

    table = pq.read_table(config["paths"]["train_parquet"], columns=["prompt", "extra_info"])
    tokenizer = AutoTokenizer.from_pretrained(config["paths"]["model_export"], trust_remote_code=True)
    indices = list(range(table.num_rows))
    random.Random(int(config["run"]["seed"])).shuffle(indices)
    required = len(config["engine"]["gpu_indices"]) * int(config["workload"]["prompts_per_gpu"])
    selected: list[dict[str, Any]] = []
    for row_index in indices:
        row = table.slice(row_index, 1).to_pylist()[0]
        messages = row["prompt"]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        token_count = len(tokenizer(formatted, add_special_tokens=False).input_ids)
        if token_count > int(config["workload"]["max_prompt_tokens"]):
            continue
        extra = row.get("extra_info") or {}
        slot = len(selected)
        selected.append(
            {
                "slot": slot,
                "gpu_index": slot % len(config["engine"]["gpu_indices"]),
                "source_row_index": row_index,
                "source_problem_sha256": extra.get("problem_sha256"),
                "prompt_id": f"numina_train_{row_index:05d}",
                "formatted_prompt": formatted,
                "prompt_sha256": hashlib.sha256(formatted.encode("utf-8")).hexdigest(),
                "prompt_tokens": token_count,
            }
        )
        if len(selected) == required:
            break
    if len(selected) != required:
        raise RuntimeError(f"needed {required} prompts but selected {len(selected)}")
    atomic_write_json(destination, selected)
    return selected


def expanded_requests(
    prompts: list[dict[str, Any]], gpu_index: int, pressure: int, group_size: int, seed: int
) -> list[dict[str, Any]]:
    local = [row for row in prompts if row["gpu_index"] == gpu_index]
    unique_needed = pressure // group_size
    if pressure % group_size or unique_needed > len(local):
        raise ValueError(f"pressure {pressure} is incompatible with group size {group_size}")
    requests = []
    for prompt in local[:unique_needed]:
        for sample_index in range(group_size):
            requests.append(
                {
                    **prompt,
                    "sample_index": sample_index,
                    "request_id": f"{prompt['prompt_id']}:sample{sample_index}",
                    "seed": seed + int(prompt["slot"]) * group_size + sample_index,
                }
            )
    return requests


def cell_is_complete(path: Path, cell: str, pressure: int, gpu_index: int) -> bool:
    if not path.exists():
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return (
        row.get("status") == "passed"
        and row.get("cell") == cell
        and row.get("pressure") == pressure
        and row.get("gpu_index") == gpu_index
        and row.get("response_count") == pressure
    )


def run_cell(
    llm: Any,
    config: dict[str, Any],
    prompt_manifest: list[dict[str, Any]],
    gpu_index: int,
    cell: str,
    pressure: int,
    path: Path,
) -> dict[str, Any]:
    if cell_is_complete(path, cell, pressure, gpu_index):
        result = json.loads(path.read_text(encoding="utf-8"))
        print(f"RESUME_SKIP gpu={gpu_index} cell={cell} pressure={pressure}", flush=True)
        return result

    from vllm import SamplingParams

    workload = config["workload"]
    decoding = workload["cells"][cell]
    requests = expanded_requests(
        prompt_manifest,
        gpu_index,
        pressure,
        int(workload["group_size"]),
        int(config["run"]["seed"]),
    )
    params = [
        SamplingParams(
            temperature=float(decoding["temperature"]),
            top_p=float(decoding["top_p"]),
            min_tokens=int(decoding["min_tokens"]),
            max_tokens=int(decoding["max_tokens"]),
            seed=int(row["seed"]),
        )
        for row in requests
    ]
    before = query_gpu_memory(gpu_index)
    started_epoch = time.time()
    started = time.perf_counter()
    print(f"CELL_START gpu={gpu_index} cell={cell} pressure={pressure} at={utc_now()}", flush=True)
    try:
        outputs = llm.generate([row["formatted_prompt"] for row in requests], params, use_tqdm=False)
        error = None
    except Exception as exc:  # The receipt must survive a vLLM OOM or engine failure.
        outputs = []
        error = f"{type(exc).__name__}: {exc}"
    wall_seconds = time.perf_counter() - started
    ended_epoch = time.time()
    after = query_gpu_memory(gpu_index)
    rows = []
    output_lengths: list[int] = []
    latencies: list[float] = []
    scheduler_times: list[float] = []
    cap_hits = 0
    prompt_tokens = 0
    generated_tokens = 0
    for request, output in zip(requests, outputs, strict=False):
        candidate = output.outputs[0]
        metrics = output.metrics
        arrival = metric_value(metrics, "arrival_time")
        finished = metric_value(metrics, "finished_time")
        latency = None if arrival is None or finished is None else finished - arrival
        scheduler_seconds = metric_value(metrics, "scheduler_time")
        if latency is not None:
            latencies.append(latency)
        if scheduler_seconds is not None:
            scheduler_times.append(scheduler_seconds)
        length = len(candidate.token_ids)
        output_lengths.append(length)
        generated_tokens += length
        prompt_tokens += len(output.prompt_token_ids)
        cap_hits += int(length >= int(decoding["max_tokens"]))
        rows.append(
            {
                "request_id": request["request_id"],
                "prompt_id": request["prompt_id"],
                "sample_index": request["sample_index"],
                "seed": request["seed"],
                "prompt_tokens": len(output.prompt_token_ids),
                "generated_tokens": length,
                "finish_reason": candidate.finish_reason,
                "stop_reason": candidate.stop_reason,
                "token_ids_sha256": hashlib.sha256(json.dumps(list(candidate.token_ids)).encode()).hexdigest(),
                "latency_seconds": latency,
                "scheduler_seconds": scheduler_seconds,
                "time_in_queue_seconds": metric_value(metrics, "time_in_queue"),
                "first_token_time": metric_value(metrics, "first_token_time"),
            }
        )
    status = "passed" if error is None and len(outputs) == pressure else "failed"
    oom = error is not None and "out of memory" in error.lower()
    result = {
        "run_id": config["run"]["id"],
        "status": "oom" if oom else status,
        "error": error,
        "gpu_index": gpu_index,
        "cell": cell,
        "pressure": pressure,
        "group_size": workload["group_size"],
        "decoding": decoding,
        "started_epoch": started_epoch,
        "ended_epoch": ended_epoch,
        "wall_seconds": wall_seconds,
        "response_count": len(outputs),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "generated_tokens_per_second": generated_tokens / wall_seconds if wall_seconds else None,
        "output_length_mean": statistics.fmean(output_lengths) if output_lengths else None,
        "output_length_p50": percentile(output_lengths, 0.50),
        "output_length_p90": percentile(output_lengths, 0.90),
        "request_latency_p50_seconds": percentile(latencies, 0.50),
        "request_latency_p90_seconds": percentile(latencies, 0.90),
        "scheduler_time_p50_seconds": percentile(scheduler_times, 0.50),
        "scheduler_metrics_available": len(scheduler_times) == len(outputs) and bool(outputs),
        "configured_sequence_occupancy": pressure / int(config["engine"]["max_num_seqs"]),
        "cap_hits": cap_hits,
        "cap_hit_fraction": cap_hits / len(outputs) if outputs else None,
        "gpu_before": before,
        "gpu_after": after,
        "requests": rows,
    }
    atomic_write_json(path, result)
    print(
        f"CELL_END gpu={gpu_index} cell={cell} pressure={pressure} status={result['status']} "
        f"wall={wall_seconds:.3f}s tokens={generated_tokens} tok_s={result['generated_tokens_per_second']}",
        flush=True,
    )
    return result


def wait_for_control(root: Path, next_pressure: int, timeout_seconds: int, poll_seconds: float) -> bool:
    allow = root / "control" / f"allow_pressure_{next_pressure}"
    stop = root / "control" / "stop"
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if allow.exists():
            return True
        if stop.exists():
            return False
        time.sleep(poll_seconds)
    raise TimeoutError(f"control timeout waiting for pressure {next_pressure}")


def worker(args: argparse.Namespace, config: dict[str, Any]) -> int:
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["SHS_INFERENCE_MUL_BACKEND"] = "reference"
    root = output_root(config)
    gpu_root = root / f"gpu{args.gpu_index}"
    receipt = gpu_root / "dispatch_receipts.jsonl"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.unlink(missing_ok=True)
    os.environ["SHS_DISPATCH_RECEIPT"] = str(receipt)
    from vllm import LLM, SamplingParams

    prompt_manifest = json.loads((root / "prompt_manifest.json").read_text(encoding="utf-8"))
    engine = config["engine"]
    load_started = time.perf_counter()
    llm = LLM(
        model=config["paths"]["model_export"],
        model_impl=engine["model_impl"],
        trust_remote_code=True,
        tensor_parallel_size=1,
        enforce_eager=bool(engine["enforce_eager"]),
        max_model_len=int(engine["max_model_len"]),
        max_num_seqs=int(engine["max_num_seqs"]),
        max_num_batched_tokens=int(engine["max_num_batched_tokens"]),
        gpu_memory_utilization=float(engine["gpu_memory_utilization"]),
        seed=int(config["run"]["seed"]),
        disable_log_stats=False,
        enable_prefix_caching=bool(engine["enable_prefix_caching"]),
        enable_chunked_prefill=bool(engine["enable_chunked_prefill"]),
    )
    engine_load_seconds = time.perf_counter() - load_started
    local_prompt = next(row for row in prompt_manifest if row["gpu_index"] == args.gpu_index)
    startup = {"engine_load_seconds": engine_load_seconds}
    for label in ("cold", "warm"):
        started = time.perf_counter()
        output = llm.generate(
            [local_prompt["formatted_prompt"]],
            [SamplingParams(temperature=0.0, max_tokens=8, seed=int(config["run"]["seed"]))],
            use_tqdm=False,
        )
        startup[f"{label}_probe_seconds"] = time.perf_counter() - started
        startup[f"{label}_probe_tokens"] = len(output[0].outputs[0].token_ids)
    atomic_write_json(gpu_root / "startup.json", startup)

    pressures = [int(value) for value in config["workload"]["pressures"]]
    exit_code = 0
    for position, pressure in enumerate(pressures):
        results = {}
        for cell in config["workload"]["cells"]:
            path = gpu_root / "cells" / f"{cell}_p{pressure}.json"
            results[cell] = run_cell(llm, config, prompt_manifest, args.gpu_index, cell, pressure, path)
            if results[cell]["status"] != "passed":
                break
        stage_status = "passed" if len(results) == len(config["workload"]["cells"]) and all(
            row["status"] == "passed" for row in results.values()
        ) else "failed"
        atomic_write_json(
            gpu_root / "stages" / f"pressure_{pressure}.json",
            {"gpu_index": args.gpu_index, "pressure": pressure, "status": stage_status, "cells": results},
        )
        if stage_status != "passed":
            exit_code = 2
            break
        if position + 1 < len(pressures):
            if not wait_for_control(
                root,
                pressures[position + 1],
                int(config["gates"]["stage_timeout_seconds"]),
                float(config["gates"]["control_poll_seconds"]),
            ):
                break

    dispatches = []
    if receipt.exists():
        dispatches = [json.loads(line) for line in receipt.read_text(encoding="utf-8").splitlines() if line]
    backends = [row.get("backend") for row in dispatches]
    required = int(config["gates"]["required_dispatches_per_gpu"])
    dispatch_passed = len(backends) == required and backends == ["reference"] * required
    summary = {
        "gpu_index": args.gpu_index,
        "status": "passed" if exit_code == 0 and dispatch_passed else "failed",
        "dispatch_backends": backends,
        "dispatch_passed": dispatch_passed,
        "startup": startup,
    }
    atomic_write_json(gpu_root / "worker_summary.json", summary)
    try:
        llm.llm_engine.shutdown()
    except Exception:
        pass
    return exit_code if exit_code else (0 if dispatch_passed else 3)


def gpu_sampler(path: Path, stop: threading.Event) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestamp_epoch", "gpu_index", "memory_used_mib", "memory_total_mib", "utilization_percent", "temperature_c"],
        )
        writer.writeheader()
        while not stop.is_set():
            for gpu_index in (0, 1):
                try:
                    writer.writerow({"timestamp_epoch": time.time(), "gpu_index": gpu_index, **query_gpu_memory(gpu_index)})
                except Exception as exc:
                    writer.writerow({"timestamp_epoch": time.time(), "gpu_index": gpu_index, "memory_used_mib": "", "memory_total_mib": "", "utilization_percent": "", "temperature_c": f"error:{exc}"})
            handle.flush()
            stop.wait(1.0)


def stage_is_healthy(root: Path, pressure: int, config: dict[str, Any]) -> tuple[bool, list[str]]:
    failures = []
    for gpu_index in config["engine"]["gpu_indices"]:
        path = root / f"gpu{gpu_index}" / "stages" / f"pressure_{pressure}.json"
        if not path.exists():
            failures.append(f"gpu{gpu_index}:missing_stage")
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("status") != "passed":
            failures.append(f"gpu{gpu_index}:stage_{row.get('status')}")
        memory = query_gpu_memory(int(gpu_index))
        if memory["memory_used_mib"] / memory["memory_total_mib"] >= float(config["gates"]["memory_stop_fraction"]):
            failures.append(f"gpu{gpu_index}:memory_stop_fraction")
    return not failures, failures


def premature_stage_failures(processes: list[subprocess.Popen], expected: list[Path]) -> list[str]:
    return [
        str(path)
        for process, path in zip(processes, expected, strict=True)
        if process.poll() is not None and not path.exists()
    ]


def wait_for_stage(root: Path, pressure: int, processes: list[subprocess.Popen], config: dict[str, Any]) -> None:
    deadline = time.monotonic() + int(config["gates"]["stage_timeout_seconds"])
    expected = [root / f"gpu{gpu}" / "stages" / f"pressure_{pressure}.json" for gpu in config["engine"]["gpu_indices"]]
    while time.monotonic() < deadline:
        if all(path.exists() for path in expected):
            return
        premature = premature_stage_failures(processes, expected)
        if premature:
            raise RuntimeError(f"worker exited before its own stage {pressure}: {premature}")
        time.sleep(float(config["gates"]["control_poll_seconds"]))
    raise TimeoutError(f"pair stage {pressure} exceeded timeout")


def load_samples(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append({key: float(value) for key, value in row.items() if value != ""})
            except ValueError:
                continue
    return rows


def load_prior_peaks(root: Path) -> dict[tuple[str, int], dict[str, float | None]]:
    peaks = {}
    for path in sorted((root / "attempts").glob("*/manifest.json")):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for row in manifest.get("aggregate", {}).get("cells", []):
            value = row.get("peak_memory_mib_by_gpu")
            if value and any(item is not None for item in value.values()):
                peaks[(row["cell"], int(row["pressure_per_gpu"]))] = value
    return peaks


def pair_concurrency(rows: list[dict[str, Any]]) -> dict[str, float | bool]:
    start_skew = max(row["started_epoch"] for row in rows) - min(row["started_epoch"] for row in rows)
    overlap = max(0.0, min(row["ended_epoch"] for row in rows) - max(row["started_epoch"] for row in rows))
    maximum_wall = max(row["wall_seconds"] for row in rows)
    return {
        "pair_concurrent": start_skew <= 5.0,
        "start_skew_seconds": start_skew,
        "concurrent_overlap_seconds": overlap,
        "concurrent_overlap_fraction": overlap / maximum_wall,
    }


def aggregate_results(
    root: Path,
    config: dict[str, Any],
    gpu_samples: list[dict[str, float]],
    prior_peaks: dict[tuple[str, int], dict[str, float | None]] | None = None,
) -> dict[str, Any]:
    aggregates = []
    failures = []
    for pressure in config["workload"]["pressures"]:
        for cell in config["workload"]["cells"]:
            rows = []
            for gpu_index in config["engine"]["gpu_indices"]:
                path = root / f"gpu{gpu_index}" / "cells" / f"{cell}_p{pressure}.json"
                if path.exists():
                    rows.append(json.loads(path.read_text(encoding="utf-8")))
            if len(rows) != 2 or any(row.get("status") != "passed" for row in rows):
                failures.append(f"{cell}:pressure{pressure}:incomplete")
                continue
            started = min(row["started_epoch"] for row in rows)
            ended = max(row["ended_epoch"] for row in rows)
            pair_wall = ended - started
            concurrency = pair_concurrency(rows)
            pair_concurrent = bool(concurrency["pair_concurrent"])
            lengths = [request["generated_tokens"] for row in rows for request in row["requests"]]
            latencies = [request["latency_seconds"] for row in rows for request in row["requests"] if request["latency_seconds"] is not None]
            peak_by_gpu = {}
            for gpu_index in config["engine"]["gpu_indices"]:
                interval = [
                    sample["memory_used_mib"]
                    for sample in gpu_samples
                    if sample.get("gpu_index") == float(gpu_index) and started <= sample.get("timestamp_epoch", 0) <= ended
                ]
                peak_by_gpu[str(gpu_index)] = max(interval) if interval else None
            if all(value is None for value in peak_by_gpu.values()) and prior_peaks:
                peak_by_gpu = prior_peaks.get((cell, int(pressure)), peak_by_gpu)
            generated = sum(row["generated_tokens"] for row in rows)
            aggregates.append(
                {
                    "cell": cell,
                    "pressure_per_gpu": pressure,
                    "pair_request_count": sum(row["response_count"] for row in rows),
                    "pair_wall_seconds": pair_wall,
                    "pair_generated_tokens": generated,
                    "pair_generated_tokens_per_second": generated / pair_wall if pair_concurrent else None,
                    "ideal_sum_of_independent_gpu_rates": sum(row["generated_tokens_per_second"] for row in rows),
                    **concurrency,
                    "per_gpu_generated_tokens_per_second": {str(row["gpu_index"]): row["generated_tokens_per_second"] for row in rows},
                    "mean_per_gpu_generated_tokens_per_second": statistics.fmean(row["generated_tokens_per_second"] for row in rows),
                    "output_length_mean": statistics.fmean(lengths),
                    "output_length_p50": percentile(lengths, 0.50),
                    "output_length_p90": percentile(lengths, 0.90),
                    "request_latency_p50_seconds": percentile(latencies, 0.50),
                    "request_latency_p90_seconds": percentile(latencies, 0.90),
                    "cap_hits": sum(row["cap_hits"] for row in rows),
                    "cap_hit_fraction": sum(row["cap_hits"] for row in rows) / len(lengths),
                    "peak_memory_mib_by_gpu": peak_by_gpu,
                    "scheduler_metrics_available_on_both_gpus": all(row["scheduler_metrics_available"] for row in rows),
                    "configured_sequence_occupancy": rows[0]["configured_sequence_occupancy"],
                }
            )
    return {"cells": aggregates, "failures": failures}


def build_decision(config: dict[str, Any], aggregate: dict[str, Any], workers: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(row["cell"], row["pressure_per_gpu"]): row for row in aggregate["cells"]}
    matched = by_key.get(("matched_800_1024", 64))
    production = by_key.get(("production_cap3072", 64))
    canary = all((cell, 16) in by_key for cell in config["workload"]["cells"])
    dispatch = len(workers) == 2 and all(row.get("dispatch_passed") for row in workers)
    target = float(config["gates"]["production_mean_target_tokens"])
    tolerance = float(config["gates"]["production_mean_relative_tolerance"])
    length_match = production is not None and abs(production["output_length_mean"] - target) / target <= tolerance
    pair_concurrent = bool(matched and production and matched["pair_concurrent"] and production["pair_concurrent"])
    decision_grade = bool(canary and dispatch and matched and production and length_match and pair_concurrent)
    economics = None
    if production:
        per_gpu = production["mean_per_gpu_generated_tokens_per_second"]
        measured_pair = production["pair_generated_tokens_per_second"]
        eight_gpu = None if measured_pair is None else measured_pair * 4
        fixed_four_replica_tokens = int(config["gates"]["c2_tokens_per_replica_step"]) * 4
        economics = {
            "measured_pair_generated_tokens_per_second": measured_pair,
            "ideal_eight_gpu_generated_tokens_per_second": eight_gpu,
            "fixed_four_replica_token_work": fixed_four_replica_tokens,
            "ideal_fixed_work_rollout_seconds_on_eight_gpus": None if eight_gpu is None else fixed_four_replica_tokens / eight_gpu,
            "one_512_response_replica_seconds": 512 * production["output_length_mean"] / per_gpu,
            "scope": "pure rollout only; excludes reward, log-probs, actor, sync, checkpoint, and eval",
        }
    matched_comparison = None
    if matched:
        anchor = float(config["gates"]["matched_rtx_pro6000_anchor_tokens_per_second_per_gpu"])
        matched_comparison = {
            "rtx_pro6000_anchor_tokens_per_second_per_gpu": anchor,
            "rtx5090_mean_tokens_per_second_per_gpu": matched["mean_per_gpu_generated_tokens_per_second"],
            "rtx5090_over_rtx_pro6000": matched["mean_per_gpu_generated_tokens_per_second"] / anchor,
        }
    return {
        "status": "passed" if decision_grade else ("degraded" if canary else "failed"),
        "canary_passed": canary,
        "dispatch_passed": dispatch,
        "pressure64_complete": matched is not None and production is not None,
        "pressure64_pair_concurrent": pair_concurrent,
        "production_length_distribution_within_tolerance": length_match,
        "decision_grade": decision_grade,
        "matched_comparison": matched_comparison,
        "economic_extrapolation": economics,
        "production_grpo_authorized": False,
    }


def write_report(root: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# SHS 2xRTX 5090 Production-Length Throughput Gate",
        "",
        f"Status: **{manifest['decision']['status']}**",
        "",
        "| Cell | Pressure/GPU | Pair tok/s | Mean tok/s/GPU | Mean length | P90 length | Cap hit | Peak MiB GPU0/GPU1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in manifest["aggregate"]["cells"]:
        peak = row["peak_memory_mib_by_gpu"]
        pair_speed = "n/a" if row["pair_generated_tokens_per_second"] is None else f"{row['pair_generated_tokens_per_second']:.1f}"
        lines.append(
            f"| {row['cell']} | {row['pressure_per_gpu']} | {pair_speed} | "
            f"{row['mean_per_gpu_generated_tokens_per_second']:.1f} | {row['output_length_mean']:.1f} | "
            f"{row['output_length_p90']:.0f} | {row['cap_hit_fraction']:.2%} | {peak['0']}/{peak['1']} |"
        )
    lines.extend(
        [
            "",
            "The production decision remains false for GRPO. This gate measures rollout throughput only.",
            "Scheduler-native request timing is reported only when vLLM exposes it; configured sequence occupancy is always retained.",
        ]
    )
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def orchestrate(args: argparse.Namespace, config: dict[str, Any]) -> int:
    root = output_root(config)
    root.mkdir(parents=True, exist_ok=True)
    prior_manifest = root / "manifest.json"
    if prior_manifest.exists():
        prior = json.loads(prior_manifest.read_text(encoding="utf-8"))
        if prior.get("status") in {"passed", "degraded"} and not args.summarize_only:
            raise RuntimeError(f"refusing to overwrite completed run: {prior_manifest}")
        attempt_kind = "resummary" if args.summarize_only else "failed"
        attempt_root = root / "attempts" / f"{attempt_kind}_{int(time.time())}"
        attempt_root.mkdir(parents=True, exist_ok=False)
        prior_manifest.replace(attempt_root / "manifest.json")
        prior_report = root / "report.md"
        if prior_report.exists():
            prior_report.replace(attempt_root / "report.md")
    control = root / "control"
    control.mkdir(exist_ok=True)
    for stale in control.iterdir():
        stale.unlink()

    identity = config["identity"]
    actual_hashes = {
        "checkpoint_sha256": sha256(Path(config["paths"]["checkpoint"])),
        "export_model_sha256": sha256(Path(config["paths"]["model_export"]) / "model.safetensors"),
        "export_config_sha256": sha256(Path(config["paths"]["model_export"]) / "config.json"),
    }
    mismatches = {key: [identity[key], value] for key, value in actual_hashes.items() if identity[key] != value}
    if mismatches:
        raise RuntimeError(f"identity mismatch: {mismatches}")
    source_files = {
        "runner": Path(__file__).resolve(),
        "config": args.config.resolve(),
        "shs_hf_model": Path(config["paths"]["project_root"]) / "src/qwen_single_layer_rl/vllm/shs_hf_model.py",
        "shs_vllm_model": Path(config["paths"]["project_root"]) / "src/qwen_single_layer_rl/vllm/shs_vllm_model.py",
        "variant_module": Path(config["paths"]["project_root"]) / "src/qwen_single_layer_rl/model_surgery/qwen_swiglu_variant_modules.py",
    }
    source_hashes = {name: sha256(path) for name, path in source_files.items()}
    prompt_manifest = select_prompts(config, root / "prompt_manifest.json")
    prereg = {
        "run": config["run"],
        "timestamp_utc": utc_now(),
        "status": "preregistered",
        "identity": {**identity, **actual_hashes, "deployed_source_hashes": source_hashes},
        "engine": config["engine"],
        "workload": config["workload"],
        "gates": config["gates"],
        "prompt_manifest_sha256": sha256(root / "prompt_manifest.json"),
        "prompt_count": len(prompt_manifest),
        "claims": {"production_grpo_authorized": False, "checkpoint_mutation": False},
    }
    atomic_write_json(root / "preregistered_manifest.json", prereg)
    if args.prepare_only:
        print(json.dumps(prereg, indent=2, sort_keys=True))
        return 0

    if args.summarize_only:
        workers = []
        for gpu_index in config["engine"]["gpu_indices"]:
            path = root / f"gpu{gpu_index}" / "worker_summary.json"
            if path.exists():
                workers.append(json.loads(path.read_text(encoding="utf-8")))
        aggregate = aggregate_results(root, config, load_samples(root / "gpu_samples.csv"), load_prior_peaks(root))
        decision = build_decision(config, aggregate, workers)
        manifest = {
            **prereg,
            "status": decision["status"],
            "completed_at_utc": utc_now(),
            "worker_return_codes": [],
            "orchestration_error": None,
            "workers": workers,
            "aggregate": aggregate,
            "decision": decision,
            "summary_only": True,
        }
        atomic_write_json(root / "manifest.json", manifest)
        write_report(root, manifest)
        print((root / "report.md").read_text(encoding="utf-8"), flush=True)
        return 0 if decision["status"] in {"passed", "degraded"} else 4

    stop_sampler = threading.Event()
    sample_path = root / "gpu_samples.csv"
    sampler = threading.Thread(target=gpu_sampler, args=(sample_path, stop_sampler), daemon=True)
    sampler.start()
    processes: list[subprocess.Popen] = []
    logs = []
    return_codes: list[int] = []
    orchestration_error = None
    try:
        for gpu_index in config["engine"]["gpu_indices"]:
            gpu_root = root / f"gpu{gpu_index}"
            gpu_root.mkdir(parents=True, exist_ok=True)
            log_handle = (gpu_root / "worker.log").open("a", encoding="utf-8")
            logs.append(log_handle)
            environment = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(gpu_index),
                "PYTHONPATH": str(Path(config["paths"]["project_root"]) / "src"),
                "VLLM_USE_V1": "1",
            }
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--config", str(args.config), "--gpu-index", str(gpu_index)]
            processes.append(subprocess.Popen(command, env=environment, stdout=log_handle, stderr=subprocess.STDOUT))

        pressures = [int(value) for value in config["workload"]["pressures"]]
        gate_failures = []
        for position, pressure in enumerate(pressures):
            wait_for_stage(root, pressure, processes, config)
            healthy, failures = stage_is_healthy(root, pressure, config)
            print(f"PAIR_GATE pressure={pressure} healthy={healthy} failures={failures}", flush=True)
            if not healthy:
                gate_failures.extend(failures)
                (control / "stop").touch()
                break
            if position + 1 < len(pressures):
                (control / f"allow_pressure_{pressures[position + 1]}").touch()
        return_codes = [process.wait() for process in processes]
    except Exception as exc:
        orchestration_error = f"{type(exc).__name__}: {exc}"
        print(f"ORCHESTRATION_FAILURE {orchestration_error}", flush=True)
    finally:
        (control / "stop").touch()
        stop_sampler.set()
        sampler.join(timeout=5)
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        return_codes = [process.returncode for process in processes]
        for handle in logs:
            handle.close()

    workers = []
    for gpu_index in config["engine"]["gpu_indices"]:
        path = root / f"gpu{gpu_index}" / "worker_summary.json"
        if path.exists():
            workers.append(json.loads(path.read_text(encoding="utf-8")))
    aggregate = aggregate_results(root, config, load_samples(sample_path), load_prior_peaks(root))
    decision = build_decision(config, aggregate, workers)
    if orchestration_error is not None:
        decision["status"] = "failed"
        decision["decision_grade"] = False
    manifest = {
        **prereg,
        "status": decision["status"],
        "completed_at_utc": utc_now(),
        "worker_return_codes": return_codes,
        "orchestration_error": orchestration_error,
        "workers": workers,
        "aggregate": aggregate,
        "decision": decision,
    }
    atomic_write_json(root / "manifest.json", manifest)
    write_report(root, manifest)
    print((root / "report.md").read_text(encoding="utf-8"), flush=True)
    return 0 if decision["status"] in {"passed", "degraded"} else 4


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--config", required=True, type=Path)
    result.add_argument("--prepare-only", action="store_true")
    result.add_argument("--summarize-only", action="store_true")
    result.add_argument("--worker", action="store_true")
    result.add_argument("--gpu-index", type=int)
    return result


if __name__ == "__main__":
    parsed = parser().parse_args()
    loaded = load_config(parsed.config)
    if parsed.worker:
        if parsed.gpu_index is None:
            raise SystemExit("--worker requires --gpu-index")
        raise SystemExit(worker(parsed, loaded))
    raise SystemExit(orchestrate(parsed, loaded))
