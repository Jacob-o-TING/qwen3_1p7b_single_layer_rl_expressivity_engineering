#!/usr/bin/env python3
"""Validate a same-host RTX 5090 pair and emit durable bring-up receipts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


RUN_ID = "rtx5090_pair_bringup_20260712_v1"
EXPECTED_VERSIONS = {
    "torch": "2.8.0",
    "vllm": "0.10.2",
    "triton": "3.4.0",
    "transformers": "4.57.1",
    "evalscope": "1.8.1",
    "math-verify": "0.9.0",
    "verl": "0.6.1",
}
EXPECTED_CHECKPOINT_SHA256 = "ebb4cd92f6e890c17cf0e14a883557358dff927e901b69588f1867e6dd016712"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def command_output(*command: str) -> str:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    return (completed.stdout + completed.stderr).strip()


def nccl_worker(output_dir: Path) -> int:
    import torch.distributed as dist

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    value = torch.tensor([float(rank + 1)], device=device)
    dist.all_reduce(value)
    expected = dist.get_world_size() * (dist.get_world_size() + 1) / 2

    payload_bytes = 64 * 1024 * 1024
    tensor = torch.ones(payload_bytes // 4, dtype=torch.float32, device=device)
    for _ in range(3):
        dist.all_reduce(tensor)
        tensor.div_(dist.get_world_size())
    torch.cuda.synchronize()
    samples = []
    for _ in range(10):
        started = time.perf_counter()
        dist.all_reduce(tensor)
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - started)
        tensor.div_(dist.get_world_size())

    row = {
        "rank": rank,
        "local_rank": local_rank,
        "device": torch.cuda.get_device_name(local_rank),
        "all_reduce_value": value.item(),
        "all_reduce_expected": expected,
        "correct": value.item() == expected,
        "payload_bytes": payload_bytes,
        "seconds": samples,
        "median_seconds": sorted(samples)[len(samples) // 2],
    }
    write_json(output_dir / f"nccl_rank_{rank}.json", row)
    dist.barrier()
    dist.destroy_process_group()
    return 0 if row["correct"] else 2


def bf16_smoke(device_index: int) -> dict[str, Any]:
    torch.manual_seed(20260712)
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    x = torch.randn(512, 512, device=device, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.nn.Parameter(torch.randn(512, 512, device=device, dtype=torch.bfloat16) * 0.01)
    optimizer = torch.optim.AdamW([weight], lr=1.0e-4)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    loss = (x @ weight).float().square().mean()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)
    return {
        "device_index": device_index,
        "loss": float(loss.detach()),
        "gradient_finite": bool(torch.isfinite(weight.grad).all()),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }


def main(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = output_dir / "fresh_compiler_caches"
    inductor_cache = cache_root / "inductor"
    triton_cache = cache_root / "triton"
    inductor_cache.mkdir(parents=True, exist_ok=True)
    triton_cache.mkdir(parents=True, exist_ok=True)

    versions = {name: package_version(name) for name in EXPECTED_VERSIONS}
    version_matches = {
        name: versions[name] == expected or (name == "torch" and versions[name].startswith(expected + "+"))
        for name, expected in EXPECTED_VERSIONS.items()
    }
    checkpoint_hash = sha256(args.checkpoint)
    device_count = torch.cuda.device_count()
    environment = {
        "python": platform.python_version(),
        "versions": versions,
        "expected_versions": EXPECTED_VERSIONS,
        "version_matches": version_matches,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": device_count,
        "devices": [torch.cuda.get_device_name(index) for index in range(device_count)],
        "capabilities": [torch.cuda.get_device_capability(index) for index in range(device_count)],
        "arch_list": torch.cuda.get_arch_list(),
        "nccl_available": torch.distributed.is_nccl_available(),
        "nccl_version": torch.cuda.nccl.version() if torch.cuda.is_available() else None,
        "nvidia_smi": command_output("nvidia-smi"),
        "nvidia_smi_topology": command_output("nvidia-smi", "topo", "-m"),
        "disk": command_output("df", "-h", "/", str(args.source_root)),
        "memory": command_output("free", "-h"),
        "network": command_output("sh", "-lc", "ip -br addr 2>/dev/null || hostname -I || true"),
        "fresh_cache_paths": {"inductor": str(inductor_cache), "triton": str(triton_cache)},
    }
    write_json(output_dir / "environment.json", environment)

    bf16 = [bf16_smoke(index) for index in range(device_count)]
    nccl_command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={device_count}",
        str(Path(__file__).resolve()),
        "--nccl-worker",
        "--output-dir",
        str(output_dir),
    ]
    nccl_environment = {
        **os.environ,
        "NCCL_DEBUG": "INFO",
        "TORCHINDUCTOR_CACHE_DIR": str(inductor_cache),
        "TRITON_CACHE_DIR": str(triton_cache),
    }
    nccl_started = time.perf_counter()
    completed = subprocess.run(nccl_command, env=nccl_environment, capture_output=True, text=True)
    nccl_seconds = time.perf_counter() - nccl_started
    (output_dir / "nccl.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    nccl_rows = []
    for rank in range(device_count):
        path = output_dir / f"nccl_rank_{rank}.json"
        if path.exists():
            nccl_rows.append(json.loads(path.read_text(encoding="utf-8")))

    failures = []
    if device_count != 2:
        failures.append(f"expected_two_gpus_found_{device_count}")
    if any(name != "NVIDIA GeForce RTX 5090" for name in environment["devices"]):
        failures.append("unexpected_gpu_model")
    if any(capability != (12, 0) for capability in environment["capabilities"]):
        failures.append("unexpected_compute_capability")
    if "sm_120" not in environment["arch_list"]:
        failures.append("torch_missing_sm_120")
    if not all(version_matches.values()):
        failures.append("pinned_version_mismatch")
    if checkpoint_hash != EXPECTED_CHECKPOINT_SHA256:
        failures.append("checkpoint_hash_mismatch")
    if not all(row["gradient_finite"] for row in bf16):
        failures.append("bf16_gradient_nonfinite")
    if completed.returncode or len(nccl_rows) != device_count or not all(row["correct"] for row in nccl_rows):
        failures.append("nccl_pair_probe_failed")

    manifest = {
        "run_id": RUN_ID,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not failures else "failed",
        "source_root": str(args.source_root.resolve()),
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": checkpoint_hash,
            "expected_sha256": EXPECTED_CHECKPOINT_SHA256,
        },
        "environment": environment,
        "bf16_optimizer_smoke": bf16,
        "nccl": {
            "command": nccl_command,
            "returncode": completed.returncode,
            "wall_seconds": nccl_seconds,
            "ranks": nccl_rows,
        },
        "failures": failures,
    }
    write_json(output_dir / "manifest.json", manifest)
    lines = [
        f"# {RUN_ID}",
        "",
        f"Status: **{manifest['status']}**",
        "",
        f"GPUs: {device_count} x {', '.join(environment['devices'])}",
        f"Checkpoint SHA-256: `{checkpoint_hash}`",
        f"Pinned versions match: `{all(version_matches.values())}`",
        f"BF16 optimizer smokes pass: `{all(row['gradient_finite'] for row in bf16)}`",
        f"NCCL two-rank probe pass: `{not completed.returncode and len(nccl_rows) == device_count}`",
        "",
        "Failures:",
        *([f"- `{failure}`" for failure in failures] or ["- None"]),
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if not failures else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--nccl-worker", action="store_true")
    args = parser.parse_args()
    if not args.nccl_worker and args.checkpoint is None:
        parser.error("--checkpoint is required for the parent bring-up process")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.nccl_worker:
        raise SystemExit(nccl_worker(parsed.output_dir.resolve()))
    raise SystemExit(main(parsed))
