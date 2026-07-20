#!/usr/bin/env python3
"""Compare an updated SHS export with its initial 29-tensor trainable contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open


BLOCK_ID_SUFFIXES = (
    "mul_row_block_ids",
    "mul_col_block_ids",
    "add_row_block_ids",
    "add_col_block_ids",
)


def tensor_hash(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous().view(torch.uint8)
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-trainable-state", type=Path, required=True)
    parser.add_argument("--updated-safetensors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    initial = torch.load(args.initial_trainable_state, map_location="cpu", weights_only=False)
    if not isinstance(initial, dict):
        raise TypeError("initial trainable state must be a tensor mapping")
    initial_keys = sorted(initial)
    rows = []
    with safe_open(args.updated_safetensors, framework="pt", device="cpu") as handle:
        updated_keys = sorted(handle.keys())
        leaked_buffers = [key for key in updated_keys if key.endswith(BLOCK_ID_SUFFIXES)]
        missing = sorted(set(initial_keys) - set(updated_keys))
        for name in initial_keys:
            if name in missing:
                continue
            before = initial[name]
            after = handle.get_tensor(name)
            shape_equal = tuple(before.shape) == tuple(after.shape)
            dtype_equal = before.dtype == after.dtype
            equal_raw = shape_equal and dtype_equal and torch.equal(before, after)
            normalized_before = before.to(after.dtype) if shape_equal else before
            equal_after_cast = shape_equal and torch.equal(normalized_before, after)
            max_abs_delta = None
            if shape_equal:
                max_abs_delta = float((normalized_before.float() - after.float()).abs().max())
            rows.append(
                {
                    "name": name,
                    "shape": list(before.shape),
                    "dtype_before": str(before.dtype),
                    "dtype_after": str(after.dtype),
                    "shape_equal": shape_equal,
                    "dtype_equal": dtype_equal,
                    "equal_raw": equal_raw,
                    "equal_after_cast": equal_after_cast,
                    "hash_before": tensor_hash(before),
                    "hash_before_cast_to_updated_dtype": tensor_hash(normalized_before),
                    "hash_after": tensor_hash(after),
                    "max_abs_delta": max_abs_delta,
                }
            )

    raw_changed = [row["name"] for row in rows if not row["equal_raw"]]
    changed = [row["name"] for row in rows if not row["equal_after_cast"]]
    unchanged = [row["name"] for row in rows if row["equal_after_cast"]]
    failures = []
    if len(initial_keys) != 29:
        failures.append(f"expected_29_initial_trainable_tensors_found_{len(initial_keys)}")
    if missing:
        failures.append("updated_export_missing_trainable_tensors")
    if leaked_buffers:
        failures.append("deterministic_block_ids_leaked_into_export")
    if len(changed) != 29:
        failures.append(f"expected_29_changed_tensors_found_{len(changed)}")
    if any(not row["shape_equal"] for row in rows):
        failures.append("shape_contract_changed")

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not failures else "failed",
        "initial_trainable_state": str(args.initial_trainable_state.resolve()),
        "updated_safetensors": str(args.updated_safetensors.resolve()),
        "initial_tensor_count": len(initial_keys),
        "updated_weight_key_count": len(updated_keys),
        "raw_changed_tensor_count_including_dtype_casts": len(raw_changed),
        "changed_tensor_count": len(changed),
        "changed_tensor_names": changed,
        "unchanged_tensor_names": unchanged,
        "dtype_transition_tensor_names": [row["name"] for row in rows if not row["dtype_equal"]],
        "missing_tensor_names": missing,
        "leaked_deterministic_buffer_names": leaked_buffers,
        "rows": rows,
        "failures": failures,
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
