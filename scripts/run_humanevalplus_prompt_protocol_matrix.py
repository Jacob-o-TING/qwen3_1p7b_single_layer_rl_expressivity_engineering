from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from qwen_single_layer_rl.eval.humanevalplus_parser import parse_humanevalplus_prediction
from qwen_single_layer_rl.eval.humanevalplus_prompt_matrix import (
    MATRIX_CELLS,
    classify_generation,
    decide_canary_escalation,
    load_source_tasks,
    render_prompt,
    request_seed,
    select_tasks,
    stable_json_sha256,
    summarize_cell,
)
from qwen_single_layer_rl.eval.local_code_sandbox import PrivilegeDroppedLocalSandboxBackend
from qwen_single_layer_rl.vllm.custom_ffn_contract import archive_incomplete_dispatch_receipt


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("matrix config must be a mapping")
    return config


def _config_cells(config: dict[str, Any]) -> tuple[str, ...]:
    cells = tuple(config.get("cells", MATRIX_CELLS))
    if not cells:
        raise ValueError("matrix config must select at least one cell")
    unknown = sorted(set(cells) - set(MATRIX_CELLS))
    if unknown:
        raise ValueError(f"unknown matrix cells: {unknown}")
    if len(set(cells)) != len(cells):
        raise ValueError("matrix config contains duplicate cells")
    return cells


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def prepare(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    config = _load_config(args.config)
    source_path = _resolve(root, config["source_prediction_jsonl"])
    output = _resolve(root, config["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    ledger_path = output / "task_ledger.json"
    manifest_path = output / "preflight_manifest.json"

    tasks = load_source_tasks(source_path)
    selected = select_tasks(tasks, seed=int(config["seed"]), count=int(config["sample_count"]))
    ledger = {
        "seed": int(config["seed"]),
        "sample_count": int(config["sample_count"]),
        "selection_rule": "sort_sha256(f'{seed}:{task_id}')_ascending_take_first_n",
        "tasks": selected,
    }
    ledger_sha256 = stable_json_sha256(ledger)
    expected_ledger_sha256 = config.get("expected_task_ledger_sha256")
    if expected_ledger_sha256 and ledger_sha256 != str(expected_ledger_sha256):
        raise RuntimeError(
            "task ledger changed: "
            f"expected {expected_ledger_sha256}, observed {ledger_sha256}"
        )
    manifest = {
        "run_id": config["run_id"],
        "model_label": str(config.get("model_label", "untuned_base")),
        "model_path": str(Path(config["model_path"]).resolve()),
        "source_prediction_jsonl": str(source_path.resolve()),
        "source_prediction_sha256": _sha256_file(source_path),
        "cells": list(_config_cells(config)),
        "seed": int(config["seed"]),
        "sample_count": int(config["sample_count"]),
        "max_tokens": int(config["max_tokens"]),
        "temperature": 0.0,
        "do_sample": False,
        "task_ledger_sha256": ledger_sha256,
        "expected_task_ledger_sha256": expected_ledger_sha256,
    }
    for path, payload in ((ledger_path, ledger), (manifest_path, manifest)):
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != payload:
                raise RuntimeError(f"refusing to overwrite mismatched preflight artifact: {path}")
        else:
            _write_json_atomic(path, payload)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def worker(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    root = args.root.resolve()
    output = _resolve(root, config["output_root"])
    ledger = json.loads((output / "task_ledger.json").read_text(encoding="utf-8"))
    tasks = [
        task
        for task in ledger["tasks"]
        if int(task["ledger_index"]) % args.shard_count == args.shard_index
    ]
    worker_id = f"{args.cell}.shard{args.shard_index:02d}-of-{args.shard_count:02d}"
    worker_root = output / "workers" / worker_id
    complete_path = worker_root / "WORKER_COMPLETE"
    receipts_path = worker_root / "receipts.jsonl"
    summary_path = worker_root / "summary.json"
    progress_path = worker_root / "progress.json"
    if complete_path.exists():
        print(f"WORKER_ALREADY_COMPLETE {worker_id}")
        return
    worker_root.mkdir(parents=True, exist_ok=True)
    for stale in (receipts_path.with_suffix(".jsonl.tmp"), summary_path.with_suffix(".json.tmp")):
        stale.unlink(missing_ok=True)

    dispatch_receipt_path = worker_root / "triglu_dispatch_receipts.jsonl"
    if config.get("vllm_plugin") == "triglu":
        archived_receipt = archive_incomplete_dispatch_receipt(dispatch_receipt_path)
        if archived_receipt is not None:
            print(f"ARCHIVED_INCOMPLETE_DISPATCH_RECEIPT {archived_receipt}")
        os.environ["TRIGLU_DISPATCH_RECEIPT"] = str(dispatch_receipt_path)

    model_path = Path(config["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompts = [render_prompt(tokenizer, args.cell, task) for task in tasks]
    prompt_receipts = []
    for task, prompt in zip(tasks, prompts, strict=True):
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_receipts.append(
            {
                "task_id": task["task_id"],
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt_token_ids_sha256": stable_json_sha256(token_ids),
                "prompt_chars": len(prompt),
                "prompt_tokens": len(token_ids),
            }
        )

    engine_started = time.perf_counter()
    engine = LLM(
        model=str(model_path),
        tokenizer=str(model_path),
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=float(config["vllm"]["gpu_memory_utilization"]),
        enforce_eager=bool(config["vllm"]["enforce_eager"]),
        model_impl="auto",
        max_num_seqs=int(config["vllm"]["max_num_seqs"]),
        max_num_batched_tokens=int(config["vllm"]["max_num_batched_tokens"]),
    )
    engine_load_seconds = time.perf_counter() - engine_started
    params = [
        SamplingParams(
            max_tokens=int(config["max_tokens"]),
            temperature=0.0,
            top_p=1.0,
            seed=request_seed(int(config["seed"]), str(task["task_id"])),
        )
        for task in tasks
    ]
    generation_started = time.perf_counter()
    outputs = engine.generate(prompts, params, use_tqdm=False)
    generation_seconds = time.perf_counter() - generation_started
    if len(outputs) != len(tasks):
        raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(tasks)} tasks")
    if config.get("vllm_plugin") == "triglu":
        from qwen_single_layer_rl.vllm.custom_ffn_contract import validate_dispatch_receipts

        validate_dispatch_receipts(
            dispatch_receipt_path,
            variant="qwen_swiglu_triglu_side",
            backend="reference_pytorch_cublas",
            expected_count=1,
        )

    backend = PrivilegeDroppedLocalSandboxBackend(
        root / "scripts" / "run_restricted_eval_python.py",
        Path(sys.executable),
        Path(config["sandbox_scratch_root"]) / worker_id,
    )
    backend.start()
    rows = []
    review_started = time.perf_counter()
    try:
        for index, (task, prompt_receipt, output) in enumerate(
            zip(tasks, prompt_receipts, outputs, strict=True), start=1
        ):
            if len(output.outputs) != 1:
                raise RuntimeError("matrix requires exactly one completion per request")
            completion = output.outputs[0]
            raw = completion.text
            parsed, parser_receipt = parse_humanevalplus_prediction(raw)
            finish_reason = str(completion.finish_reason)
            classification = classify_generation(
                raw=raw,
                parsed=parsed,
                canonical_prompt=str(task["prompt"]),
                finish_reason=finish_reason,
            )
            program = (
                f"{task['prompt']}{parsed}\n{task['test']}\n"
                f"check({task['entry_point']})"
            )
            execution = backend.execute(program, int(config["sandbox_timeout_seconds"]), "python")
            execution_exit_code = execution.get("exit_code")
            timed_out = execution.get("status") == "timeout" or execution_exit_code == -24
            row = {
                "cell": args.cell,
                "worker_id": worker_id,
                "ledger_index": task["ledger_index"],
                "selection_sha256": task["selection_sha256"],
                "source_index": task["source_index"],
                "task_id": task["task_id"],
                "entry_point": task["entry_point"],
                **prompt_receipt,
                "raw_output": raw,
                "raw_output_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                "generated_tokens": len(completion.token_ids),
                "finish_reason": finish_reason,
                "parser": parser_receipt.to_dict(),
                "classification": classification,
                "passed": execution.get("status") == "success",
                "execution_status": execution.get("status"),
                "execution_exit_code": execution_exit_code,
                "execution_timed_out": timed_out,
                "execution_stderr_tail": str(execution.get("stderr", ""))[-2000:],
            }
            rows.append(row)
            _write_json_atomic(
                progress_path,
                {"worker_id": worker_id, "reviewed": index, "total": len(tasks)},
            )
    finally:
        backend.stop()
    review_seconds = time.perf_counter() - review_started

    temporary_receipts = receipts_path.with_suffix(".jsonl.tmp")
    with temporary_receipts.open("x", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary_receipts, receipts_path)
    summary = {
        "worker_id": worker_id,
        "cell": args.cell,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "rows": len(rows),
        "task_ids_sha256": stable_json_sha256([row["task_id"] for row in rows]),
        "engine_load_seconds": engine_load_seconds,
        "generation_seconds": generation_seconds,
        "review_seconds": review_seconds,
        "generated_tokens": sum(int(row["generated_tokens"]) for row in rows),
        "receipts_sha256": _sha256_file(receipts_path),
        "vllm_plugin": config.get("vllm_plugin"),
        "dispatch_receipt_sha256": (
            _sha256_file(dispatch_receipt_path) if dispatch_receipt_path.exists() else None
        ),
    }
    _write_json_atomic(summary_path, summary)
    complete_path.touch()
    print(json.dumps(summary, indent=2, sort_keys=True))


def merge(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    config = _load_config(args.config)
    output = _resolve(root, config["output_root"])
    ledger = json.loads((output / "task_ledger.json").read_text(encoding="utf-8"))
    expected_ids = [str(task["task_id"]) for task in ledger["tasks"]]
    all_rows: dict[str, list[dict[str, Any]]] = {}
    worker_summaries = []
    cells = _config_cells(config)
    for cell in cells:
        rows = []
        for shard_index in range(args.shard_count):
            worker_id = f"{cell}.shard{shard_index:02d}-of-{args.shard_count:02d}"
            worker_root = output / "workers" / worker_id
            if not (worker_root / "WORKER_COMPLETE").exists():
                raise RuntimeError(f"incomplete worker: {worker_id}")
            worker_summaries.append(
                json.loads((worker_root / "summary.json").read_text(encoding="utf-8"))
            )
            with (worker_root / "receipts.jsonl").open(encoding="utf-8") as source:
                rows.extend(json.loads(line) for line in source if line.strip())
        actual_ids = [str(row["task_id"]) for row in sorted(rows, key=lambda row: row["ledger_index"])]
        if actual_ids != expected_ids:
            raise RuntimeError(f"identity coverage mismatch for {cell}")
        if len(set(actual_ids)) != len(actual_ids):
            raise RuntimeError(f"duplicate task IDs for {cell}")
        all_rows[cell] = sorted(rows, key=lambda row: row["ledger_index"])

    summaries = {cell: summarize_cell(rows) for cell, rows in all_rows.items()}
    decision = None
    full_validation = None
    if set(cells) == set(MATRIX_CELLS):
        decision = decide_canary_escalation(summaries)
    elif len(cells) == 1 and int(config["sample_count"]) == 164:
        cell_summary = summaries[cells[0]]
        full_validation = {
            "cell": cells[0],
            "minimum_passes": 49,
            "maximum_collapse_loops": 16,
            "observed_passes": int(cell_summary["passed"]),
            "observed_collapse_loops": int(cell_summary["collapse_loops"]),
            "materially_recovered": (
                int(cell_summary["passed"]) >= 49
                and int(cell_summary["collapse_loops"]) <= 16
            ),
        }
    summary = {
        "run_id": config["run_id"],
        "model_label": str(config.get("model_label", "untuned_base")),
        "status": "complete",
        "protocol": str(config.get("protocol", "humanevalplus_prompt_protocol_matrix")),
        "sample_count": int(config["sample_count"]),
        "task_ledger_sha256": stable_json_sha256(ledger),
        "cells": summaries,
        "decision": decision,
        "full_validation": full_validation,
        "worker_summaries": worker_summaries,
        "scientific_boundary": {
            "model_mutated": False,
            "checkpoint_mutated": False,
            "parser": "humanevalplus_parser_v2_fixed",
            "sandbox": "uid65534+no_new_privs+rlimits+seccomp_no_network_no_exec",
            "paper_protocol_parity_claimed": False,
        },
    }
    _write_json_atomic(output / "summary.json", summary)
    lines = [
        "HumanEval+ prompt protocol matrix: COMPLETE",
        f"tasks per cell: {config['sample_count']}",
    ]
    for cell in cells:
        cell_summary = summaries[cell]
        lines.append(
            f"{cell}: {cell_summary['passed']}/{cell_summary['rows']} "
            f"({100.0 * cell_summary['score']:.3f}%), "
            f"collapse_loops={cell_summary['collapse_loops']}, "
            f"syntax_valid={cell_summary['syntax_valid_completions']}, "
            f"cap_hits={cell_summary['cap_hits']}"
        )
    if decision is not None:
        lines.append(
            "decision: winner={winning_cell} pass_gain={pass_gain_tasks} "
            "loop_reduction={collapse_loop_reduction_tasks} escalate_full={escalate_to_full_untuned_base}".format(
                **decision
            )
        )
    if full_validation is not None:
        lines.append(
            "full_validation: passes={observed_passes}/164 collapse_loops={observed_collapse_loops}/164 "
            "materially_recovered={materially_recovered}".format(**full_validation)
        )
    (output / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / str(config.get("completion_marker", "CANARY_COMPLETE"))).touch()
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--cell", choices=MATRIX_CELLS, required=True)
    worker_parser.add_argument("--shard-index", type=int, required=True)
    worker_parser.add_argument("--shard-count", type=int, required=True)
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "worker":
        if not 0 <= args.shard_index < args.shard_count:
            raise SystemExit("--shard-index must be in [0, --shard-count)")
        worker(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
