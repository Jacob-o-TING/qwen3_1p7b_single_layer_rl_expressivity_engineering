from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import yaml

from qwen_single_layer_rl.eval.gpqa_freeform import (
    EXPECTED_ROWS,
    GENERATION_SHARDS,
    MATCHER_ROWS,
    MATCHER_SHARDS,
    build_matcher_ledger,
    canonical_json_sha256,
    extract_answer_tag,
    file_sha256,
    load_jsonl,
    merge_exact_shards,
    parse_matcher_decision,
    render_generation_prompt,
    render_matcher_prompt,
    select_shard,
    summarize_matches,
    write_json_atomic,
    write_jsonl_atomic,
)


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("free-form eval config must be a mapping")
    return value


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def render_chat(tokenizer: Any, prompt: str, *, enable_thinking: bool) -> str:
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    return prompt + "\nResponse:\n"


def request_seed(base_seed: int, identity: str) -> int:
    payload = f"{base_seed}:{identity}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def validate_completed_shard(output: Path, *, expected_rows: int, payload_name: str) -> None:
    payload = output / payload_name
    summary_path = output / "summary.json"
    if not payload.is_file() or not summary_path.is_file():
        raise RuntimeError(f"completion marker has incomplete payloads: {output}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("rows", -1)) != expected_rows:
        raise RuntimeError(f"completion marker has wrong row count: {output}")
    hash_key = "responses_sha256" if payload_name == "responses.jsonl" else "matches_sha256"
    if summary.get(hash_key) != file_sha256(payload):
        raise RuntimeError(f"completion marker has hash drift: {output}")


def load_engine(config: dict[str, Any], model: Path, *, plugin: str | None) -> tuple[Any, Any, float]:
    if plugin == "triglu":
        from qwen_single_layer_rl.vllm.triglu_plugin import register

        register()
    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    vllm_config = config["vllm"]
    started = time.perf_counter()
    engine = LLM(
        model=str(model),
        tokenizer=str(model),
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype=config["generation"]["dtype"],
        gpu_memory_utilization=float(vllm_config["gpu_memory_utilization"]),
        max_num_seqs=int(vllm_config["max_num_seqs"]),
        max_num_batched_tokens=int(vllm_config["max_num_batched_tokens"]),
        enforce_eager=bool(vllm_config["enforce_eager"]),
        model_impl="auto",
    )
    return engine, tokenizer, time.perf_counter() - started


def generate(args: argparse.Namespace, config: dict[str, Any]) -> None:
    from vllm import SamplingParams

    root = args.root.resolve()
    ledger = load_jsonl(resolve(root, config["dataset"]["ledger"]))
    tasks = select_shard(ledger, rank=args.rank, shard_count=GENERATION_SHARDS)
    if len(tasks) != EXPECTED_ROWS // GENERATION_SHARDS:
        raise RuntimeError(f"generation rank {args.rank} received {len(tasks)} rows")
    output = args.output.resolve()
    complete = output / "SHARD_COMPLETE"
    if complete.exists():
        validate_completed_shard(output, expected_rows=EXPECTED_ROWS // GENERATION_SHARDS, payload_name="responses.jsonl")
        print(f"GENERATION_SHARD_ALREADY_COMPLETE rank={args.rank}")
        return
    output.mkdir(parents=True, exist_ok=True)
    dispatch_receipt = output / "triglu_dispatch_receipts.jsonl"
    if args.plugin == "triglu":
        dispatch_receipt.unlink(missing_ok=True)
        os.environ["TRIGLU_DISPATCH_RECEIPT"] = str(dispatch_receipt)
    engine, tokenizer, load_seconds = load_engine(config, args.model.resolve(), plugin=args.plugin)
    prompts = [render_chat(tokenizer, render_generation_prompt(row["question"]), enable_thinking=True) for row in tasks]
    params = [
        SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=int(config["generation"]["max_tokens"]),
            seed=request_seed(int(config["seed"]), row["question_id"]),
        )
        for row in tasks
    ]
    started = time.perf_counter()
    results = engine.generate(prompts, params, use_tqdm=False)
    generation_seconds = time.perf_counter() - started
    if len(results) != len(tasks):
        raise RuntimeError(f"vLLM returned {len(results)} outputs for {len(tasks)} prompts")
    if args.plugin == "triglu":
        from qwen_single_layer_rl.vllm.custom_ffn_contract import validate_dispatch_receipts

        validate_dispatch_receipts(
            dispatch_receipt,
            variant="qwen_swiglu_triglu_side",
            backend="reference_pytorch_cublas",
            expected_count=1,
        )
    rows: list[dict[str, Any]] = []
    total_tokens = 0
    for task, prompt, result in zip(tasks, prompts, results, strict=True):
        if len(result.outputs) != 1:
            raise RuntimeError("free-form evaluator requires exactly one completion")
        completion = result.outputs[0]
        generated_tokens = len(completion.token_ids)
        total_tokens += generated_tokens
        raw = completion.text
        row = {
            **task,
            "variant": args.variant,
            "global_step": args.global_step,
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "raw_response": raw,
            "raw_response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "parsed_answer": extract_answer_tag(raw),
            "generated_tokens": generated_tokens,
            "finish_reason": str(completion.finish_reason),
            "cap_hit": generated_tokens >= int(config["generation"]["max_tokens"]),
            "request_seed": request_seed(int(config["seed"]), task["question_id"]),
        }
        row["generation_receipt_sha256"] = canonical_json_sha256(row)
        rows.append(row)
    receipts = output / "responses.jsonl"
    write_jsonl_atomic(receipts, rows)
    summary = {
        "status": "GENERATION_SHARD_COMPLETE",
        "variant": args.variant,
        "global_step": args.global_step,
        "rank": args.rank,
        "rows": len(rows),
        "generated_tokens": total_tokens,
        "engine_load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "tokens_per_second": total_tokens / generation_seconds if generation_seconds else None,
        "cap_hits": sum(int(row["cap_hit"]) for row in rows),
        "missing_answer_tags": sum(row["parsed_answer"] is None for row in rows),
        "responses_sha256": file_sha256(receipts),
        "dispatch_receipt_sha256": file_sha256(dispatch_receipt) if dispatch_receipt.exists() else None,
    }
    write_json_atomic(output / "summary.json", summary)
    complete.write_text("complete\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def merge_generation(args: argparse.Namespace, config: dict[str, Any]) -> None:
    root = args.root.resolve()
    ledger = load_jsonl(resolve(root, config["dataset"]["ledger"]))
    for rank in range(6):
        validate_completed_shard(
            args.cell / "shards" / f"rank_{rank}",
            expected_rows=EXPECTED_ROWS // GENERATION_SHARDS,
            payload_name="responses.jsonl",
        )
    shard_paths = [args.cell / "shards" / f"rank_{rank}" / "responses.jsonl" for rank in range(6)]
    rows = merge_exact_shards(
        shard_paths,
        expected_rows=EXPECTED_ROWS,
        expected_ids=[row["question_id"] for row in ledger],
        identity_key="question_id",
    )
    output = args.cell / "responses.jsonl"
    write_jsonl_atomic(output, rows)
    summary = {
        "status": "GENERATION_CELL_COMPLETE",
        "variant": args.variant,
        "global_step": args.global_step,
        "rows": len(rows),
        "ledger_sha256": file_sha256(resolve(root, config["dataset"]["ledger"])),
        "responses_sha256": file_sha256(output),
        "cap_hits": sum(int(row["cap_hit"]) for row in rows),
        "missing_answer_tags": sum(row["parsed_answer"] is None for row in rows),
        "generated_tokens": sum(int(row["generated_tokens"]) for row in rows),
    }
    write_json_atomic(args.cell / "summary.json", summary)
    (args.cell / "CELL_COMPLETE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def matcher_cells(config: dict[str, Any], root: Path, run_root: Path) -> list[tuple[str, int, Path]]:
    return [
        (cell["variant"], int(cell["global_step"]), run_root / "generation" / cell["label"] / "responses.jsonl")
        for cell in config["cells"]
    ]


def prepare_matcher_ledger(args: argparse.Namespace, config: dict[str, Any]) -> None:
    cells = matcher_cells(config, args.root.resolve(), args.run_root.resolve())
    for _, _, path in cells:
        summary = json.loads((path.parent / "summary.json").read_text(encoding="utf-8"))
        if summary.get("responses_sha256") != file_sha256(path):
            raise RuntimeError(f"generation cell hash drift before matching: {path}")
    rows = build_matcher_ledger(cells)
    output = args.run_root / "matching" / "matcher_ledger.jsonl"
    write_jsonl_atomic(output, rows)
    manifest = {
        "status": "MATCHER_LEDGER_READY",
        "rows": len(rows),
        "rows_per_rank": [sum(int(row["matcher_rank"]) == rank for row in rows) for rank in range(6)],
        "candidate_ids_sha256": canonical_json_sha256([row["candidate_id"] for row in rows]),
        "ledger_sha256": file_sha256(output),
    }
    write_json_atomic(args.run_root / "matching" / "matcher_ledger_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def match(args: argparse.Namespace, config: dict[str, Any]) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    ledger_path = args.run_root / "matching" / "matcher_ledger.jsonl"
    tasks = select_shard(load_jsonl(ledger_path), rank=args.rank, shard_count=MATCHER_SHARDS)
    if len(tasks) != MATCHER_ROWS // MATCHER_SHARDS:
        raise RuntimeError(f"matcher rank {args.rank} received {len(tasks)} rows")
    output = args.output.resolve()
    complete = output / "SHARD_COMPLETE"
    if complete.exists():
        validate_completed_shard(output, expected_rows=MATCHER_ROWS // MATCHER_SHARDS, payload_name="matches.jsonl")
        print(f"MATCHER_SHARD_ALREADY_COMPLETE rank={args.rank}")
        return
    output.mkdir(parents=True, exist_ok=True)
    matcher_model = resolve(args.root.resolve(), config["matcher"]["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(matcher_model, trust_remote_code=True)
    prompts = [
        render_chat(
            tokenizer,
            render_matcher_prompt(
                question=row["question"],
                reference_answer=row["reference_answer"],
                response=row["candidate_response"],
            ),
            enable_thinking=False,
        )
        for row in tasks
    ]
    vllm_config = config["vllm"]
    load_started = time.perf_counter()
    engine = LLM(
        model=str(matcher_model),
        tokenizer=str(matcher_model),
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype=config["matcher"]["dtype"],
        gpu_memory_utilization=float(vllm_config["gpu_memory_utilization"]),
        max_num_seqs=int(vllm_config["max_num_seqs"]),
        max_num_batched_tokens=int(vllm_config["max_num_batched_tokens"]),
        enforce_eager=bool(vllm_config["enforce_eager"]),
    )
    load_seconds = time.perf_counter() - load_started
    params = [
        SamplingParams(
            temperature=0.0,
            max_tokens=int(config["matcher"]["max_tokens"]),
            seed=request_seed(int(config["seed"]), row["candidate_id"]),
        )
        for row in tasks
    ]
    started = time.perf_counter()
    results = engine.generate(prompts, params, use_tqdm=False)
    match_seconds = time.perf_counter() - started
    if len(results) != len(tasks):
        raise RuntimeError("matcher output count differs from matcher input count")
    rows = []
    for task, prompt, result in zip(tasks, prompts, results, strict=True):
        completion = result.outputs[0]
        raw = completion.text
        rows.append(
            {
                **task,
                "matcher_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "matcher_raw_response": raw,
                "matcher_decision": parse_matcher_decision(raw),
                "matcher_generated_tokens": len(completion.token_ids),
                "matcher_finish_reason": str(completion.finish_reason),
            }
        )
    receipts = output / "matches.jsonl"
    write_jsonl_atomic(receipts, rows)
    summary = {
        "status": "MATCHER_SHARD_COMPLETE",
        "rank": args.rank,
        "rows": len(rows),
        "matcher_failures": sum(row["matcher_decision"] is None for row in rows),
        "engine_load_seconds": load_seconds,
        "match_seconds": match_seconds,
        "matches_sha256": file_sha256(receipts),
    }
    write_json_atomic(output / "summary.json", summary)
    complete.write_text("complete\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def merge_matcher(args: argparse.Namespace, config: dict[str, Any]) -> None:
    del config
    ledger = load_jsonl(args.run_root / "matching" / "matcher_ledger.jsonl")
    for rank in range(6):
        validate_completed_shard(
            args.run_root / "matching" / "shards" / f"rank_{rank}",
            expected_rows=MATCHER_ROWS // MATCHER_SHARDS,
            payload_name="matches.jsonl",
        )
    rows = merge_exact_shards(
        [args.run_root / "matching" / "shards" / f"rank_{rank}" / "matches.jsonl" for rank in range(6)],
        expected_rows=MATCHER_ROWS,
        expected_ids=[row["candidate_id"] for row in ledger],
        identity_key="candidate_id",
    )
    output = args.run_root / "matching" / "matches.jsonl"
    write_jsonl_atomic(output, rows)
    summary = summarize_matches(rows)
    summary.update({"status": "MATCHER_MERGE_COMPLETE", "matches_sha256": file_sha256(output)})
    write_json_atomic(args.run_root / "matching" / "summary.json", summary)
    (args.run_root / "matching" / "MATCHING_COMPLETE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def summarize(args: argparse.Namespace, config: dict[str, Any]) -> None:
    rows = load_jsonl(args.run_root / "matching" / "matches.jsonl")
    summary = summarize_matches(rows)
    disagreements = []
    grouped = {(row["global_step"], row["variant"], row["question_id"]): row for row in rows}
    for step in config["global_steps"]:
        for question_id in sorted({row["question_id"] for row in rows}):
            left = grouped.get((step, "triglu", question_id))
            right = grouped.get((step, "baseline", question_id))
            if left and right and left.get("matcher_decision") != right.get("matcher_decision"):
                disagreements.extend((left, right))
    failures = [row for row in rows if row.get("matcher_decision") is None]
    correct = sorted(
        (row for row in rows if row.get("matcher_decision") == 1),
        key=lambda row: canonical_json_sha256([config["seed"], row["candidate_id"]]),
    )[:12]
    incorrect = sorted(
        (row for row in rows if row.get("matcher_decision") == 0),
        key=lambda row: canonical_json_sha256([config["seed"], row["candidate_id"]]),
    )[:12]
    deterministic = correct + incorrect
    audit = {row["candidate_id"]: row for row in failures + disagreements + deterministic}
    audit_rows = [audit[key] for key in sorted(audit)]
    write_jsonl_atomic(args.run_root / "human_audit_queue.jsonl", audit_rows)
    summary.update(
        {
            "status": "FREEFORM_EVAL_COMPLETE",
            "run_id": config["run_id"],
            "audit_queue_rows": len(audit_rows),
            "same_step_disagreement_candidates": len(disagreements),
        }
    )
    write_json_atomic(args.run_root / "summary.json", summary)
    (args.run_root / "WAVE_COMPLETE").write_text("complete\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="action", required=True)
    generation = subparsers.add_parser("generate")
    generation.add_argument("--model", type=Path, required=True)
    generation.add_argument("--variant", choices=("triglu", "baseline"), required=True)
    generation.add_argument("--global-step", type=int, required=True)
    generation.add_argument("--rank", type=int, choices=range(6), required=True)
    generation.add_argument("--output", type=Path, required=True)
    generation.add_argument("--plugin", choices=("triglu",), default=None)
    merge_gen = subparsers.add_parser("merge-generation")
    merge_gen.add_argument("--variant", choices=("triglu", "baseline"), required=True)
    merge_gen.add_argument("--global-step", type=int, required=True)
    merge_gen.add_argument("--cell", type=Path, required=True)
    matcher_ledger = subparsers.add_parser("prepare-matcher-ledger")
    matcher_ledger.add_argument("--run-root", type=Path, required=True)
    matcher = subparsers.add_parser("match")
    matcher.add_argument("--run-root", type=Path, required=True)
    matcher.add_argument("--rank", type=int, choices=range(6), required=True)
    matcher.add_argument("--output", type=Path, required=True)
    merge_match = subparsers.add_parser("merge-matcher")
    merge_match.add_argument("--run-root", type=Path, required=True)
    summarize_parser = subparsers.add_parser("summarize")
    summarize_parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    actions = {
        "generate": generate,
        "merge-generation": merge_generation,
        "prepare-matcher-ledger": prepare_matcher_ledger,
        "match": match,
        "merge-matcher": merge_matcher,
        "summarize": summarize,
    }
    actions[args.action](args, config)


if __name__ == "__main__":
    main()
