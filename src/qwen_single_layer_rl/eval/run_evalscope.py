from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from .evalscope_custom_model import register_evalscope_model
from .live_status import build_model_summary
from .paper_benchmarks import register_paper_benchmarks


def _evaluation_phases(
    *, amc_first: bool, amc_only: bool = False, include_amc_greedy: bool = True
) -> tuple[str, ...]:
    amc_phases = ("amc", "amc_greedy") if include_amc_greedy else ("amc",)
    if amc_only:
        return amc_phases
    return (*amc_phases, "main") if amc_first else ("main", *amc_phases)


def _validate_model_source(*, checkpoint_dir: Path | None, base_model_only: bool) -> None:
    if base_model_only and checkpoint_dir is not None:
        raise ValueError("--checkpoint-dir cannot be combined with --base-model-only")
    if not base_model_only and checkpoint_dir is None:
        raise ValueError("--checkpoint-dir is required unless --base-model-only is set")


def _generation_config(
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> dict[str, Any]:
    do_sample = temperature > 0.0
    config: dict[str, Any] = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "do_sample": do_sample,
        "seed": seed,
    }
    if do_sample:
        config["top_p"] = top_p
    return config


def _task_config(
    *,
    model: Any,
    datasets: list[str],
    work_dir: Path,
    repeats: int,
    limit: int | None,
    seed: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    eval_batch_size: int,
    local_code_sandbox: bool = False,
    use_cache: Path | None = None,
    dataset_args: dict[str, Any] | None = None,
) -> Any:
    from evalscope import TaskConfig

    kwargs: dict[str, Any] = {
        "model": model,
        "datasets": datasets,
        "repeats": repeats,
        "work_dir": str(work_dir),
        "seed": seed,
        "eval_batch_size": eval_batch_size,
        "generation_config": _generation_config(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        ),
    }
    if limit is not None:
        kwargs["limit"] = limit
    if use_cache is not None:
        kwargs["use_cache"] = str(use_cache)
    if dataset_args:
        kwargs["dataset_args"] = dataset_args
    if local_code_sandbox:
        kwargs["sandbox"] = {
            "enabled": True,
            "engine": "docker",
            "pool_size": eval_batch_size,
        }
    return TaskConfig(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--base-model-only", action="store_true")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument(
        "--benchmark-snapshot-dir",
        type=Path,
        default=Path("data/eval/qwen2p5_math_a45202bd16f1ec06f433442dc1152d0074773465"),
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--main-use-cache",
        type=Path,
        help="Resume the main benchmark phase from an EvalScope timestamp directory.",
    )
    parser.add_argument(
        "--amc-use-cache",
        type=Path,
        help="Resume the AMC Average@32 phase from an EvalScope timestamp directory.",
    )
    parser.add_argument(
        "--amc-greedy-use-cache",
        type=Path,
        help="Resume the separate AMC greedy pass@1 phase from an EvalScope timestamp directory.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument(
        "--dataset-subsets",
        nargs="+",
        help="Restrict the requested EvalScope dataset to explicit subsets.",
    )
    parser.add_argument("--dataset-shard-count", type=int)
    parser.add_argument("--dataset-shard-index", type=int)
    parser.add_argument("--humanevalplus-parser-v2", action="store_true")
    parser.add_argument("--amc-repeats", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--amc-temperature", type=float, default=1.0)
    parser.add_argument("--amc-top-p", type=float, default=1.0)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--backend", choices=("hf", "vllm"), default="hf")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument("--vllm-model-impl", choices=("auto", "transformers"), default="auto")
    parser.add_argument("--vllm-max-num-seqs", type=int)
    parser.add_argument("--vllm-max-num-batched-tokens", type=int)
    parser.add_argument("--microbatch-wait-seconds", type=float, default=0.010)
    parser.add_argument(
        "--shs-backend",
        choices=("reference", "triton", "triton_fp32"),
        help="Explicit SHS projection backend; requires --vllm-model-impl transformers.",
    )
    parser.add_argument("--amc-first", action="store_true")
    parser.add_argument("--amc-only", action="store_true")
    parser.add_argument("--skip-amc-greedy", action="store_true")
    parser.add_argument(
        "--local-code-sandbox",
        action="store_true",
        help="Use the project privilege-dropped/seccomp backend for code benchmark review.",
    )
    args = parser.parse_args()
    if args.eval_batch_size <= 0:
        raise SystemExit("--eval-batch-size must be positive")
    if args.amc_only and args.amc_repeats <= 0:
        raise SystemExit("--amc-only requires a positive --amc-repeats value")
    try:
        _validate_model_source(
            checkpoint_dir=args.checkpoint_dir,
            base_model_only=args.base_model_only,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.shs_backend and args.backend != "vllm":
        raise SystemExit("--shs-backend requires --backend vllm")
    if args.shs_backend and args.vllm_model_impl != "transformers":
        raise SystemExit("--shs-backend requires --vllm-model-impl transformers")
    shard_args = (args.dataset_shard_index, args.dataset_shard_count)
    if (shard_args[0] is None) != (shard_args[1] is None):
        raise SystemExit("--dataset-shard-index and --dataset-shard-count must be set together")
    if args.dataset_shard_count is not None:
        if args.dataset_shard_count <= 0:
            raise SystemExit("--dataset-shard-count must be positive")
        if not 0 <= args.dataset_shard_index < args.dataset_shard_count:
            raise SystemExit("--dataset-shard-index must be in [0, --dataset-shard-count)")
        if args.datasets is None or len(args.datasets) != 1:
            raise SystemExit("dataset sharding requires exactly one requested dataset")
        if args.datasets == ["live_code_bench"] and args.dataset_subsets != ["release_latest"]:
            raise SystemExit(
                "sharded live_code_bench requires --dataset-subsets release_latest"
            )
        if args.datasets != ["live_code_bench"] and args.dataset_subsets is not None:
            raise SystemExit("generic dataset sharding does not accept --dataset-subsets")
    if args.humanevalplus_parser_v2 and args.datasets != ["humaneval_plus"]:
        raise SystemExit("--humanevalplus-parser-v2 requires --datasets humaneval_plus")

    try:
        import evalscope
        from evalscope import run_task
    except ModuleNotFoundError as exc:
        raise SystemExit("EvalScope is missing from the pinned evaluation environment") from exc

    if args.local_code_sandbox:
        from .local_code_sandbox import install_evalscope_local_code_sandbox

        install_evalscope_local_code_sandbox(
            runner=(Path(__file__).resolve().parents[3] / "scripts" / "run_restricted_eval_python.py"),
            python=Path(sys.executable),
            scratch_root=Path("/tmp/qwen_ood_code_sandbox"),
        )
    if args.humanevalplus_parser_v2:
        from .humanevalplus_parser import install_evalscope_humanevalplus_parser

        install_evalscope_humanevalplus_parser()
    if args.dataset_shard_count is not None:
        if args.datasets == ["live_code_bench"]:
            from .live_code_bench_sharding import install_evalscope_live_code_bench_sharding

            install_evalscope_live_code_bench_sharding(
                shard_index=args.dataset_shard_index,
                shard_count=args.dataset_shard_count,
                receipt_path=args.work_dir / "dataset_shard_receipt.jsonl",
            )
        else:
            from .generic_evalscope_sharding import install_evalscope_generic_sharding

            install_evalscope_generic_sharding(
                dataset_name=args.datasets[0],
                shard_index=args.dataset_shard_index,
                shard_count=args.dataset_shard_count,
                receipt_path=args.work_dir / "dataset_shard_receipt.jsonl",
            )

    model_class = register_evalscope_model()
    math500_dataset, gsm8k_dataset, olympiad_dataset, amc_dataset = register_paper_benchmarks(
        args.benchmark_snapshot_dir.resolve()
    )
    model = model_class(
        model_name="qwen3-1p7b-single-layer-sft",
        config_path=str(args.config.resolve()),
        checkpoint_dir=str(args.checkpoint_dir.resolve()) if args.checkpoint_dir else None,
        model_path=str(args.model_path.resolve()) if args.model_path else None,
        device="cuda",
        base_model_only=args.base_model_only,
        backend=args.backend,
        receipt_jsonl=str((args.work_dir / "generation_receipts.jsonl").resolve()),
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        enforce_eager=args.vllm_enforce_eager,
        vllm_model_impl=args.vllm_model_impl,
        max_num_seqs=args.vllm_max_num_seqs,
        max_num_batched_tokens=args.vllm_max_num_batched_tokens,
        microbatch_wait_seconds=args.microbatch_wait_seconds,
        identity_namespace=",".join(args.datasets or ["paper_benchmarks"]),
        shs_backend=args.shs_backend,
        shs_dispatch_receipt=(
            str((args.work_dir / "shs_dispatch_receipts.jsonl").resolve())
            if args.shs_backend
            else None
        ),
    )
    args.work_dir.mkdir(parents=True, exist_ok=True)
    main_datasets = (
        []
        if args.amc_only
        else (args.datasets or [math500_dataset, gsm8k_dataset, olympiad_dataset])
    )
    main_dataset_args = (
        {dataset: {"subset_list": args.dataset_subsets} for dataset in main_datasets}
        if args.dataset_subsets
        else None
    )
    manifest = {
        "evalscope_version": getattr(evalscope, "__version__", "unknown"),
        "config": str(args.config.resolve()),
        "checkpoint_dir": str(args.checkpoint_dir.resolve()) if args.checkpoint_dir else None,
        "base_model_only": args.base_model_only,
        "model_path": str(args.model_path.resolve()) if args.model_path else None,
        "main_datasets": main_datasets,
        "main_dataset_args": main_dataset_args,
        "dataset_shard_count": args.dataset_shard_count,
        "dataset_shard_index": args.dataset_shard_index,
        "amc_dataset": amc_dataset,
        "benchmark_snapshot_dir": str(args.benchmark_snapshot_dir.resolve()),
        "amc_repeats": args.amc_repeats,
        "amc_only": args.amc_only,
        "eval_batch_size": args.eval_batch_size,
        "requested_backend": args.backend,
        "vllm_model_impl": args.vllm_model_impl,
        "vllm_enforce_eager": args.vllm_enforce_eager,
        "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
        "vllm_max_num_seqs": args.vllm_max_num_seqs,
        "vllm_max_num_batched_tokens": args.vllm_max_num_batched_tokens,
        "microbatch_wait_seconds": args.microbatch_wait_seconds,
        "requested_shs_backend": args.shs_backend,
        "local_code_sandbox": args.local_code_sandbox,
        "humanevalplus_parser_v2": args.humanevalplus_parser_v2,
        "main_use_cache": str(args.main_use_cache.resolve()) if args.main_use_cache else None,
        "amc_use_cache": str(args.amc_use_cache.resolve()) if args.amc_use_cache else None,
        "amc_greedy_use_cache": (
            str(args.amc_greedy_use_cache.resolve()) if args.amc_greedy_use_cache else None
        ),
        "main_generation": _generation_config(
            max_tokens=args.max_tokens,
            temperature=0.0,
            top_p=1.0,
            seed=args.seed,
        ),
        "amc_generation": _generation_config(
            max_tokens=args.max_tokens,
            temperature=args.amc_temperature,
            top_p=args.amc_top_p,
            seed=args.seed,
        ),
        "amc_greedy_generation": _generation_config(
            max_tokens=args.max_tokens,
            temperature=0.0,
            top_p=1.0,
            seed=args.seed,
        ),
        "seed": args.seed,
        "limit": args.limit,
    }
    (args.work_dir / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    for phase in _evaluation_phases(
        amc_first=args.amc_first,
        amc_only=args.amc_only,
        include_amc_greedy=not args.skip_amc_greedy,
    ):
        phase_started = time.perf_counter()
        if phase == "amc":
            if args.amc_repeats > 0:
                run_task(
                    _task_config(
                        model=model,
                        datasets=[manifest["amc_dataset"]],
                        work_dir=args.work_dir / "amc_average_at_32",
                        repeats=args.amc_repeats,
                        limit=args.limit,
                        seed=args.seed,
                        max_tokens=args.max_tokens,
                        temperature=args.amc_temperature,
                        top_p=args.amc_top_p,
                        eval_batch_size=args.eval_batch_size,
                        local_code_sandbox=args.local_code_sandbox,
                        use_cache=args.amc_use_cache.resolve() if args.amc_use_cache else None,
                    )
                )
            else:
                phase_receipt = {
                    "phase": phase,
                    "status": "skipped",
                    "reason": "amc_repeats_is_zero",
                    "requested_backend": args.backend,
                    "elapsed_seconds": time.perf_counter() - phase_started,
                }
                (args.work_dir / f"{phase}_completion_receipt.json").write_text(
                    json.dumps(phase_receipt, indent=2, sort_keys=True), encoding="utf-8"
                )
                continue
        elif phase == "amc_greedy":
            run_task(
                _task_config(
                    model=model,
                    datasets=[manifest["amc_dataset"]],
                    work_dir=args.work_dir / "amc_greedy",
                    repeats=1,
                    limit=args.limit,
                    seed=args.seed,
                    max_tokens=args.max_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    eval_batch_size=args.eval_batch_size,
                    local_code_sandbox=args.local_code_sandbox,
                    use_cache=(
                        args.amc_greedy_use_cache.resolve() if args.amc_greedy_use_cache else None
                    ),
                )
            )
        else:
            run_task(
                _task_config(
                    model=model,
                    datasets=manifest["main_datasets"],
                    work_dir=args.work_dir / "main",
                    repeats=1,
                    limit=args.limit,
                    seed=args.seed,
                    max_tokens=args.max_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    eval_batch_size=args.eval_batch_size,
                    local_code_sandbox=args.local_code_sandbox,
                    use_cache=args.main_use_cache.resolve() if args.main_use_cache else None,
                    dataset_args=main_dataset_args,
                )
            )
        phase_receipt = {
            "phase": phase,
            "status": "complete",
            "requested_backend": args.backend,
            "elapsed_seconds": time.perf_counter() - phase_started,
        }
        (args.work_dir / f"{phase}_completion_receipt.json").write_text(
            json.dumps(phase_receipt, indent=2, sort_keys=True), encoding="utf-8"
        )
    (args.work_dir / "model_summary.json").write_text(
        json.dumps(build_model_summary(args.work_dir), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
