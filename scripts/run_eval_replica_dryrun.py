from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

from qwen_single_layer_rl.eval.replica_sharding import (
    RequestIdentity,
    merge_rank_jsonl,
    shard_identities,
    topology_preflight,
    write_rank_receipt,
)


def _visible_gpu_ids() -> list[str]:
    import torch

    return [str(index) for index in range(torch.cuda.device_count())]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--logical-replicas", type=int, default=7)
    parser.add_argument("--topology-replicas", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260707)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    identities = [
        RequestIdentity(benchmark, item_id, sample_id)
        for benchmark, count, repeats in (
            ("paper_math500", 11, 1),
            ("paper_gsm8k", 13, 1),
            ("paper_olympiadbench", 7, 1),
            ("paper_amc23", 5, 4),
        )
        for item_id in range(count)
        for sample_id in range(repeats)
    ]
    preflight = topology_preflight(
        visible_gpu_ids=_visible_gpu_ids(),
        requested_replicas=args.topology_replicas,
        tensor_parallel_size=args.tensor_parallel_size,
        identities=identities,
        output_root=args.output_dir,
    )
    (args.output_dir / "topology_preflight.json").write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    shards = shard_identities(identities, args.logical_replicas)
    rank_paths: dict[int, Path] = {}
    randomizer = random.Random(args.seed)
    for rank, expected in shards.items():
        rank_dir = args.output_dir / f"rank_{rank:05d}"
        rank_dir.mkdir()
        completion_order = list(expected)
        randomizer.shuffle(completion_order)
        rank_path = rank_dir / "rows.jsonl"
        with rank_path.open("w", encoding="utf-8", newline="\n") as handle:
            for identity in completion_order:
                handle.write(
                    json.dumps(
                        {"identity": asdict(identity), "payload": identity.key()},
                        sort_keys=True,
                    )
                    + "\n"
                )
        write_rank_receipt(
            path=rank_dir / "receipt.json",
            rank=rank,
            replicas=args.logical_replicas,
            expected=expected,
            completed=completion_order,
        )
        rank_paths[rank] = rank_path

    merged_path = args.output_dir / "merged.jsonl"
    first = merge_rank_jsonl(
        rank_paths=rank_paths,
        merged_path=merged_path,
        expected=identities,
        replicas=args.logical_replicas,
    )
    second = merge_rank_jsonl(
        rank_paths=rank_paths,
        merged_path=merged_path,
        expected=identities,
        replicas=args.logical_replicas,
    )
    result = {"first_merge": first, "second_merge": second, "logical_replicas": args.logical_replicas}
    (args.output_dir / "dryrun_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if first["status"] != "complete" or second["status"] != "complete":
        raise SystemExit("Replica dry-run merge did not complete")
    if second["newly_merged"] != 0 or second["idempotent_skips"] != len(identities):
        raise SystemExit("Second merge was not idempotent")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
