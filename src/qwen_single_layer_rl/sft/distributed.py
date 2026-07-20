from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class RankMicroBatchSchedule:
    rank: int
    world_size: int
    micro_batches: tuple[tuple[int, ...], ...]
    global_order_sha256: str
    rank_order_sha256: str
    dropped_items: int


def gradient_accumulation_for_effective_batch(
    target_effective_batch_size: int,
    *,
    world_size: int,
    micro_batch_size: int,
) -> int:
    if target_effective_batch_size <= 0:
        raise ValueError("target_effective_batch_size must be positive")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")
    denominator = world_size * micro_batch_size
    if target_effective_batch_size % denominator:
        raise ValueError(
            f"target effective batch {target_effective_batch_size} is not divisible by "
            f"world_size * micro_batch_size ({denominator})"
        )
    return target_effective_batch_size // denominator


def epoch_order(num_items: int, *, seed: int, epoch: int, shuffle: bool) -> list[int]:
    if num_items < 0:
        raise ValueError("num_items must be non-negative")
    if not shuffle:
        return list(range(num_items))
    order = list(range(num_items))
    random.Random(int(seed) + int(epoch)).shuffle(order)
    return order


def _hash_indices(indices: list[int] | tuple[int, ...]) -> str:
    payload = json.dumps(list(indices), separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def build_rank_micro_batch_schedule(
    num_items: int,
    *,
    seed: int,
    epoch: int,
    shuffle: bool,
    rank: int,
    world_size: int,
    micro_batch_size: int,
    drop_last: bool = True,
) -> RankMicroBatchSchedule:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank {rank} is outside world size {world_size}")
    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")

    order = epoch_order(num_items, seed=seed, epoch=epoch, shuffle=shuffle)
    global_micro_batch = world_size * micro_batch_size
    usable = len(order)
    if drop_last:
        usable -= usable % global_micro_batch
    elif usable % global_micro_batch:
        raise ValueError("drop_last=False is unsupported for uneven distributed micro-batches")
    used_order = order[:usable]
    micro_batches: list[tuple[int, ...]] = []
    for offset in range(0, usable, global_micro_batch):
        global_batch = used_order[offset : offset + global_micro_batch]
        start = rank * micro_batch_size
        micro_batches.append(tuple(global_batch[start : start + micro_batch_size]))
    rank_order = [index for batch in micro_batches for index in batch]
    return RankMicroBatchSchedule(
        rank=rank,
        world_size=world_size,
        micro_batches=tuple(micro_batches),
        global_order_sha256=_hash_indices(used_order),
        rank_order_sha256=_hash_indices(rank_order),
        dropped_items=len(order) - usable,
    )


def reconstruct_global_micro_batches(schedules: list[RankMicroBatchSchedule]) -> list[tuple[int, ...]]:
    if not schedules:
        return []
    ordered = sorted(schedules, key=lambda item: item.rank)
    lengths = {len(item.micro_batches) for item in ordered}
    if len(lengths) != 1:
        raise ValueError("rank schedules have different micro-batch counts")
    return [
        tuple(index for schedule in ordered for index in schedule.micro_batches[step])
        for step in range(len(ordered[0].micro_batches))
    ]
