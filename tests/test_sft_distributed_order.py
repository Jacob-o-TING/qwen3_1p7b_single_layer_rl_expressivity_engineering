from __future__ import annotations

import unittest

from qwen_single_layer_rl.sft.distributed import (
    build_rank_micro_batch_schedule,
    epoch_order,
    gradient_accumulation_for_effective_batch,
    reconstruct_global_micro_batches,
)


class SFTDistributedOrderTests(unittest.TestCase):
    def test_rank_slices_reconstruct_one_global_order(self) -> None:
        schedules = [
            build_rank_micro_batch_schedule(
                41,
                seed=20260707,
                epoch=0,
                shuffle=True,
                rank=rank,
                world_size=4,
                micro_batch_size=2,
            )
            for rank in range(4)
        ]
        rebuilt = [index for batch in reconstruct_global_micro_batches(schedules) for index in batch]
        expected = epoch_order(41, seed=20260707, epoch=0, shuffle=True)[:40]
        self.assertEqual(rebuilt, expected)
        self.assertEqual(len(set(rebuilt)), len(rebuilt))
        self.assertEqual({schedule.global_order_sha256 for schedule in schedules}, {schedules[0].global_order_sha256})
        self.assertEqual({schedule.dropped_items for schedule in schedules}, {1})

    def test_same_seed_is_repeatable_and_epoch_changes_order(self) -> None:
        kwargs = dict(
            num_items=64,
            seed=20260707,
            shuffle=True,
            rank=0,
            world_size=4,
            micro_batch_size=2,
        )
        first = build_rank_micro_batch_schedule(epoch=0, **kwargs)
        repeated = build_rank_micro_batch_schedule(epoch=0, **kwargs)
        next_epoch = build_rank_micro_batch_schedule(epoch=1, **kwargs)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first.global_order_sha256, next_epoch.global_order_sha256)

    def test_unshuffled_order_is_identity(self) -> None:
        self.assertEqual(epoch_order(5, seed=7, epoch=9, shuffle=False), [0, 1, 2, 3, 4])

    def test_accumulation_preserves_effective_batch_across_gpu_counts(self) -> None:
        self.assertEqual(
            gradient_accumulation_for_effective_batch(8, world_size=1, micro_batch_size=1),
            8,
        )
        self.assertEqual(
            gradient_accumulation_for_effective_batch(8, world_size=4, micro_batch_size=1),
            2,
        )

    def test_accumulation_rejects_non_divisible_topology(self) -> None:
        with self.assertRaises(ValueError):
            gradient_accumulation_for_effective_batch(8, world_size=3, micro_batch_size=1)


if __name__ == "__main__":
    unittest.main()
