from __future__ import annotations

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

from qwen_single_layer_rl.eval.evalscope_custom_model import (
    _SynchronousMicroBatcher,
    canonical_request_identity,
)


class EvalBatcherTests(unittest.TestCase):
    def test_generic_evalscope_identity_is_prompt_stable_and_namespaced(self) -> None:
        first = canonical_request_identity(None, prompt="prompt", namespace="gpqa_diamond")
        repeated = canonical_request_identity(None, prompt="prompt", namespace="gpqa_diamond")
        other = canonical_request_identity(None, prompt="prompt", namespace="mmlu_pro")
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, other)
        self.assertEqual(first["source"], "evalscope_rendered_prompt")

    def test_explicit_benchmark_identity_is_preserved(self) -> None:
        explicit = {"benchmark": "paper_amc23", "item_id": 7, "sample_id": 2}
        self.assertEqual(
            canonical_request_identity(explicit, prompt="ignored", namespace="paper_benchmarks"),
            explicit,
        )

    def test_concurrent_requests_form_one_micro_batch(self) -> None:
        calls: list[list[int]] = []
        call_lock = threading.Lock()

        def process(values: list[int]) -> list[int]:
            with call_lock:
                calls.append(list(values))
            return [value * 2 for value in values]

        batcher = _SynchronousMicroBatcher(process, wait_seconds=0.050)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    batcher.submit,
                    value,
                    signature=("greedy",),
                    max_batch_size=4,
                )
                for value in range(4)
            ]
            results = [future.result(timeout=2) for future in futures]

        self.assertEqual(results, [0, 2, 4, 6])
        self.assertEqual(len(calls), 1)
        self.assertEqual(sorted(calls[0]), [0, 1, 2, 3])

    def test_different_generation_signatures_do_not_mix(self) -> None:
        calls: list[list[str]] = []

        def process(values: list[str]) -> list[str]:
            calls.append(list(values))
            return values

        batcher = _SynchronousMicroBatcher(process, wait_seconds=0.010)
        with ThreadPoolExecutor(max_workers=2) as executor:
            greedy = executor.submit(
                batcher.submit,
                "greedy",
                signature=(0.0,),
                max_batch_size=2,
            )
            sampled = executor.submit(
                batcher.submit,
                "sampled",
                signature=(1.0,),
                max_batch_size=2,
            )
            self.assertEqual(greedy.result(timeout=2), "greedy")
            self.assertEqual(sampled.result(timeout=2), "sampled")

        self.assertEqual(sorted(len(call) for call in calls), [1, 1])


if __name__ == "__main__":
    unittest.main()
