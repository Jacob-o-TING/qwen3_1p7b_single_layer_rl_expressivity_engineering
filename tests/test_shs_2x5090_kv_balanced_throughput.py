import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_shs_2x5090_kv_balanced_throughput.py"
SPEC = importlib.util.spec_from_file_location("balanced_gate", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class BalancedGateTests(unittest.TestCase):
    def test_parse_engine_profile_uses_last_engine(self):
        text = """
Available KV cache memory: 1.22 GiB
GPU KV cache size: 11,440 tokens
Maximum concurrency for 4,096 tokens per request: 2.79x
Actual usage is 3.25 GiB for weight, 20.6 GiB for peak activation
Available KV cache memory: 12.50 GiB
GPU KV cache size: 117,500 tokens
Maximum concurrency for 4,096 tokens per request: 28.69x
Actual usage is 3.25 GiB for weight, 7.4 GiB for peak activation
"""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "worker.log"
            path.write_text(text, encoding="utf-8")
            result = MODULE.parse_engine_profile(path)
        self.assertEqual(result["kv_cache_tokens"], 117500)
        self.assertEqual(result["full_context_concurrency"], 28.69)
        self.assertEqual(result["peak_activation_gib"], 7.4)

    def test_conditional_runs_only_on_capacity_trigger(self):
        receipt = {
            "oom": False,
            "engine_profiles": [
                {"gpu_index": 0, "kv_cache_tokens": 117500, "full_context_concurrency": 28.69, "peak_activation_gib": 7.4, "available_kv_cache_gib": 12.5},
                {"gpu_index": 1, "kv_cache_tokens": 117500, "full_context_concurrency": 28.69, "peak_activation_gib": 7.4, "available_kv_cache_gib": 12.5},
            ],
        }
        trigger = {"minimum_kv_cache_tokens": 32768, "minimum_full_context_concurrency": 8.0, "run_on_primary_oom": True}
        self.assertEqual(MODULE.conditional_decision(receipt, trigger)["action"], "accept_primary")
        receipt["engine_profiles"][1]["full_context_concurrency"] = 7.0
        self.assertEqual(MODULE.conditional_decision(receipt, trigger)["action"], "run_conditional")

    def test_missing_profile_fails_without_blind_conditional(self):
        receipt = {
            "oom": False,
            "engine_profiles": [
                {"gpu_index": 0, "kv_cache_tokens": None, "full_context_concurrency": None, "peak_activation_gib": None, "available_kv_cache_gib": None},
                {"gpu_index": 1, "kv_cache_tokens": None, "full_context_concurrency": None, "peak_activation_gib": None, "available_kv_cache_gib": None},
            ],
        }
        trigger = {"minimum_kv_cache_tokens": 32768, "minimum_full_context_concurrency": 8.0, "run_on_primary_oom": True}
        self.assertEqual(MODULE.conditional_decision(receipt, trigger)["action"], "fail_stop")

    def test_sampling_stability_keeps_pressure_cells_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            control = temporary / "control"
            balanced = temporary / "balanced"
            for pressure in (16, 32, 64):
                for gpu_index in (0, 1):
                    requests = [
                        {
                            "request_id": f"p{pressure}:g{gpu_index}",
                            "seed": pressure + gpu_index,
                            "generated_tokens": pressure,
                            "token_ids_sha256": "same" if pressure == 16 else f"control-{gpu_index}",
                        }
                    ]
                    for root, suffix in ((control, "control"), (balanced, "balanced")):
                        row = dict(requests[0])
                        if root == balanced and pressure != 16:
                            row["generated_tokens"] += 1
                            row["token_ids_sha256"] = f"balanced-{gpu_index}"
                        path = root / f"gpu{gpu_index}" / "cells" / f"production_cap3072_p{pressure}.json"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps({"requests": [row]}), encoding="utf-8")
            result = MODULE.sampling_stability(control, balanced)
        self.assertEqual(result["16"]["token_trace_equal_count"], 2)
        self.assertEqual(result["64"]["token_trace_equal_count"], 0)
        self.assertEqual(result["64"]["seed_equal_count"], 2)


if __name__ == "__main__":
    unittest.main()
