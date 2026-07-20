import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_shs_2x5090_production_length_throughput.py"
SPEC = importlib.util.spec_from_file_location("throughput_gate", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ThroughputGateTests(unittest.TestCase):
    def test_percentile_and_group_expansion(self):
        self.assertEqual(MODULE.percentile([1, 2, 3, 4], 0.5), 3.0)
        prompts = [
            {"gpu_index": 0, "slot": index, "prompt_id": f"p{index}", "formatted_prompt": "x"}
            for index in range(4)
        ]
        rows = MODULE.expanded_requests(prompts, 0, 16, 4, 100)
        self.assertEqual(len(rows), 16)
        self.assertEqual({row["sample_index"] for row in rows}, {0, 1, 2, 3})
        self.assertEqual(len({row["seed"] for row in rows}), 16)

    def test_atomic_complete_cell_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cell.json"
            MODULE.atomic_write_json(
                path,
                {"status": "passed", "cell": "production_cap3072", "pressure": 16, "gpu_index": 0, "response_count": 16},
            )
            self.assertTrue(MODULE.cell_is_complete(path, "production_cap3072", 16, 0))
            self.assertFalse(MODULE.cell_is_complete(path, "production_cap3072", 32, 0))
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_failed_manifest_is_not_a_completed_cell(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cell.json"
            MODULE.atomic_write_json(
                path,
                {"status": "failed", "cell": "matched_800_1024", "pressure": 16, "gpu_index": 0, "response_count": 0},
            )
            self.assertFalse(MODULE.cell_is_complete(path, "matched_800_1024", 16, 0))

    def test_stage_wait_allows_a_fast_rank_to_exit_after_own_receipt(self):
        class Process:
            def __init__(self, return_code):
                self.return_code = return_code

            def poll(self):
                return self.return_code

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fast = root / "gpu0" / "stages" / "pressure_64.json"
            slow = root / "gpu1" / "stages" / "pressure_64.json"
            MODULE.atomic_write_json(fast, {"gpu_index": 0, "pressure": 64, "status": "passed"})
            failures = MODULE.premature_stage_failures([Process(0), Process(None)], [fast, slow])
            self.assertEqual(failures, [])
            failures = MODULE.premature_stage_failures([Process(0), Process(1)], [fast, slow])
            self.assertEqual(failures, [str(slow)])

    def test_pair_concurrency_uses_start_skew_not_equal_duration(self):
        rows = [
            {"started_epoch": 100.0, "ended_epoch": 283.0, "wall_seconds": 183.0},
            {"started_epoch": 101.5, "ended_epoch": 349.5, "wall_seconds": 248.0},
        ]
        result = MODULE.pair_concurrency(rows)
        self.assertTrue(result["pair_concurrent"])
        self.assertAlmostEqual(result["start_skew_seconds"], 1.5)
        rows[1]["started_epoch"] = 110.0
        result = MODULE.pair_concurrency(rows)
        self.assertFalse(result["pair_concurrent"])

    def test_decision_requires_both_pressure64_cells_and_length_match(self):
        config = {
            "workload": {"cells": {"matched_800_1024": {}, "production_cap3072": {}}},
            "gates": {
                "production_mean_target_tokens": 897.6,
                "production_mean_relative_tolerance": 0.25,
                "matched_rtx_pro6000_anchor_tokens_per_second_per_gpu": 2262.8,
                "c2_tokens_per_replica_step": 516253,
            },
        }
        cells = []
        for pressure in (16, 64):
            for cell in config["workload"]["cells"]:
                cells.append(
                    {
                        "cell": cell,
                        "pressure_per_gpu": pressure,
                        "output_length_mean": 900.0,
                        "mean_per_gpu_generated_tokens_per_second": 2000.0,
                        "pair_generated_tokens_per_second": 3900.0,
                        "pair_concurrent": True,
                    }
                )
        decision = MODULE.build_decision(config, {"cells": cells}, [{"dispatch_passed": True}, {"dispatch_passed": True}])
        self.assertTrue(decision["decision_grade"])
        self.assertAlmostEqual(decision["matched_comparison"]["rtx5090_over_rtx_pro6000"], 2000 / 2262.8)
        cells[-1]["output_length_mean"] = 300.0
        decision = MODULE.build_decision(config, {"cells": cells}, [{"dispatch_passed": True}, {"dispatch_passed": True}])
        self.assertFalse(decision["decision_grade"])
        cells[-1]["output_length_mean"] = 900.0
        cells[-1]["pair_concurrent"] = False
        decision = MODULE.build_decision(config, {"cells": cells}, [{"dispatch_passed": True}, {"dispatch_passed": True}])
        self.assertFalse(decision["decision_grade"])


if __name__ == "__main__":
    unittest.main()
