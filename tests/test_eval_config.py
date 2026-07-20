from __future__ import annotations

import inspect
import sys
import types
import unittest
from pathlib import Path

from qwen_single_layer_rl.eval.run_evalscope import (
    _evaluation_phases,
    _generation_config,
    _task_config,
)


class EvalConfigTests(unittest.TestCase):
    def test_amc_first_changes_only_evaluation_phase_order(self) -> None:
        self.assertEqual(_evaluation_phases(amc_first=False), ("main", "amc", "amc_greedy"))
        self.assertEqual(_evaluation_phases(amc_first=True), ("amc", "amc_greedy", "main"))
        self.assertEqual(
            _evaluation_phases(amc_first=False, amc_only=True),
            ("amc", "amc_greedy"),
        )
        self.assertEqual(
            _evaluation_phases(amc_first=False, amc_only=True, include_amc_greedy=False),
            ("amc",),
        )

    def test_main_greedy_and_amc_sampling_configs_are_distinct(self) -> None:
        greedy = _generation_config(max_tokens=3072, temperature=0.0, top_p=1.0, seed=7)
        sampled = _generation_config(max_tokens=3072, temperature=1.0, top_p=1.0, seed=7)

        self.assertFalse(greedy["do_sample"])
        self.assertNotIn("top_p", greedy)
        self.assertTrue(sampled["do_sample"])
        self.assertEqual(sampled["temperature"], 1.0)
        self.assertEqual(sampled["top_p"], 1.0)
        self.assertEqual(greedy["seed"], 7)
        self.assertEqual(sampled["seed"], 7)

    def test_task_config_requires_explicit_eval_batch_size(self) -> None:
        parameter = inspect.signature(_task_config).parameters["eval_batch_size"]
        self.assertIs(parameter.default, inspect.Parameter.empty)

    def test_task_config_passes_explicit_cache_directory(self) -> None:
        fake_evalscope = types.ModuleType("evalscope")

        class FakeTaskConfig:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

        fake_evalscope.TaskConfig = FakeTaskConfig
        previous = sys.modules.get("evalscope")
        sys.modules["evalscope"] = fake_evalscope
        try:
            config = _task_config(
                model=object(),
                datasets=["paper_math500"],
                work_dir=Path("work"),
                repeats=1,
                limit=None,
                seed=7,
                max_tokens=64,
                temperature=0.0,
                top_p=1.0,
                eval_batch_size=8,
                use_cache=Path("cached/main/20260711_213137"),
            )
        finally:
            if previous is None:
                del sys.modules["evalscope"]
            else:
                sys.modules["evalscope"] = previous

        self.assertEqual(
            config.kwargs["use_cache"],
            str(Path("cached/main/20260711_213137")),
        )
        self.assertEqual(config.kwargs["generation_config"]["seed"], 7)

    def test_task_config_enables_explicit_local_code_sandbox(self) -> None:
        fake_evalscope = types.ModuleType("evalscope")

        class FakeTaskConfig:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

        fake_evalscope.TaskConfig = FakeTaskConfig
        previous = sys.modules.get("evalscope")
        sys.modules["evalscope"] = fake_evalscope
        try:
            config = _task_config(
                model=object(), datasets=["humaneval_plus"], work_dir=Path("work"),
                repeats=1, limit=None, seed=7, max_tokens=64, temperature=0.0,
                top_p=1.0, eval_batch_size=8, local_code_sandbox=True,
            )
        finally:
            if previous is None:
                del sys.modules["evalscope"]
            else:
                sys.modules["evalscope"] = previous
        self.assertTrue(config.kwargs["sandbox"]["enabled"])
        self.assertEqual(config.kwargs["sandbox"]["pool_size"], 8)


if __name__ == "__main__":
    unittest.main()
