from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_LEDGER = "7bb4ed06a3a4c725a9893f38e76087c9d6bf2c3caa8d0c880e061ffecc0a1baa"


class HumanEvalPlusAllModelsRerunTests(unittest.TestCase):
    def _config_text(self, name: str) -> str:
        return (ROOT / "configs" / "eval" / name).read_text(encoding="utf-8")

    def test_model_configs_share_the_exact_full_ledger_and_protocol(self) -> None:
        triglu = self._config_text("qwen3_1p7b_heplus_nochat_triglu294_full164_20260717_v1.yaml")
        baseline = self._config_text("qwen3_1p7b_heplus_nochat_baseline196_full164_20260717_v1.yaml")
        for config in (triglu, baseline):
            self.assertIn("sample_count: 164", config)
            self.assertIn("seed: 20260707", config)
            self.assertIn("max_tokens: 3072", config)
            self.assertIn("  - evalscope_raw_instruction_nochat", config)
            self.assertIn(f"expected_task_ledger_sha256: {EXPECTED_LEDGER}", config)
        self.assertIn("vllm_plugin: triglu", triglu)
        self.assertNotIn("vllm_plugin:", baseline)

    def test_controller_is_six_way_serial_triglu_then_baseline(self) -> None:
        script = (
            ROOT / "scripts" / "run_qwen3_1p7b_heplus_nochat_allmodels_full164_20260717_v1.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("for gpu in 0 1 2 3 4 5", script)
        self.assertLess(script.rindex("triglu_step294"), script.rindex("baseline_step196"))
        self.assertIn("HEPLUS_ALLMODELS_GPU_BUSY_REFUSING_TO_CONTEND", script)
        self.assertIn("ALLMODELS_COMPLETE", script)

    def test_both_human_readable_monitors_include_corrected_protocol(self) -> None:
        for name in (
            "monitor_qwen3_1p7b_ood_6x5090_step294_20260716_v1.sh",
            "monitor_triglu_priority_to294_then_baseline196_20260715_v1.sh",
        ):
            text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
            self.assertIn("summarize_qwen_eval_dashboard.py", text)


if __name__ == "__main__":
    unittest.main()
