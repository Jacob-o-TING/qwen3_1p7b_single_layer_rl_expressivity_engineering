from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class UnifiedEvalMonitorTests(unittest.TestCase):
    def test_main_monitor_embeds_one_shared_dashboard_and_ood_progress(self) -> None:
        main = (
            ROOT / "scripts/monitor_triglu_priority_to294_then_baseline196_20260715_v1.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("summarize_qwen_eval_dashboard.py", main)
        self.assertIn("monitor_qwen3_1p7b_ood_6x5090_step294_20260716_v1.sh\" --embedded", main)

    def test_main_monitor_keeps_math_results_before_ood_section(self) -> None:
        main = (
            ROOT / "scripts/monitor_triglu_priority_to294_then_baseline196_20260715_v1.sh"
        ).read_text(encoding="utf-8")
        math = main.index("=== Core math evaluation results ===")
        milestones = main.index("milestone evaluation comparison by global step")
        ood = main.index("=== OOD / out-of-domain evaluation ===")
        dashboard = main.index("summarize_qwen_eval_dashboard.py")
        self.assertLess(math, milestones)
        self.assertLess(milestones, ood)
        self.assertLess(ood, dashboard)

    def test_standalone_ood_monitor_uses_current_first_dashboard(self) -> None:
        ood = (
            ROOT / "scripts/monitor_qwen3_1p7b_ood_6x5090_step294_20260716_v1.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('MODE="${1:-full}"', ood)
        self.assertIn("summarize_qwen_eval_dashboard.py", ood)
        self.assertIn('[[ "$MODE" != "--embedded" ]]', ood)

    def test_dashboard_prioritizes_corrected_and_preserves_heritage(self) -> None:
        dashboard = (ROOT / "scripts/summarize_qwen_eval_dashboard.py").read_text(encoding="utf-8")
        primary = dashboard.index("PRIMARY corrected code protocol")
        current = dashboard.index("PRIMARY current corrected-protocol summary")
        heritage = dashboard.index("HERITAGE chat-protocol results")
        self.assertLess(primary, current)
        self.assertLess(current, heritage)
        self.assertLess(primary, heritage)
        self.assertIn("corrected CodeAvg", dashboard)
        self.assertIn("PRIMARY reasoning OOD benchmarks", dashboard)
        self.assertIn("PRIMARY language OOD benchmarks", dashboard)
        for benchmark in ("GPQA-Diamond", "MMLU-Pro", "C-Eval", "IFEval", "MGSM"):
            self.assertIn(benchmark, dashboard)
        self.assertIn("heritage OOD means remain immutable", dashboard)
        self.assertIn("baseline_step294", dashboard)


if __name__ == "__main__":
    unittest.main()
