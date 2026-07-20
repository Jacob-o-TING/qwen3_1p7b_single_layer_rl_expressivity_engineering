from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

from qwen_single_layer_rl.config import load_config
from qwen_single_layer_rl.training.verl_command import build_verl_command


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    variant: ROOT
    / "configs"
    / "runtime"
    / f"{variant}_6x5090_grpo_untunedbase_resume98_to196_seed20260707_v1.yaml"
    for variant in ("triglu", "baseline")
}
CONTROLLER = ROOT / "scripts" / (
    "run_triglu_baseline_6x5090_grpo_retro30_60_then_98to196_interleaved_20260714_v1.sh"
)
WATCHER = ROOT / "scripts" / "autostart_triglu_baseline_6x5090_grpo_98to196_interleaved_20260714_v1.sh"
MONITOR = ROOT / "scripts" / "monitor_triglu_baseline_6x5090_grpo_98to196_interleaved_20260714_v1.sh"
PLAN = ROOT / "docs" / "experiment_plans" / (
    "2026-07-14_triglu-baseline-6x5090-grpo-retro30-60-and-98to196-interleaved-plan.md"
)


class InterleavedContinuationTests(unittest.TestCase):
    def test_legacy_monitor_forwards_to_active_continuation(self) -> None:
        legacy_monitor = (
            ROOT / "scripts" / "monitor_triglu_baseline_6x5090_grpo_20260712_v1.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('if [[ -f "$CONTINUATION_ROOT/state.env" ]]', legacy_monitor)
        self.assertIn('exec bash "$CONTINUATION_MONITOR"', legacy_monitor)

    def test_configs_preserve_matched_macro_contract(self) -> None:
        configs = {name: load_config(path) for name, path in CONFIGS.items()}
        fields = (
            "train_batch_size",
            "ppo_mini_batch_size",
            "ppo_micro_batch_size",
            "group_size",
            "total_training_steps",
            "save_freq",
            "test_freq",
            "val_before_train",
        )
        observed = {
            name: {field: cfg["grpo"][field] for field in fields}
            for name, cfg in configs.items()
        }
        self.assertEqual(observed["triglu"], observed["baseline"])
        self.assertEqual(observed["triglu"]["train_batch_size"], 504)
        self.assertEqual(observed["triglu"]["ppo_mini_batch_size"], 126)
        self.assertEqual(observed["triglu"]["ppo_micro_batch_size"], 6)
        self.assertEqual(observed["triglu"]["group_size"], 4)
        self.assertEqual(observed["triglu"]["total_training_steps"], 196)
        self.assertEqual(observed["triglu"]["save_freq"], 10)
        for name, cfg in configs.items():
            self.assertEqual(cfg["experiment"]["seed"], 20260707)
            self.assertEqual(cfg["experiment"]["rollout_seed"], 20260707)
            self.assertEqual(cfg["continuation"]["source_global_step"], 98)
            self.assertEqual(cfg["continuation"]["reference_policy_source_global_step"], 98)
            self.assertEqual(cfg["continuation"]["reference_policy_scope"], "per_variant_frozen")
            self.assertEqual(cfg["experiment"]["reference_policy_contract"], "frozen_own_global_step_98")
            self.assertEqual(cfg["continuation"]["milestones"], [128, 158, 196])
            self.assertEqual(
                cfg["continuation"]["lr_decay"],
                {
                    "enabled": True,
                    "scheduler": "cosine",
                    "start_global_step": 128,
                    "end_global_step": 196,
                    "min_lr_ratio": 0.1,
                },
            )
            self.assertIn(f"{name}_6x5090", cfg["logging"]["run_id_template"])

    def test_verl_command_supports_explicit_first_resume(self) -> None:
        resume = Path("/tmp/source/global_step_98")
        reference = Path("/tmp/reference/global_step_98")
        command, manifest = build_verl_command(
            CONFIGS["triglu"],
            project_root=ROOT,
            verl_root=Path("/tmp/verl"),
            model_path=Path("/tmp/model"),
            data_dir=ROOT / "data",
            run_root=Path("/tmp/new-wave"),
            resume_from_path=resume,
            reference_model_path=reference,
        )
        self.assertIn("trainer.resume_mode=resume_path", command)
        self.assertIn(f"trainer.resume_from_path={resume.resolve()}", command)
        self.assertEqual(manifest["paper_hyperparams"]["resume_mode"], "resume_path")
        self.assertEqual(manifest["paper_hyperparams"]["resume_from_path"], str(resume.resolve()))
        self.assertIn(f"+actor_rollout_ref.ref.model.path={reference.resolve()}", command)
        self.assertEqual(manifest["reference_model_path"], str(reference.resolve()))
        self.assertEqual(manifest["paper_hyperparams"]["reference_policy_contract"], "explicit_frozen_reference")

        automatic, auto_manifest = build_verl_command(
            CONFIGS["triglu"],
            project_root=ROOT,
            verl_root=Path("/tmp/verl"),
            model_path=Path("/tmp/model"),
            data_dir=ROOT / "data",
            run_root=Path("/tmp/new-wave"),
        )
        self.assertIn("trainer.resume_mode=auto", automatic)
        self.assertNotIn("trainer.resume_from_path=", automatic)
        self.assertEqual(auto_manifest["paper_hyperparams"]["resume_mode"], "auto")
        self.assertEqual(auto_manifest["paper_hyperparams"]["reference_policy_contract"], "shared_initial_model")
        self.assertFalse(auto_manifest["paper_hyperparams"]["continuation_lr_decay"]["active"])

    def test_lr_decay_activates_only_after_step_128_with_fixed_horizon(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            verl = temp / "verl"
            trainer = verl / "verl" / "trainer" / "ppo" / "ray_trainer.py"
            trainer.parent.mkdir(parents=True)
            trainer.write_text("optimizer_schedule_total_training_steps\n", encoding="utf-8")
            run_root = temp / "runs"
            for name, config in CONFIGS.items():
                cfg = load_config(config)
                run_id = cfg["logging"]["run_id_template"]
                tracker = run_root / run_id / "checkpoints" / "latest_checkpointed_iteration.txt"
                tracker.parent.mkdir(parents=True)
                tracker.write_text("global_step_128\n", encoding="utf-8")
                command, manifest = build_verl_command(
                    config,
                    project_root=ROOT,
                    verl_root=verl,
                    model_path=temp / "model",
                    data_dir=ROOT / "data",
                    run_root=run_root,
                    reference_model_path=temp / "reference",
                )
                joined = " ".join(command)
                self.assertIn("++trainer.optimizer_schedule_total_training_steps=196", joined)
                self.assertIn("++actor_rollout_ref.actor.optim.lr_scheduler_type=cosine", joined)
                self.assertIn("++actor_rollout_ref.actor.optim.lr_warmup_steps=128", joined)
                self.assertIn("++actor_rollout_ref.actor.optim.min_lr_ratio=0.1", joined)
                self.assertTrue(manifest["paper_hyperparams"]["continuation_lr_decay"]["active"], name)

    def test_controller_freezes_each_variants_own_step98_reference(self) -> None:
        text = CONTROLLER.read_text(encoding="utf-8")
        self.assertIn("reference_model_for() {", text)
        self.assertIn('--reference-model-path "$reference"', text)
        self.assertIn("prepare_reference triglu", text)
        self.assertIn("prepare_reference baseline", text)
        self.assertIn('"reference_global_step": 98', text)
        self.assertIn('[[ "$step" -eq 98 || "$step" -eq 196 ]]', text)

    def test_controller_has_exact_interleaved_schedule(self) -> None:
        text = CONTROLLER.read_text(encoding="utf-8")
        schedule = text.rsplit("write_resume_receipt", 1)[-1]
        expected = [
            "evaluate triglu 30",
            "evaluate baseline 30",
            "evaluate triglu 60",
            "evaluate baseline 60",
            'train_to triglu "$target"',
            'evaluate triglu "$target"',
            'train_to baseline "$target"',
            'evaluate baseline "$target"',
            'data_order_receipt "$target" new',
        ]
        positions = [schedule.index(item) for item in expected]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("for target in 128 158 196", schedule)
        self.assertIn("--resume-from-path", text)
        self.assertIn('if (( completed == 0 ))', text)

    def test_checkpoint_retention_deduplicates_near_endpoint_saves(self) -> None:
        text = CONTROLLER.read_text(encoding="utf-8")
        self.assertIn("prune_redundant_regular_checkpoint", text)
        self.assertIn("128) redundant_step=100", text)
        self.assertIn("158) redundant_step=130", text)
        self.assertIn("196) redundant_step=160", text)
        self.assertNotIn('rm -rf "$RUN_ROOT"', text)
        self.assertNotIn("rm -rf \"$OLD_ROOT", text)
        for cfg in map(load_config, CONFIGS.values()):
            self.assertEqual(
                cfg["continuation"]["retained_checkpoint_steps"],
                [110, 120, 128, 140, 150, 158, 170, 180, 190, 196],
            )
            self.assertEqual(cfg["continuation"]["redundant_regular_checkpoint_steps"], [100, 130, 160])
            self.assertEqual(cfg["continuation"]["force_segment_endpoint_checkpoints"], [128, 158, 196])

    def test_watcher_requires_durable_old_completion(self) -> None:
        text = WATCHER.read_text(encoding="utf-8")
        self.assertIn('[[ ! -f "$OLD_ROOT/WAVE_COMPLETE" ]]', text)
        self.assertIn("qwen_grpo_98to196_interleaved_20260714_v1", text)
        self.assertIn("gpu_compute_processes_remain_after_old_wave", text)
        self.assertIn("disk_free_below_100G", text)
        self.assertNotIn("shutdown", text)

    def test_monitor_exposes_paired_live_metrics(self) -> None:
        text = MONITOR.read_text(encoding="utf-8")
        for token in (
            "global 128=round2+30, 158=round2+60, 196=round2+98",
            "paired {pair} cell may be pending",
            "recent speed",
            "segment ETA",
            "learning_rate",
            "summarize_parallel_eval.py",
            "paired data-order receipts",
            "--compare-subdirs --steps 128 158 196",
        ):
            self.assertIn(token, text)

    def test_plan_carries_every_mandatory_pending_gate(self) -> None:
        text = PLAN.read_text(encoding="utf-8")
        self.assertIn("## Pending Obligations Carried Forward", text)
        for pending in ("PENDING-01", "PENDING-02", "PENDING-03"):
            self.assertIn(pending, text)


if __name__ == "__main__":
    unittest.main()
