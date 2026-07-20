from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from qwen_single_layer_rl.config import load_config
from qwen_single_layer_rl.training.verl_command import build_verl_command


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "runtime" / (
    "triglu_6x5090_grpo_resume196_to294_cosine5e7_to5e8_seed20260707_v1.yaml"
)
BASELINE_CONFIG = ROOT / "configs" / "runtime" / (
    "baseline_6x5090_grpo_resume196_to294_cosine5e7_to5e8_seed20260707_v1.yaml"
)
CONTROLLER = ROOT / "scripts" / "run_triglu_priority_to294_then_baseline196_20260715_v1.sh"
WATCHER = ROOT / "scripts" / "autostart_triglu_priority_to294_then_baseline196_20260715_v1.sh"
MONITOR = ROOT / "scripts" / "monitor_triglu_priority_to294_then_baseline196_20260715_v1.sh"
PARALLEL_EVAL = ROOT / "scripts" / "run_parallel_vllm_eval_6gpu_20260712_v1.sh"
TRANSITION = ROOT / "scripts" / "prepare_grpo_lr_stage_transition_checkpoint.py"
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
PLAN = ROOT / "docs" / "experiment_plans" / (
    "2026-07-14_triglu-baseline-6x5090-grpo-retro30-60-and-98to196-interleaved-plan.md"
)


class TriGLUPriorityContinuationTests(unittest.TestCase):
    def test_third_stage_config_has_exact_approved_contract(self) -> None:
        cfg = load_config(CONFIG)
        self.assertEqual(cfg["grpo"]["learning_rate"], 5e-7)
        self.assertEqual(cfg["grpo"]["total_training_steps"], 294)
        self.assertEqual(cfg["grpo"]["save_freq"], 10)
        self.assertEqual(cfg["continuation"]["source_global_step"], 196)
        self.assertEqual(cfg["continuation"]["reference_policy_source_global_step"], 196)
        self.assertEqual(cfg["continuation"]["milestones"], [226, 256, 294])
        self.assertEqual(
            cfg["continuation"]["retained_checkpoint_steps"],
            [210, 220, 226, 240, 250, 256, 270, 280, 290, 294],
        )
        self.assertEqual(cfg["continuation"]["redundant_regular_checkpoint_steps"], [200, 230, 260])
        self.assertEqual(
            cfg["continuation"]["lr_decay"],
            {
                "enabled": True,
                "scheduler": "cosine",
                "start_global_step": 196,
                "end_global_step": 294,
                "min_lr_ratio": 0.1,
            },
        )

    def test_baseline_third_stage_matches_triglu_schedule(self) -> None:
        triglu = load_config(CONFIG)
        baseline = load_config(BASELINE_CONFIG)
        for key in ("learning_rate", "total_training_steps", "save_freq"):
            self.assertEqual(baseline["grpo"][key], triglu["grpo"][key])
        self.assertEqual(baseline["experiment"]["initialization_contract"], "untuned_base_exact_noop")
        self.assertEqual(baseline["experiment"]["reference_policy_contract"], "frozen_own_global_step_196")
        self.assertEqual(baseline["continuation"]["reference_policy_source_global_step"], 196)
        self.assertEqual(baseline["continuation"]["milestones"], [226, 256, 294])
        self.assertEqual(baseline["continuation"]["lr_decay"], triglu["continuation"]["lr_decay"])

    def test_controller_finishes_near_complete_baseline_before_third_triglu_stage(self) -> None:
        text = CONTROLLER.read_text(encoding="utf-8")
        schedule = text.rsplit("validate_checkpoint \"$(new_checkpoint triglu 158)\"", 1)[-1]
        baseline_196 = schedule.index("for target in 158 196")
        triglu_transition = schedule.index("prepare_third_stage_transition triglu", baseline_196)
        triglu_loop = schedule.index("for target in 226 256 294", triglu_transition)
        baseline_transition = schedule.index("prepare_third_stage_transition baseline", triglu_loop)
        baseline_loop = schedule.index("for target in 226 256 294", baseline_transition)
        positions = [baseline_196, triglu_transition, triglu_loop, baseline_transition, baseline_loop]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("5e-7", text)
        self.assertIn("SOURCE_ROOT", text)
        self.assertIn("TRIGLU_THIRD_RUN_ID", text)
        self.assertIn("BASELINE_THIRD_RUN_ID", text)
        self.assertIn('for archive_root in "$REORDER_ROOT" "$SOURCE_ROOT"', text)
        self.assertIn('archived="$archive_root/evaluations/${variant}_step_${step}"', text)
        self.assertIn("PARALLEL_EVAL_ARCHIVED_COMPLETE", text)
        self.assertIn("THIRD_STAGE_TRANSITION_ALREADY_READY", text)

    def test_controller_resets_each_third_stage_reference_to_own_step196(self) -> None:
        text = CONTROLLER.read_text(encoding="utf-8")
        self.assertIn('$SOURCE_ROOT/exports/${variant}_step_98', text)
        self.assertIn('$RUN_ROOT/exports/${variant}_step_196', text)
        self.assertIn('reference=$(reference_model_for "$variant" "$target")', text)
        self.assertIn('prepare_third_stage_reference triglu', text)
        self.assertIn('prepare_third_stage_reference baseline', text)
        self.assertIn('FROZEN_OWN_STEP196_REFERENCE_READY', text)
        schedule = text.rsplit('validate_checkpoint "$(new_checkpoint triglu 158)"', 1)[-1]
        baseline_loop = schedule.index("for target in 158 196")
        triglu_reference = schedule.index("prepare_third_stage_reference triglu", baseline_loop)
        baseline_reference = schedule.index("prepare_third_stage_reference baseline", triglu_reference)
        triglu_transition = schedule.index("prepare_third_stage_transition triglu", baseline_reference)
        self.assertEqual(
            [baseline_loop, triglu_reference, baseline_reference, triglu_transition],
            sorted([baseline_loop, triglu_reference, baseline_reference, triglu_transition]),
        )

    def test_eval_resume_preserves_finished_shards_and_uses_evalscope_cache(self) -> None:
        controller = CONTROLLER.read_text(encoding="utf-8")
        evaluator = PARALLEL_EVAL.read_text(encoding="utf-8")
        self.assertIn("PARALLEL_EVAL_RESUME_PRESERVING_SHARDS", controller)
        self.assertIn("EVAL_RANK_ALREADY_COMPLETE", evaluator)
        self.assertIn("EVAL_RANK_RESUME", evaluator)
        self.assertIn("--main-use-cache", evaluator)
        self.assertIn("--amc-use-cache", evaluator)
        self.assertIn("--amc-greedy-use-cache", evaluator)

    def test_late_resume_preflight_accepts_retained_step196_after_old_boundaries_are_pruned(self) -> None:
        controller = CONTROLLER.read_text(encoding="utf-8")
        self.assertIn('validate_checkpoint "$(new_checkpoint baseline 196)"', controller)
        self.assertIn('validate_checkpoint "$(new_checkpoint triglu 196)"', controller)
        self.assertIn("PREFLIGHT_LATE_RESUME_CHECKPOINT", controller)
        self.assertIn("PREFLIGHT_LATE_RESUME_SKIPS_BLOCKED_STEP98_REFERENCES", controller)
        completed = controller.index("if (( completed >= target ))")
        reference = controller.index('reference=$(reference_model_for "$variant" "$target")')
        self.assertLess(completed, reference)

    def test_existing_handoff_watcher_arms_the_both_to_294_extension(self) -> None:
        text = WATCHER.read_text(encoding="utf-8")
        self.assertIn('CONTROLLER_NEXT="$CONTROLLER.next"', text)
        self.assertIn("WAITING_FOR_BOTH_TO_294_EXTENSION", text)
        self.assertIn("BOTH_TO_294_EXTENSION_LAUNCHED", text)
        self.assertIn("triglu_step294_checkpoint_missing", text)
        self.assertIn("baseline_step196_checkpoint_missing", text)
        self.assertIn("VALIDATING_STEP196_REFERENCE_RESET_BOUNDARY", text)
        self.assertIn("third_stage_update_exists_before_step196_reference_reset", text)
        self.assertIn("STEP196_REFERENCE_RESET_BOUNDARY_VALIDATED", text)

    def test_watcher_takes_over_only_after_durable_triglu_158(self) -> None:
        text = WATCHER.read_text(encoding="utf-8")
        self.assertIn("WAITING_FOR_TRIGLU_158", text)
        self.assertIn('validate_checkpoint "$checkpoint"', text)
        self.assertIn("! triglu_158_trainer_running", text)
        self.assertIn('screen -S "$SOURCE_SCREEN" -X quit', text)
        self.assertIn('validate_checkpoint "$SOURCE_ROOT/$SOURCE_BASELINE_RUN/checkpoints/global_step_128"', text)
        self.assertIn("baseline_step128_eval_missing", text)
        self.assertIn('screen -dmS "$SUCCESSOR_SCREEN"', text)
        self.assertNotIn("shutdown", text)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is required for checkpoint transition tests")
    def test_scheduler_transition_rebases_without_resetting_step(self) -> None:
        spec = importlib.util.spec_from_file_location("stage_transition", TRANSITION)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        state = {
            "lr_scheduler": {
                "base_lrs": [5e-6],
                "_last_lr": [5e-7],
                "last_epoch": 196,
                "_step_count": 197,
            }
        }
        result = module.rebase_scheduler(state, base_lr=5e-7, global_step=196)
        self.assertEqual(result["lr_scheduler"]["base_lrs"], [5e-7])
        self.assertEqual(result["lr_scheduler"]["_last_lr"], [5e-7])
        self.assertEqual(result["lr_scheduler"]["last_epoch"], 196)
        self.assertEqual(result["lr_scheduler"]["_step_count"], 197)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is required for checkpoint transition tests")
    def test_scheduler_transition_writes_verl_integer_tracker(self) -> None:
        spec = importlib.util.spec_from_file_location("stage_transition_tracker", TRANSITION)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            tracker = module.write_latest_checkpoint_tracker(Path(tmp), 196)
            self.assertEqual(tracker.read_text(encoding="utf-8"), "196\n")

    def test_third_stage_command_uses_fixed_294_horizon(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            trainer = temp / "verl" / "verl" / "trainer" / "ppo" / "ray_trainer.py"
            trainer.parent.mkdir(parents=True)
            trainer.write_text("optimizer_schedule_total_training_steps\n", encoding="utf-8")
            cfg = load_config(CONFIG)
            run_id = cfg["logging"]["run_id_template"]
            tracker = temp / "run" / run_id / "checkpoints" / "latest_checkpointed_iteration.txt"
            tracker.parent.mkdir(parents=True)
            tracker.write_text("196\n", encoding="utf-8")
            command, manifest = build_verl_command(
                CONFIG,
                project_root=ROOT,
                verl_root=temp / "verl",
                model_path=temp / "model",
                data_dir=ROOT / "data",
                run_root=temp / "run",
                reference_model_path=temp / "reference",
            )
            joined = " ".join(command)
            self.assertIn("actor_rollout_ref.actor.optim.lr=5e-07", joined)
            self.assertIn("++trainer.optimizer_schedule_total_training_steps=294", joined)
            self.assertIn("++actor_rollout_ref.actor.optim.lr_warmup_steps=196", joined)
            self.assertIn("++actor_rollout_ref.actor.optim.min_lr_ratio=0.1", joined)
            self.assertTrue(manifest["paper_hyperparams"]["continuation_lr_decay"]["active"])

    def test_monitor_and_plan_keep_full_contract_visible(self) -> None:
        monitor = MONITOR.read_text(encoding="utf-8")
        for token in (
            "both variants to 294",
            "cosine 5e-7 to 5e-8 at step 294",
            "baseline 226/256/294",
            "own frozen step 196 for steps 197-294",
            "segment ETA",
            "latest completed metrics",
            "archived log",
            "summarize_parallel_eval.py",
            "for step in 98 128 158 196 226 256 294",
            'for root in "$R" "$REORDER_ROOT" "$SOURCE_ROOT" "$OLD_ROOT"',
            'if errors and phase in {"TRAIN", "EVAL"}',
            "pre-baseline OOD: active",
        ):
            self.assertIn(token, monitor)
        plan = PLAN.read_text(encoding="utf-8")
        for pending in ("PENDING-01", "PENDING-02", "PENDING-03"):
            self.assertIn(pending, plan)


if __name__ == "__main__":
    unittest.main()
