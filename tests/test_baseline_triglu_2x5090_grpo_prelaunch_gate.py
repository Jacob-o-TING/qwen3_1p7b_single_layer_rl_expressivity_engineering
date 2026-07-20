from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is required for the GRPO prelaunch gate tests")

import torch
import torch.nn as nn

from qwen_single_layer_rl.config import load_config
from qwen_single_layer_rl.training.verl_command import build_verl_command
from qwen_single_layer_rl.training.verl_model_hook import apply_model_surgery_before_fsdp


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/runtime/baseline_triglu_2x5090_grpo_prelaunch_gate_20260712_v1.yaml"
SCRIPT = ROOT / "scripts/run_baseline_triglu_2x5090_grpo_prelaunch_gate.py"


def load_script():
    spec = importlib.util.spec_from_file_location("prelaunch_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _TriGLUMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_mlp = nn.Linear(2, 2, bias=False)
        self.triglu_side = nn.ModuleDict({"up": nn.Linear(2, 2)})


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _TriGLUMLP()


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Layer()])


class _TriGLUVariant:
    name = "qwen_swiglu_triglu_side"

    def apply(self, model, cfg):
        raise AssertionError("preconstructed TriGLU must not be injected twice")


class BaselineTriGLUPrelaunchTests(unittest.TestCase):
    def test_config_carries_pending_and_exact_scientific_contract(self) -> None:
        cfg = load_config(CONFIG)
        gate = cfg["prelaunch_gate"]
        self.assertEqual(gate["run_id"], "baseline_triglu_2x5090_grpo_prelaunch_gate_20260712_v1")
        self.assertEqual(gate["group_size"], 4)
        self.assertEqual(gate["variants"], ["baseline", "triglu"])
        self.assertEqual(cfg["experiment"]["initialization_contract"], "untuned_base_exact_noop")
        self.assertTrue(cfg["freeze_policy"]["train_adapter_modules"])
        self.assertEqual(set(cfg["pending_obligations_carried_forward"]), {"PENDING-01", "PENDING-02", "PENDING-03"})

    def test_reward_failure_classification_is_bounded_and_explicit(self) -> None:
        module = load_script()
        self.assertEqual(module.classify_zero_reward("", 0, 128, "stop"), "empty")
        self.assertEqual(module.classify_zero_reward("work", 128, 128, "length"), "token_cap")
        self.assertEqual(module.classify_zero_reward("answer is \\boxed{3", 7, 128, "stop"), "invalid_latex")
        self.assertEqual(module.classify_zero_reward("I cannot solve this", 4, 128, "stop"), "extraction_or_format")
        self.assertEqual(module.classify_zero_reward("The answer is 7", 5, 128, "stop"), "valid_but_wrong")

    def test_ratio_and_delta_receipts_detect_drift(self) -> None:
        module = load_script()
        delta = module.summarize_deltas([[-1.0, -2.0]], [[-1.0, -1.5]])
        ratio = module.ratio_summary([[-1.0, -2.0]], [[-1.0, -1.5]], 0.2)
        self.assertEqual(delta["count"], 2)
        self.assertEqual(delta["max_abs"], 0.5)
        self.assertEqual(ratio["finite_count"], 2)
        self.assertEqual(ratio["clip_fraction"], 0.5)

    def test_live_sync_rpc_is_explicitly_same_host_opted_in(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"', source)
        self.assertIn("llm.collective_rpc", source)

    def test_preconstructed_triglu_allows_explicit_untuned_noop_without_overlay(self) -> None:
        cfg = {
            "experiment": {"seed": 17, "init_seed": 17},
            "model": {"num_layers": 1},
            "architecture_variant": {
                "name": "qwen_swiglu_triglu_side",
                "params": {"target_layers": [0]},
            },
            "freeze_policy": {
                "backbone_train_mode": "selected",
                "train_layers": [0],
                "train_adapter_modules": False,
                "freeze_embeddings": True,
                "freeze_lm_head": True,
            },
        }
        model = _Model()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = root / "audit"
            env = {
                "QWEN_SINGLE_LAYER_RL_CONFIG": str(root / "config.yaml"),
                "QWEN_SINGLE_LAYER_RL_INITIALIZATION": "untuned_base_exact_noop",
                "QWEN_SINGLE_LAYER_RL_AUDIT_DIR": str(audit),
            }
            with patch.dict("os.environ", env, clear=False), patch(
                "qwen_single_layer_rl.training.verl_model_hook.load_config", return_value=cfg
            ), patch(
                "qwen_single_layer_rl.training.verl_model_hook.build_variant", return_value=_TriGLUVariant()
            ):
                result = apply_model_surgery_before_fsdp(model, role="actor", rank=0)
            payload = json.loads((audit / "actor_rank0_model_surgery_audit.json").read_text())
        self.assertIs(result, model)
        self.assertEqual(payload["construction_mode"], "preconstructed")
        self.assertEqual(payload["initialization_contract"], "untuned_base_exact_noop")
        self.assertEqual(payload["checkpoint_overlay_count"], 0)

    def test_verl_command_accepts_untuned_triglu_without_checkpoint(self) -> None:
        command, manifest = build_verl_command(
            CONFIG,
            project_root=ROOT,
            verl_root=ROOT / "vendor/verl",
            model_path=ROOT / "models/untuned-triglu",
            data_dir=ROOT / "data/numina_math_cot_50k_decontam_v3_verl",
            run_root=ROOT / "runs",
            checkpoint_dir=None,
        )
        self.assertEqual(manifest["checkpoint_dir"], None)
        self.assertEqual(manifest["env"]["QWEN_SINGLE_LAYER_RL_INITIALIZATION"], "untuned_base_exact_noop")
        self.assertIn("TRIGLU_DISPATCH_RECEIPT", manifest["env"])
        self.assertIn("+actor_rollout_ref.rollout.engine_kwargs.vllm.model_impl=auto", command)


if __name__ == "__main__":
    unittest.main()
