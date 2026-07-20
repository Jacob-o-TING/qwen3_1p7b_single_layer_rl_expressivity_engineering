from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None
ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(TRANSFORMERS_AVAILABLE, "transformers is required")
class VLLMOFTScaffoldTests(unittest.TestCase):
    def test_fp32_rotation_application_is_exact_identity_for_bf16_input(self) -> None:
        import torch

        from qwen_single_layer_rl.model_surgery.qwen_swiglu_variant_modules import OFTRotation

        rotation = OFTRotation(16, 8, fp32_compute=True)
        value = torch.randn(2, 3, 16, dtype=torch.bfloat16)
        observed = rotation(value)
        self.assertEqual(observed.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(observed, value))
        self.assertEqual(rotation.oft_like.dtype, torch.float32)

    def test_explicit_model_preserves_oft_keys_and_fp32_contract(self) -> None:
        import torch

        from qwen_single_layer_rl.model_surgery.qwen_swiglu_variant_modules import QwenSwiGLUOFTWrapper
        from qwen_single_layer_rl.vllm.oft_hf_model import (
            Qwen3OFTConfig,
            Qwen3OFTForCausalLM,
            expected_layer_key_prefixes,
        )

        config = Qwen3OFTConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            oft_variant={"target_layers": [1], "block_size": 8, "fp32_compute": True},
        )
        model = Qwen3OFTForCausalLM(config)
        keys = model.state_dict().keys()
        for key in expected_layer_key_prefixes(1):
            self.assertIn(key, keys)
        wrapper = model.model.layers[1].mlp
        self.assertIsInstance(wrapper, QwenSwiGLUOFTWrapper)
        self.assertEqual(wrapper.custom_ffn_layer_index, 1)
        self.assertTrue(wrapper.fp32_compute)
        self.assertFalse(isinstance(model.model.layers[1].self_attn, QwenSwiGLUOFTWrapper))
        oft_parameters = [
            parameter
            for module in (wrapper.oft_gate, wrapper.oft_up, wrapper.oft_down)
            for parameter in module.oft_like.parameters()
        ]
        self.assertEqual({parameter.dtype for parameter in oft_parameters}, {torch.float32})

    def test_explicit_model_save_reload_and_exact_trainable_partition(self) -> None:
        import torch

        from qwen_single_layer_rl.config import load_config
        from qwen_single_layer_rl.layers import apply_freeze_policy
        from qwen_single_layer_rl.vllm.oft_hf_model import Qwen3OFTConfig, Qwen3OFTForCausalLM

        config = Qwen3OFTConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=11,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            oft_variant={
                "target_layers": [10],
                "target_modules": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
                "block_size": 8,
                "fp32_compute": True,
            },
        )
        model = Qwen3OFTForCausalLM(config)
        cfg = load_config(ROOT / "configs/layer10_whole_layer_oft_fp32.yaml")
        report = apply_freeze_policy(model, cfg)
        names = list(report.trainable_parameter_names)
        self.assertEqual(len(names), 11)
        self.assertEqual(sum(".self_attn." in name for name in names), 6)
        self.assertEqual(sum(name.endswith("layernorm.weight") for name in names), 2)
        self.assertEqual(sum(name.endswith(".oft_like.oft_like") for name in names), 3)
        self.assertFalse(any(".base_mlp." in name for name in names))

        with tempfile.TemporaryDirectory() as directory:
            before = {name: value.detach().clone() for name, value in model.state_dict().items()}
            model.save_pretrained(directory, safe_serialization=True)
            loaded, info = Qwen3OFTForCausalLM.from_pretrained(
                directory,
                torch_dtype=torch.bfloat16,
                output_loading_info=True,
            )
            self.assertFalse(info["missing_keys"])
            self.assertFalse(info["unexpected_keys"])
            self.assertEqual(before.keys(), loaded.state_dict().keys())
            for name, value in loaded.state_dict().items():
                self.assertTrue(torch.equal(before[name].to(value.dtype), value), name)
            loaded_wrapper = loaded.model.layers[10].mlp
            loaded_oft_dtypes = {
                parameter.dtype
                for module in (loaded_wrapper.oft_gate, loaded_wrapper.oft_up, loaded_wrapper.oft_down)
                for parameter in module.oft_like.parameters()
            }
            self.assertEqual(loaded_oft_dtypes, {torch.float32})

    def test_export_metadata_uses_oft_contract(self) -> None:
        from qwen_single_layer_rl.vllm.oft_hf_model import build_oft_export_config

        base = {"model_type": "qwen3", "hidden_size": 2048, "architectures": ["Qwen3ForCausalLM"]}
        params = {"target_layers": [10], "block_size": 64, "fp32_compute": True}
        exported = build_oft_export_config(base, params)
        self.assertEqual(base["model_type"], "qwen3")
        self.assertEqual(exported["model_type"], "qwen3_oft")
        self.assertEqual(exported["oft_variant"], params)
        self.assertEqual(exported["architectures"], ["Qwen3OFTForCausalLM"])

    def test_dispatch_receipt_proves_fp32_oft_path(self) -> None:
        import torch
        import torch.nn as nn

        from qwen_single_layer_rl.model_surgery.qwen_swiglu_variant_modules import QwenSwiGLUOFTWrapper
        from qwen_single_layer_rl.vllm.custom_ffn_contract import validate_dispatch_receipts

        class TinyMLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gate_proj = nn.Linear(8, 16, bias=False)
                self.up_proj = nn.Linear(8, 16, bias=False)
                self.down_proj = nn.Linear(16, 8, bias=False)
                self.act_fn = torch.nn.functional.silu

        wrapper = QwenSwiGLUOFTWrapper(TinyMLP(), {"block_size": 4, "fp32_compute": True})
        wrapper.custom_ffn_layer_index = 10
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "dispatch.jsonl"
            with patch.dict(os.environ, {"OFT_DISPATCH_RECEIPT": str(receipt)}):
                wrapper(torch.randn(1, 2, 8))
                wrapper(torch.randn(1, 2, 8))
            rows = validate_dispatch_receipts(
                receipt,
                variant="qwen_swiglu_oft",
                backend="reference_pytorch_cublas",
                expected_count=1,
            )
        self.assertEqual(rows[0]["layer_index"], 10)
        self.assertEqual(rows[0]["block_size"], 4)
        self.assertTrue(rows[0]["fp32_compute"])
        self.assertEqual(rows[0]["parameter_dtypes"], ["torch.float32"])

    def test_verl_command_allows_exact_noop_oft_vllm_without_overlay(self) -> None:
        from qwen_single_layer_rl.training.verl_command import build_verl_command

        config = ROOT / "configs/runtime/oft_fp32_swigluonly_6x5090_grpo_untunedbase_to196_seed20260707_v1.yaml"
        command, manifest = build_verl_command(
            config,
            project_root=ROOT,
            verl_root=Path("/tmp/verl-v0.6.1-qwenpatch"),
            model_path=Path("/tmp/Qwen3-1.7B-OFT-FP32"),
            data_dir=Path("/tmp/verl_data"),
            run_root=Path("/tmp/runs"),
        )
        self.assertIn("actor_rollout_ref.model.trust_remote_code=True", command)
        self.assertIn("+actor_rollout_ref.rollout.engine_kwargs.vllm.model_impl=auto", command)
        self.assertEqual(manifest["architecture_variant"], "qwen_swiglu_oft")
        self.assertEqual(manifest["env"]["QWEN_SINGLE_LAYER_RL_INITIALIZATION"], "untuned_base_exact_noop")
        self.assertIn("OFT_DISPATCH_RECEIPT", manifest["env"])
        self.assertNotIn("QWEN_SINGLE_LAYER_RL_CHECKPOINT_DIR", manifest["env"])

    def test_plugin_and_entrypoint_use_oft_runtime(self) -> None:
        plugin = (ROOT / "src/qwen_single_layer_rl/vllm/oft_plugin.py").read_text(encoding="utf-8")
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        runtime = (ROOT / "src/qwen_single_layer_rl/vllm/oft_vllm_model.py").read_text(encoding="utf-8")
        self.assertIn("Qwen3OFTTransformersModel", plugin)
        self.assertIn("qwen_single_layer_rl_oft", pyproject)
        self.assertIn("QwenSwiGLUOFTWrapper", runtime)
        self.assertIn("TransformersForCausalLM", runtime)
        self.assertIn("def load_weights", runtime)
        self.assertIn("self._finalize_oft_runtime()", runtime)

    def test_autonomous_controller_keeps_the_approved_order_and_gates(self) -> None:
        controller = (ROOT / "scripts/run_baseline_then_oft_fp32_after_triglu196_20260715_v1.sh").read_text(
            encoding="utf-8"
        )
        baseline = controller.index("for target in 158 196")
        oft_export = controller.index("prepare_oft_initial_model", baseline)
        oft_preflight = controller.index("preflight_oft", oft_export)
        oft_early = controller.index("for target in 1 20 98", oft_preflight)
        oft_late = controller.index("for target in 128 158 196", oft_early)
        self.assertLess(baseline, oft_export)
        self.assertLess(oft_export, oft_preflight)
        self.assertLess(oft_preflight, oft_early)
        self.assertLess(oft_early, oft_late)
        self.assertIn("len(names) != 11", controller)
        self.assertIn("len(oft) != 3 or len(attention) != 6 or len(norms) != 2", controller)
        self.assertIn("third_completed <= 196", controller)
        self.assertIn("data_order_receipt", controller)
        self.assertIn("run_parallel_vllm_eval_6gpu_20260712_v1.sh", controller)

    def test_boundary_watcher_uses_the_training_checkpoint_root(self) -> None:
        watcher = (ROOT / "scripts/autostart_reorder_after_triglu196_20260715_v1.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("CHECKPOINT_ROOT=", watcher)
        self.assertIn('checkpoint="$CHECKPOINT_ROOT/$SOURCE_RUN_ID/checkpoints/global_step_196"', watcher)
        self.assertIn('$SOURCE_ROOT/evaluations/triglu_step_196/PARALLEL_EVAL_COMPLETE', watcher)

    def test_priority_monitor_no_longer_forwards_to_deferred_oft_wave(self) -> None:
        monitor = (ROOT / "scripts/monitor_triglu_priority_to294_then_baseline196_20260715_v1.sh").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("monitor_baseline_then_oft_fp32_after_triglu196_20260715_v1.sh", monitor)
        self.assertIn("for step in 98 128 158 196 226 256 294", monitor)
        self.assertIn('for root in "$R" "$REORDER_ROOT" "$SOURCE_ROOT" "$OLD_ROOT"', monitor)
        self.assertIn('echo "${variant}: pending"', monitor)

    def test_successor_monitor_preserves_all_archived_variant_evaluations(self) -> None:
        monitor = (ROOT / "scripts/monitor_baseline_then_oft_fp32_after_triglu196_20260715_v1.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("PRIORITY=", monitor)
        self.assertIn("OLD=", monitor)
        self.assertIn("for step in 98 128 158 196", monitor)
        self.assertIn("for variant in triglu baseline oft", monitor)
        self.assertNotIn("for step in 1 20 30 60 98 128 158 196", monitor)
        self.assertIn('for root in "$R" "$PRIORITY" "$SOURCE" "$OLD"', monitor)
        self.assertIn('echo "${variant}: pending"', monitor)
        self.assertIn("196/196 archived complete", monitor)


if __name__ == "__main__":
    unittest.main()
