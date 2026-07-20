from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None
ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(TRANSFORMERS_AVAILABLE, "transformers is required")
class VLLMTriGLUScaffoldTests(unittest.TestCase):
    def test_explicit_model_preserves_runtime_surgery_key_and_precision_contract(self) -> None:
        import torch

        from qwen_single_layer_rl.vllm.triglu_hf_model import (
            Qwen3TriGLUConfig,
            Qwen3TriGLUForCausalLM,
            expected_layer_key_prefixes,
        )

        config = Qwen3TriGLUConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            triglu_variant={"target_layers": [1], "side_dim": 8, "side_hidden": 12, "side_scale": 0.1},
        )
        model = Qwen3TriGLUForCausalLM(config)
        keys = model.state_dict().keys()
        for key in expected_layer_key_prefixes(1):
            self.assertIn(key, keys)
        self.assertNotIn("model.layers.1.mlp.gate_proj.weight", keys)
        wrapper = model.model.layers[1].mlp
        self.assertEqual(wrapper.custom_ffn_layer_index, 1)
        self.assertEqual({parameter.dtype for parameter in wrapper.triglu_side.parameters()}, {torch.float32})

    def test_exact_noop_initialization_matches_base_mlp(self) -> None:
        import copy
        import torch
        import torch.nn as nn

        from qwen_single_layer_rl.model_surgery.qwen_swiglu_variant_modules import (
            QwenSwiGLUTriGLUSideWrapper,
        )

        class TinyMLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gate_proj = nn.Linear(8, 16, bias=False)
                self.up_proj = nn.Linear(8, 16, bias=False)
                self.down_proj = nn.Linear(16, 8, bias=False)
                self.act_fn = torch.nn.functional.silu

            def forward(self, x):
                return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        base = TinyMLP()
        reference = copy.deepcopy(base)
        wrapper = QwenSwiGLUTriGLUSideWrapper(
            base, 8, 16, {"side_dim": 4, "side_hidden": 6, "side_scale": 0.1}
        )
        x = torch.randn(2, 3, 8)
        self.assertTrue(torch.equal(reference(x), wrapper(x)))

    def test_training_reference_path_keeps_base_and_side_gradients(self) -> None:
        import torch
        import torch.nn as nn

        from qwen_single_layer_rl.model_surgery.qwen_swiglu_variant_modules import (
            QwenSwiGLUTriGLUSideWrapper,
        )

        class TinyMLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gate_proj = nn.Linear(8, 16, bias=False)
                self.up_proj = nn.Linear(8, 16, bias=False)
                self.down_proj = nn.Linear(16, 8, bias=False)
                self.act_fn = torch.nn.functional.silu

        wrapper = QwenSwiGLUTriGLUSideWrapper(
            TinyMLP(), 8, 16, {"side_dim": 4, "side_hidden": 6, "side_scale": 0.1}
        ).train()
        loss = wrapper(torch.randn(2, 3, 8)).square().mean()
        loss.backward()
        base_gradient = wrapper.base_mlp.gate_proj.weight.grad
        side_gradient = wrapper.triglu_side["up"].weight.grad
        self.assertIsNotNone(base_gradient)
        self.assertIsNotNone(side_gradient)
        self.assertTrue(torch.isfinite(base_gradient).all())
        self.assertTrue(torch.isfinite(side_gradient).all())
        self.assertGreater(float(base_gradient.abs().sum()), 0.0)
        self.assertGreater(float(side_gradient.abs().sum()), 0.0)

    def test_export_metadata_uses_architecture_neutral_contract(self) -> None:
        from qwen_single_layer_rl.vllm.triglu_hf_model import build_triglu_export_config

        base = {"model_type": "qwen3", "hidden_size": 2048, "architectures": ["Qwen3ForCausalLM"]}
        params = {"target_layers": [10], "side_dim": 512, "side_hidden": 2048}
        exported = build_triglu_export_config(base, params)
        self.assertEqual(base["model_type"], "qwen3")
        self.assertEqual(exported["model_type"], "qwen3_triglu")
        self.assertEqual(exported["triglu_variant"], params)
        self.assertEqual(exported["architectures"], ["Qwen3TriGLUForCausalLM"])

    def test_dispatch_receipt_proves_custom_path(self) -> None:
        import torch
        import torch.nn as nn

        from qwen_single_layer_rl.model_surgery.qwen_swiglu_variant_modules import (
            QwenSwiGLUTriGLUSideWrapper,
        )
        from qwen_single_layer_rl.vllm.custom_ffn_contract import validate_dispatch_receipts

        class TinyMLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gate_proj = nn.Linear(8, 16, bias=False)
                self.up_proj = nn.Linear(8, 16, bias=False)
                self.down_proj = nn.Linear(16, 8, bias=False)
                self.act_fn = torch.nn.functional.silu

        wrapper = QwenSwiGLUTriGLUSideWrapper(
            TinyMLP(), 8, 16, {"side_dim": 4, "side_hidden": 6, "side_scale": 0.1}
        )
        wrapper.custom_ffn_layer_index = 10
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "dispatch.jsonl"
            with patch.dict(os.environ, {"TRIGLU_DISPATCH_RECEIPT": str(receipt)}):
                wrapper(torch.randn(1, 2, 8))
                wrapper(torch.randn(1, 2, 8))
            rows = validate_dispatch_receipts(
                receipt,
                variant="qwen_swiglu_triglu_side",
                backend="reference_pytorch_cublas",
                expected_count=1,
            )
        self.assertEqual(rows[0]["layer_index"], 10)
        self.assertEqual(rows[0]["side_dim"], 4)
        self.assertFalse(rows[0]["fallback"])

    def test_plugin_and_entrypoint_use_triglu_runtime(self) -> None:
        plugin = (ROOT / "src/qwen_single_layer_rl/vllm/triglu_plugin.py").read_text(encoding="utf-8")
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        runtime = (ROOT / "src/qwen_single_layer_rl/vllm/triglu_vllm_model.py").read_text(encoding="utf-8")
        variant = (ROOT / "src/qwen_single_layer_rl/model_surgery/qwen_swiglu_variant_modules.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("Qwen3TriGLUTransformersModel", plugin)
        self.assertIn("qwen_single_layer_rl_triglu", pyproject)
        self.assertIn("QwenSwiGLUTriGLUSideWrapper", runtime)
        self.assertIn("TransformersForCausalLM", runtime)
        self.assertIn("reference_pytorch_cublas", runtime)
        self.assertIn("def load_weights", runtime)
        self.assertIn("self._finalize_triglu_runtime()", runtime)
        self.assertIn("Emit the runtime receipt before vLLM compiles its dummy forward", runtime)
        self.assertIn("write_dispatch_receipt_once(", runtime)
        self.assertIn('if not getattr(self, "_custom_ffn_dispatch_receipt_written", False):', variant)
        self.assertNotIn('triglu_side["down"].out_features', variant)


if __name__ == "__main__":
    unittest.main()
