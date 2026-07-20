from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None
ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(TRANSFORMERS_AVAILABLE, "transformers is required")
class VLLMSHSScaffoldTests(unittest.TestCase):
    def test_explicit_model_preserves_runtime_surgery_key_contract(self) -> None:
        import torch

        from qwen_single_layer_rl.vllm.shs_hf_model import (
            Qwen3SHSConfig,
            Qwen3SHSForCausalLM,
            expected_layer10_key_prefixes,
        )

        config = Qwen3SHSConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            shs_variant={
                "target_layers": [1],
                "hypergrid_rows": 4,
                "hypergrid_cols": 4,
                "hypergrid_generator_hidden": 8,
                "hypergrid_add_rank": 4,
                "hypergrid_shuffle_seed": 20260707,
            },
        )
        model = Qwen3SHSForCausalLM(config)
        keys = model.state_dict().keys()
        for key in expected_layer10_key_prefixes(1):
            self.assertIn(key, keys)
        self.assertNotIn("model.layers.1.mlp.gate_proj.weight", keys)
        self.assertEqual(model.model.layers[1].mlp.base_mlp.gate_proj.weight.dtype, model.dtype)
        self.assertEqual(model.model.layers[1].mlp.shs["grid_generator"].up.weight.dtype, torch.float32)

    def test_export_config_changes_metadata_only(self) -> None:
        from qwen_single_layer_rl.vllm.shs_hf_model import build_shs_export_config

        base = {"model_type": "qwen3", "hidden_size": 2048, "architectures": ["Qwen3ForCausalLM"]}
        params = {"target_layers": [10], "hypergrid_rows": 32, "hypergrid_cols": 32}
        exported = build_shs_export_config(base, params)
        self.assertEqual(base["model_type"], "qwen3")
        self.assertEqual(exported["model_type"], "qwen3_shs")
        self.assertEqual(exported["shs_variant"], params)

    def test_plugin_uses_shs_aware_vllm_wrapper(self) -> None:
        source = (ROOT / "src/qwen_single_layer_rl/vllm/shs_plugin.py").read_text(encoding="utf-8")
        self.assertIn("Qwen3SHSTransformersModel", source)

    def test_vllm_wrapper_requires_explicit_backend_without_hardcoded_triton(self) -> None:
        source = (ROOT / "src/qwen_single_layer_rl/vllm/shs_vllm_model.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('os.environ.get("SHS_INFERENCE_MUL_BACKEND")', source)
        self.assertIn("module.set_inference_mul_backend(requested_backend)", source)
        self.assertNotIn('module.set_inference_mul_backend("triton")', source)


if __name__ == "__main__":
    unittest.main()
