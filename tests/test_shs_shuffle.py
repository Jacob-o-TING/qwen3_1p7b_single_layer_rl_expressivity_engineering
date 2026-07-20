from __future__ import annotations

import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from qwen_single_layer_rl.model_surgery.qwen_swiglu_variant_modules import (
        QwenSwiGLUSHSWrapper,
        SHS_DELTA_SEED_OFFSETS,
        ShuffledHyperGridDeltaLinear,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for SHS shuffle tests")
class SHSShuffleTests(unittest.TestCase):
    def test_mul_and_add_shuffle_maps_are_separate_and_reproducible(self) -> None:
        module_a = ShuffledHyperGridDeltaLinear(
            nn.Linear(16, 16, bias=False),
            rows=4,
            cols=4,
            shuffle_seed=1234,
            mul_init_scale=0.001,
            add_init_scale=0.001,
            add_rank=4,
        )
        module_b = ShuffledHyperGridDeltaLinear(
            nn.Linear(16, 16, bias=False),
            rows=4,
            cols=4,
            shuffle_seed=1234,
            mul_init_scale=0.001,
            add_init_scale=0.001,
            add_rank=4,
        )

        self.assertEqual(module_a.shuffle_seed_offsets, SHS_DELTA_SEED_OFFSETS)
        for name in ("mul_row_block_ids", "mul_col_block_ids", "add_row_block_ids", "add_col_block_ids"):
            self.assertTrue(torch.equal(getattr(module_a, name), getattr(module_b, name)))

        self.assertFalse(torch.equal(module_a.mul_row_block_ids, module_a.add_row_block_ids))
        self.assertFalse(torch.equal(module_a.mul_col_block_ids, module_a.add_col_block_ids))
        for col_id in range(module_a.cols):
            mul_indices = getattr(module_a, module_a._mul_col_index_names[col_id])
            add_indices = getattr(module_a, module_a._add_col_index_names[col_id])
            self.assertTrue(torch.all(module_a.mul_col_block_ids.index_select(0, mul_indices) == col_id))
            self.assertTrue(torch.all(module_a.add_col_block_ids.index_select(0, add_indices) == col_id))

        persistent_state = module_a.state_dict()
        self.assertIn("mul_col_block_ids", persistent_state)
        self.assertIn("add_col_block_ids", persistent_state)
        self.assertNotIn("_mul_col_indices_0", persistent_state)
        self.assertNotIn("_add_col_indices_0", persistent_state)

    def test_projection_shuffle_maps_remain_reproducible(self) -> None:
        wrapper_a = QwenSwiGLUSHSWrapper(FakeQwenMLP(16, 32), 16, _tiny_shs_params(seed=777))
        wrapper_b = QwenSwiGLUSHSWrapper(FakeQwenMLP(16, 32), 16, _tiny_shs_params(seed=777))

        for projection in ("gate", "up", "down"):
            for name in ("mul_row_block_ids", "mul_col_block_ids", "add_row_block_ids", "add_col_block_ids"):
                self.assertTrue(torch.equal(getattr(wrapper_a.shs[projection], name), getattr(wrapper_b.shs[projection], name)))

        self.assertFalse(torch.equal(wrapper_a.shs["gate"].mul_col_block_ids, wrapper_a.shs["up"].mul_col_block_ids))
        state_keys = set(wrapper_a.state_dict())
        self.assertIn("base_mlp.gate_proj.weight", state_keys)
        self.assertNotIn("shs.gate.base_linear.weight", state_keys)

    def test_zero_grid_generator_keeps_exact_base_swiglu_function(self) -> None:
        torch.manual_seed(20260707)
        base_mlp = FakeQwenMLP(16, 32)
        wrapper = QwenSwiGLUSHSWrapper(base_mlp, 16, _tiny_shs_params(seed=20260707))
        x = torch.randn(2, 3, 16)

        with torch.no_grad():
            expected = base_mlp.down_proj(F.silu(base_mlp.gate_proj(x)) * base_mlp.up_proj(x))
            actual = wrapper(x)

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


if TORCH_AVAILABLE:

    class FakeQwenMLP(nn.Module):
        def __init__(self, d_model: int, intermediate: int) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(d_model, intermediate, bias=False)
            self.up_proj = nn.Linear(d_model, intermediate, bias=False)
            self.down_proj = nn.Linear(intermediate, d_model, bias=False)
            self.act_fn = F.silu


def _tiny_shs_params(seed: int) -> dict[str, object]:
    return {
        "hypergrid_rows": 4,
        "hypergrid_cols": 4,
        "hypergrid_generator_hidden": 8,
        "hypergrid_add_rank": 4,
        "hypergrid_shuffle_seed": seed,
        "hypergrid_mul_init_scale": 0.001,
        "hypergrid_add_init_scale": 0.001,
    }


if __name__ == "__main__":
    unittest.main()
