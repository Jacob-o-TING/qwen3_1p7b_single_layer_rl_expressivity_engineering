from __future__ import annotations

import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    import torch.nn.functional as F

    from qwen_single_layer_rl.kernels.shs_modulated_projection import (
        shs_grouped_multiplicative_delta_reference,
        shs_grouped_multiplicative_delta_triton_reference_recompute,
        shs_modulated_projection,
        shs_multiplicative_delta_triton,
        shs_modulated_projection_reference,
    )
    from qwen_single_layer_rl.model_surgery.qwen_swiglu_variant_modules import make_shuffled_block_ids
    from qwen_single_layer_rl.model_surgery.qwen_swiglu_variant_modules import QwenSwiGLUSHSWrapper
    from qwen_single_layer_rl.model_surgery.qwen_swiglu_variant_modules import ShuffledHyperGridDeltaLinear

CUDA_AVAILABLE = TORCH_AVAILABLE and torch.cuda.is_available()


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required")
class SHSModulatedProjectionTests(unittest.TestCase):
    def test_reference_matches_dense_equation_with_odd_tails(self) -> None:
        torch.manual_seed(20260711)
        m, n, k, rows, cols = 3, 19, 37, 4, 5
        x = torch.randn(m, k)
        weight = torch.randn(n, k)
        grid = torch.randn(m, rows, cols)
        row_ids = make_shuffled_block_ids(n, rows, 101)
        col_ids = make_shuffled_block_ids(k, cols, 202)
        scale = 0.03125
        expected_weight = weight[None] * (
            1 + scale * torch.tanh(grid[:, row_ids[:, None], col_ids[None, :]])
        )
        expected = torch.einsum("mk,mnk->mn", x, expected_weight)
        actual = shs_modulated_projection_reference(x, weight, grid, row_ids, col_ids, scale)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    def test_reference_zero_grid_and_scale_preserve_noop(self) -> None:
        torch.manual_seed(20260711)
        x = torch.randn(2, 17)
        weight = torch.randn(23, 17)
        row_ids = make_shuffled_block_ids(23, 4, 303)
        col_ids = make_shuffled_block_ids(17, 4, 404)
        zero_grid = torch.zeros(2, 4, 4)
        expected = F.linear(x, weight)
        by_grid = shs_modulated_projection_reference(x, weight, zero_grid, row_ids, col_ids, 0.1)
        by_scale = shs_modulated_projection_reference(x, weight, torch.randn_like(zero_grid), row_ids, col_ids, 0.0)
        torch.testing.assert_close(by_grid, expected, rtol=0.0, atol=0.0)
        torch.testing.assert_close(by_scale, expected, rtol=0.0, atol=0.0)

    def test_auto_fallback_is_reported(self) -> None:
        x = torch.randn(2, 7)
        weight = torch.randn(5, 7)
        grid = torch.randn(2, 2, 2)
        row_ids = make_shuffled_block_ids(5, 2, 1)
        col_ids = make_shuffled_block_ids(7, 2, 2)
        result = shs_modulated_projection(x, weight, grid, row_ids, col_ids, 0.1, backend="auto")
        self.assertEqual(result.backend, "reference")
        self.assertIn("CUDA", result.fallback_reason or "")

    def test_grouped_reference_propagates_all_gradients(self) -> None:
        torch.manual_seed(20260712)
        x = torch.randn(2, 7, requires_grad=True)
        weight = torch.randn(5, 7, requires_grad=True)
        grid = torch.randn(2, 2, 3, requires_grad=True)
        row_ids = make_shuffled_block_ids(5, 2, 55)
        col_ids = make_shuffled_block_ids(7, 3, 66)
        groups = [torch.nonzero(col_ids == col, as_tuple=True)[0] for col in range(3)]
        permutation = torch.cat(groups)
        offsets = torch.tensor([0, *list(__import__("itertools").accumulate(group.numel() for group in groups))])
        output = shs_grouped_multiplicative_delta_reference(
            x, weight, grid, row_ids, permutation, offsets
        )
        output.square().mean().backward()
        for tensor in (x, weight, grid):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_explicit_triton_never_silently_falls_back(self) -> None:
        with self.assertRaises(RuntimeError):
            shs_modulated_projection(
                torch.randn(1, 3),
                torch.randn(2, 3),
                torch.randn(1, 2, 2),
                torch.tensor([0, 1]),
                torch.tensor([0, 1, 0]),
                0.1,
                backend="triton",
            )

    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA is required for Triton integration")
    def test_module_opt_in_replaces_only_multiplicative_path(self) -> None:
        torch.manual_seed(20260711)
        base = torch.nn.Linear(37, 19, bias=False, device="cuda", dtype=torch.bfloat16)
        module = ShuffledHyperGridDeltaLinear(
            base,
            rows=4,
            cols=5,
            shuffle_seed=777,
            mul_init_scale=0.03125,
            add_init_scale=0.015625,
            add_rank=7,
        ).cuda().to(torch.bfloat16).eval()
        x = torch.randn(3, 37, device="cuda", dtype=torch.bfloat16)
        mul_grid = torch.randn(3, 4, 5, device="cuda", dtype=torch.bfloat16)
        add_grid = torch.randn(3, 4, 5, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            expected = module(x, mul_grid, add_grid)
            module.set_inference_mul_backend("triton")
            actual = module(x, mul_grid, add_grid)
        self.assertEqual(module.last_inference_mul_backend, "triton")
        self.assertFalse(torch.equal(module.mul_row_block_ids, module.add_row_block_ids))
        self.assertFalse(torch.equal(module.mul_col_block_ids, module.add_col_block_ids))
        torch.testing.assert_close(actual, expected, rtol=3e-2, atol=8e-2)

    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA is required for Triton delta")
    def test_delta_kernel_matches_reference_delta(self) -> None:
        torch.manual_seed(20260712)
        x = torch.randn(3, 37, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(19, 37, device="cuda", dtype=torch.bfloat16)
        grid = torch.randn(3, 4, 5, device="cuda", dtype=torch.bfloat16)
        row_ids = make_shuffled_block_ids(19, 4, 55).cuda()
        col_ids = make_shuffled_block_ids(37, 5, 66).cuda()
        expected = (
            shs_modulated_projection_reference(x, weight, grid, row_ids, col_ids, 1.0)
            - torch.nn.functional.linear(x, weight)
        )
        with torch.no_grad():
            actual = shs_multiplicative_delta_triton(x, weight, grid, row_ids, col_ids)
        torch.testing.assert_close(actual, expected, rtol=3e-2, atol=8e-2)

    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA is required for Triton autograd")
    def test_triton_forward_reference_recompute_backward_matches_reference(self) -> None:
        torch.manual_seed(20260712)
        row_ids = make_shuffled_block_ids(11, 3, 55).cuda()
        col_ids = make_shuffled_block_ids(13, 4, 66).cuda()
        groups = [torch.nonzero(col_ids == col, as_tuple=True)[0] for col in range(4)]
        permutation = torch.cat(groups)
        offsets = torch.tensor(
            [0, *list(__import__("itertools").accumulate(group.numel() for group in groups))], device="cuda"
        )
        base = (
            torch.randn(2, 13, device="cuda", dtype=torch.bfloat16),
            torch.randn(11, 13, device="cuda", dtype=torch.bfloat16),
            torch.randn(2, 3, 4, device="cuda", dtype=torch.bfloat16),
        )
        reference_inputs = tuple(t.detach().clone().requires_grad_(True) for t in base)
        triton_inputs = tuple(t.detach().clone().requires_grad_(True) for t in base)
        grad_output = torch.randn(2, 11, device="cuda", dtype=torch.bfloat16)
        expected = shs_grouped_multiplicative_delta_reference(
            *reference_inputs, row_ids, permutation, offsets
        )
        expected.backward(grad_output)
        actual = shs_grouped_multiplicative_delta_triton_reference_recompute(
            *triton_inputs, row_ids, permutation, offsets
        )
        actual.backward(grad_output)
        torch.testing.assert_close(actual, expected, rtol=3e-2, atol=8e-2)
        for actual_input, expected_input in zip(triton_inputs, reference_inputs):
            torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=0.0, atol=0.0)

    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA is required for full SHS autograd")
    def test_full_shs_wrapper_triton_recompute_covers_every_parameter_group(self) -> None:
        class TinyMLP(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gate_proj = torch.nn.Linear(13, 17, bias=False)
                self.up_proj = torch.nn.Linear(13, 17, bias=False)
                self.down_proj = torch.nn.Linear(17, 13, bias=False)
                self.act_fn = F.silu

        params = {
            "hypergrid_rows": 3,
            "hypergrid_cols": 4,
            "hypergrid_generator_hidden": 11,
            "hypergrid_add_rank": 5,
            "hypergrid_shuffle_seed": 20260712,
            "hypergrid_mul_init_scale": 0.03125,
            "hypergrid_add_init_scale": 0.015625,
        }
        torch.manual_seed(20260712)
        reference = QwenSwiGLUSHSWrapper(TinyMLP(), 13, params).cuda().to(torch.bfloat16).train()
        torch.manual_seed(20260712)
        actual = QwenSwiGLUSHSWrapper(TinyMLP(), 13, params).cuda().to(torch.bfloat16).train()
        actual.load_state_dict(reference.state_dict())
        for projection in ("gate", "up", "down"):
            actual.shs[projection].set_inference_mul_backend("triton_reference_recompute")

        x_reference = torch.randn(2, 13, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        x_actual = x_reference.detach().clone().requires_grad_(True)
        grad_output = torch.randn(2, 13, device="cuda", dtype=torch.bfloat16)
        expected_output = reference(x_reference)
        actual_output = actual(x_actual)
        expected_output.backward(grad_output)
        actual_output.backward(grad_output)

        torch.testing.assert_close(actual_output, expected_output, rtol=3e-2, atol=8e-2)
        torch.testing.assert_close(x_actual.grad, x_reference.grad, rtol=3e-2, atol=8e-2)
        reference_parameters = dict(reference.named_parameters())
        expected_groups = {"base_mlp", "grid_generator", "mul_scale", "add_scale", "add_left", "add_right"}
        covered_groups = set()
        for name, parameter in actual.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            torch.testing.assert_close(
                parameter.grad,
                reference_parameters[name].grad,
                rtol=3e-2,
                atol=8e-2,
                msg=lambda message, parameter_name=name: f"{parameter_name}: {message}",
            )
            for group in expected_groups:
                if group in name:
                    covered_groups.add(group)
        self.assertEqual(covered_groups, expected_groups)
        self.assertEqual(
            [module.last_inference_mul_backend for module in actual.shs.values() if isinstance(module, ShuffledHyperGridDeltaLinear)],
            ["triton_forward_reference_recompute_backward"] * 3,
        )


if __name__ == "__main__":
    unittest.main()
