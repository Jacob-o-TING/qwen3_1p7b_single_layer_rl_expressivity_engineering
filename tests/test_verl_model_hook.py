from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("torch is required for veRL model-hook tests")

import torch
import torch.nn as nn

from qwen_single_layer_rl.layers import apply_freeze_policy
from qwen_single_layer_rl.training.verl_model_hook import apply_model_surgery_before_fsdp


class _PreconstructedSHSMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_mlp = nn.Linear(2, 2, bias=False)
        self.shs = nn.ModuleDict(
            {
                "grid_generator": nn.Linear(2, 2, bias=False),
                "gate": _Delta(),
                "up": _Delta(),
                "down": _Delta(),
            }
        )


class _Delta(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        for name in ("mul_row_block_ids", "mul_col_block_ids", "add_row_block_ids", "add_col_block_ids"):
            self.register_buffer(name, torch.arange(2), persistent=True)


class _Layer(nn.Module):
    def __init__(self, shs: bool) -> None:
        super().__init__()
        self.mlp = _PreconstructedSHSMLP() if shs else nn.Linear(2, 2, bias=False)


class _Inner(nn.Module):
    def __init__(self, shs_layers: tuple[bool, ...]) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer(shs) for shs in shs_layers])


class _Model(nn.Module):
    def __init__(self, shs_layers: tuple[bool, ...]) -> None:
        super().__init__()
        self.model = _Inner(shs_layers)


def _config(target_layers: list[int]) -> dict:
    return {
        "experiment": {"seed": 17, "init_seed": 17},
        "model": {"num_layers": len(target_layers)},
        "architecture_variant": {
            "name": "qwen_swiglu_shs",
            "params": {"target_layers": target_layers},
        },
        "freeze_policy": {
            "backbone_train_mode": "selected",
            "train_layers": target_layers,
            "train_adapter_modules": True,
            "freeze_embeddings": True,
            "freeze_lm_head": True,
        },
    }


class _Variant:
    name = "qwen_swiglu_shs"

    def apply(self, model, cfg):
        raise AssertionError("preconstructed SHS export must not be injected again")


class VerlModelHookTests(unittest.TestCase):
    def test_preconstructed_shs_overlay_is_exactly_once_and_audited(self) -> None:
        cfg = _config([0])
        model = _Model((True,))
        report = apply_freeze_policy(model, cfg)
        state = {
            name: torch.full_like(parameter, 0.25)
            for name, parameter in model.named_parameters()
            if name in report.trainable_parameter_names
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            torch.save(state, checkpoint / "trainable_state.pt")
            audit = root / "audit"
            env = {
                "QWEN_SINGLE_LAYER_RL_CONFIG": str(root / "config.yaml"),
                "QWEN_SINGLE_LAYER_RL_CHECKPOINT_DIR": str(checkpoint),
                "QWEN_SINGLE_LAYER_RL_AUDIT_DIR": str(audit),
            }
            with patch.dict("os.environ", env, clear=False), patch(
                "qwen_single_layer_rl.training.verl_model_hook.load_config", return_value=cfg
            ), patch(
                "qwen_single_layer_rl.training.verl_model_hook.build_variant", return_value=_Variant()
            ):
                result = apply_model_surgery_before_fsdp(model, role="actor", rank=0)
                with self.assertRaisesRegex(RuntimeError, "invoked twice"):
                    apply_model_surgery_before_fsdp(model, role="actor", rank=0)

            payload = json.loads((audit / "actor_rank0_model_surgery_audit.json").read_text())

        self.assertIs(result, model)
        self.assertEqual(model._qwen_single_layer_checkpoint_overlay_count, 1)
        self.assertEqual(payload["construction_mode"], "preconstructed")
        self.assertEqual(payload["checkpoint_overlay_count"], 1)
        self.assertEqual(payload["excluded_deterministic_sync_buffer_count"], 12)
        self.assertEqual(set(payload["checkpoint_overlay_keys"]), set(report.trainable_parameter_names))
        self.assertFalse(any(name.endswith("block_ids") for name in model.state_dict()))
        for name, parameter in model.named_parameters():
            if name in state:
                self.assertTrue(torch.equal(parameter, state[name]))

    def test_partial_shs_construction_is_fatal(self) -> None:
        cfg = _config([0, 1])
        model = _Model((True, False))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            "os.environ", {"QWEN_SINGLE_LAYER_RL_CONFIG": str(Path(tmp) / "config.yaml")}, clear=False
        ), patch(
            "qwen_single_layer_rl.training.verl_model_hook.load_config", return_value=cfg
        ), patch(
            "qwen_single_layer_rl.training.verl_model_hook.build_variant", return_value=_Variant()
        ):
            with self.assertRaisesRegex(RuntimeError, "Partial SHS construction"):
                apply_model_surgery_before_fsdp(model, role="actor", rank=0)


if __name__ == "__main__":
    unittest.main()
