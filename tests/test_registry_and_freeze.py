from __future__ import annotations

import unittest
import tempfile
import json
import importlib.util
from pathlib import Path
from unittest.mock import patch

from qwen_single_layer_rl.data.verl_format import materialize_verl_parquet
from qwen_single_layer_rl.data.prep_numina import main as prep_numina_main
from qwen_single_layer_rl.config import load_config
from qwen_single_layer_rl.layers import apply_freeze_policy, resolve_train_layers
from qwen_single_layer_rl.model_surgery import build_variant, list_variants
from qwen_single_layer_rl.rewards.math_reward import extract_answer
from qwen_single_layer_rl.rewards.verl_math_reward import compute_score
from qwen_single_layer_rl.training.verl_command import build_verl_command

ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = ROOT / "configs" / "smoke_tiny.yaml"
PANDAS_AVAILABLE = importlib.util.find_spec("pandas") is not None
PYARROW_AVAILABLE = importlib.util.find_spec("pyarrow") is not None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class FakeParam:
    def __init__(self) -> None:
        self.requires_grad = False


class FakeModel:
    def __init__(self) -> None:
        self.params = {
            "model.embed_tokens.weight": FakeParam(),
            "model.layers.9.self_attn.q_proj.weight": FakeParam(),
            "model.layers.10.self_attn.q_proj.weight": FakeParam(),
            "model.layers.10.input_layernorm.weight": FakeParam(),
            "model.layers.10.mlp.down_proj.weight": FakeParam(),
            "model.layers.10.mlp.base_mlp.gate_proj.weight": FakeParam(),
            "model.layers.10.mlp.oft_gate.oft_like.oft_like": FakeParam(),
            "model.layers.11.self_attn.q_proj.weight": FakeParam(),
            "model.layers.10.adapters.oft_like.weight": FakeParam(),
            "lm_head.weight": FakeParam(),
        }

    def named_parameters(self):
        return self.params.items()


class RegistryAndFreezeTests(unittest.TestCase):
    def test_registry_has_default_variants(self) -> None:
        self.assertIn("identity", list_variants())
        self.assertIn("oft_like", list_variants())
        self.assertIn("qwen_swiglu_shs", list_variants())
        self.assertIn("qwen_swiglu_triglu_side", list_variants())
        self.assertIn("qwen_swiglu_oft", list_variants())

    def test_smoke_config_resolves_layer10(self) -> None:
        cfg = load_config(SMOKE_CONFIG)
        self.assertEqual(resolve_train_layers(cfg), (10,))
        self.assertEqual(build_variant(cfg).name, "identity")

    def test_freeze_policy_only_keeps_layer10(self) -> None:
        cfg = load_config(SMOKE_CONFIG)
        model = FakeModel()
        report = apply_freeze_policy(model, cfg)
        self.assertEqual(report.train_layers, (10,))
        self.assertTrue(model.params["model.layers.10.self_attn.q_proj.weight"].requires_grad)
        self.assertTrue(model.params["model.layers.10.mlp.down_proj.weight"].requires_grad)
        self.assertFalse(model.params["model.layers.9.self_attn.q_proj.weight"].requires_grad)
        self.assertFalse(model.params["model.layers.11.self_attn.q_proj.weight"].requires_grad)
        self.assertFalse(model.params["model.layers.10.adapters.oft_like.weight"].requires_grad)
        self.assertFalse(model.params["model.embed_tokens.weight"].requires_grad)
        self.assertFalse(model.params["lm_head.weight"].requires_grad)

    def test_adapter_only_keeps_adapter_and_freezes_backbone(self) -> None:
        cfg = load_config(ROOT / "configs" / "training_modes" / "adapter_only_oft_like.yaml")
        model = FakeModel()
        report = apply_freeze_policy(model, cfg)
        self.assertEqual(report.backbone_train_mode, "frozen")
        self.assertEqual(report.train_layers, ())
        self.assertTrue(report.train_adapter_modules)
        self.assertFalse(model.params["model.layers.10.self_attn.q_proj.weight"].requires_grad)
        self.assertTrue(model.params["model.layers.10.adapters.oft_like.weight"].requires_grad)

    def test_full_joint_keeps_backbone_and_adapter(self) -> None:
        cfg = load_config(ROOT / "configs" / "training_modes" / "full_backbone_plus_adapter.yaml")
        model = FakeModel()
        report = apply_freeze_policy(model, cfg)
        self.assertEqual(report.backbone_train_mode, "full")
        self.assertEqual(len(report.train_layers), 28)
        self.assertTrue(model.params["model.layers.9.self_attn.q_proj.weight"].requires_grad)
        self.assertTrue(model.params["model.layers.10.adapters.oft_like.weight"].requires_grad)

    def test_oft_freezes_selected_mlp_base_but_keeps_layer_attention(self) -> None:
        cfg = load_config(ROOT / "configs" / "layer10_whole_layer_oft.yaml")
        model = FakeModel()
        report = apply_freeze_policy(model, cfg)
        self.assertEqual(report.backbone_train_mode, "selected")
        self.assertTrue(report.train_adapter_modules)
        self.assertTrue(model.params["model.layers.10.self_attn.q_proj.weight"].requires_grad)
        self.assertTrue(model.params["model.layers.10.input_layernorm.weight"].requires_grad)
        self.assertFalse(model.params["model.layers.10.mlp.down_proj.weight"].requires_grad)
        self.assertFalse(model.params["model.layers.10.mlp.base_mlp.gate_proj.weight"].requires_grad)
        self.assertTrue(model.params["model.layers.10.mlp.oft_gate.oft_like.oft_like"].requires_grad)

    def test_nested_boxed_answer_extraction(self) -> None:
        self.assertEqual(extract_answer(r"Thus $y=\boxed{\frac{13}{6}}$."), r"\frac{13}{6}")
        self.assertEqual(extract_answer(r"First \boxed{1}, finally \boxed{x+2}."), "x+2")

    def test_local_jsonl_prep_writes_manifest(self) -> None:
        records = [
            {"source": "unit", "problem": f"Problem {idx}", "solution": rf"Answer \boxed{{{idx}}}"}
            for idx in range(5)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "input.jsonl"
            input_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            out_dir = tmp_path / "out"
            import sys

            old_argv = sys.argv
            try:
                sys.argv = [
                    "prep_numina",
                    "--input-jsonl",
                    str(input_path),
                    "--out-dir",
                    str(out_dir),
                    "--target-size",
                    "3",
                    "--seed",
                    "7",
                ]
                prep_numina_main()
            finally:
                sys.argv = old_argv

            manifest = json.loads((out_dir / "prep_manifest.json").read_text(encoding="utf-8"))
            train_rows = [
                json.loads(line)
                for line in (out_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
                if line
            ]
            self.assertEqual(manifest["written_count"], 3)
            self.assertEqual(len(train_rows), 3)
            self.assertIn("answer", train_rows[0])

    @unittest.skipUnless(
        PANDAS_AVAILABLE and PYARROW_AVAILABLE,
        "pandas and pyarrow are required for parquet materializer tests",
    )
    def test_verl_materializer_preserves_order_and_reward_schema(self) -> None:
        records = [
            {"source": "unit", "problem": "Problem A", "answer": "1"},
            {"source": "unit", "problem": "Problem B", "answer": "2"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "input.jsonl"
            output_path = tmp_path / "train.parquet"
            input_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

            manifest = materialize_verl_parquet(input_path, output_path, source_name="unit_math", prompt_suffix="")

            import pandas as pd

            rows = pd.read_parquet(output_path).to_dict(orient="records")
            self.assertEqual(manifest["row_count"], 2)
            self.assertEqual(rows[0]["prompt"][0]["content"], "Problem A")
            self.assertEqual(rows[1]["extra_info"]["index"], 1)
            self.assertEqual(rows[1]["reward_model"]["ground_truth"], "2")

    def test_verl_reward_returns_score_dict(self) -> None:
        with patch(
            "qwen_single_layer_rl.rewards.math_reward._production_math_verify",
            return_value=1.0,
        ):
            score = compute_score("unit_math", r"Therefore \boxed{42}.", "42")
        self.assertEqual(score["score"], 1.0)
        self.assertEqual(score["predicted"], "42")
        self.assertEqual(score["verifier"], "verl_math_verify")

    def test_verl_reward_fails_closed_without_production_verifier(self) -> None:
        with patch.dict("sys.modules", {"verl.utils.reward_score.math_verify": None}):
            with self.assertRaisesRegex(RuntimeError, "string-exact fallback is forbidden"):
                compute_score("unit_math", r"Therefore \boxed{42}.", "42")

    def test_verl_command_uses_hf_rollout_and_seeded_data(self) -> None:
        command, manifest = build_verl_command(
            ROOT / "configs" / "layer10_whole_layer_baseline.yaml",
            project_root=ROOT,
            verl_root=Path("/tmp/verl-v0.6.1-qwenpatch"),
            model_path=Path("/tmp/Qwen3-1.7B-Base"),
            data_dir=Path("/tmp/verl_data"),
            run_root=Path("/tmp/runs"),
        )
        self.assertIn("actor_rollout_ref.rollout.name=hf", command)
        self.assertIn("actor_rollout_ref.rollout.mode=sync", command)
        self.assertIn("actor_rollout_ref.rollout.top_k=0", command)
        self.assertIn("data.seed=20260707", command)
        self.assertIn("actor_rollout_ref.actor.fsdp_config.use_orig_params=True", command)
        self.assertIn("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2", command)
        self.assertIn("actor_rollout_ref.rollout.tensor_model_parallel_size=1", command)
        self.assertIn("actor_rollout_ref.rollout.max_num_seqs=64", command)
        self.assertIn("actor_rollout_ref.rollout.gpu_memory_utilization=0.85", command)
        self.assertIn("actor_rollout_ref.rollout.max_num_batched_tokens=131072", command)
        self.assertNotIn("+actor_rollout_ref.rollout.micro_batch_size=64", command)
        self.assertNotIn("actor_rollout_ref.actor.ppo_micro_batch_size=8", command)
        self.assertEqual(manifest["paper_hyperparams"]["train_batch_size"], 512)
        self.assertEqual(manifest["paper_hyperparams"]["ppo_micro_batch_size_per_gpu"], 2)
        self.assertEqual(manifest["paper_hyperparams"]["hf_rollout_micro_batch_size"], 64)
        self.assertEqual(manifest["paper_hyperparams"]["rollout_tensor_model_parallel_size"], 1)
        self.assertEqual(manifest["paper_hyperparams"]["rollout_max_num_batched_tokens"], 131072)
        self.assertEqual(manifest["env"]["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")

    def test_verl_command_uses_vllm_top_k_semantics_and_single_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "vllm_single_gpu.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        f"inherits: {ROOT / 'configs' / 'layer10_whole_layer_baseline.yaml'}",
                        "runtime:",
                        "  inference_backend: vllm",
                        "  nproc_per_node: 1",
                    ]
                ),
                encoding="utf-8",
            )
            command, manifest = build_verl_command(
                config_path,
                project_root=ROOT,
                verl_root=Path("/tmp/verl-v0.6.1-qwenpatch"),
                model_path=Path("/tmp/Qwen3-1.7B-Base"),
                data_dir=Path("/tmp/verl_data"),
                run_root=Path("/tmp/runs"),
            )

        self.assertIn("actor_rollout_ref.rollout.name=vllm", command)
        self.assertIn("actor_rollout_ref.rollout.top_k=-1", command)
        self.assertIn("trainer.n_gpus_per_node=1", command)
        self.assertIn("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8", command)
        self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", manifest["env"])

    def test_shs_verl_command_requires_checkpoint_and_reference_vllm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "shs_vllm_single_gpu.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        f"inherits: {ROOT / 'configs' / 'layer10_whole_layer_shs.yaml'}",
                        "runtime:",
                        "  inference_backend: vllm",
                        "  nproc_per_node: 1",
                        "grpo:",
                        "  total_training_steps: 1",
                        "  save_freq: 1",
                    ]
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "checkpoint_dir is required"):
                build_verl_command(
                    config_path,
                    project_root=ROOT,
                    verl_root=Path("/tmp/verl-v0.6.1-qwenpatch"),
                    model_path=Path("/tmp/Qwen3-1.7B-Base"),
                    data_dir=Path("/tmp/verl_data"),
                    run_root=Path("/tmp/runs"),
                )
            checkpoint = tmp_path / "checkpoint"
            command, manifest = build_verl_command(
                config_path,
                project_root=ROOT,
                verl_root=Path("/tmp/verl-v0.6.1-qwenpatch"),
                model_path=Path("/tmp/shs_deployment_export"),
                data_dir=Path("/tmp/verl_data"),
                run_root=Path("/tmp/runs"),
                checkpoint_dir=checkpoint,
            )

        self.assertIn("actor_rollout_ref.rollout.enforce_eager=True", command)
        self.assertIn("+actor_rollout_ref.rollout.engine_kwargs.vllm.model_impl=transformers", command)
        self.assertIn("trainer.total_training_steps=1", command)
        self.assertIn("trainer.save_freq=1", command)
        self.assertEqual(manifest["env"]["SHS_INFERENCE_MUL_BACKEND"], "reference")
        self.assertEqual(manifest["env"]["QWEN_SINGLE_LAYER_RL_CHECKPOINT_DIR"], str(checkpoint.resolve()))
        self.assertEqual(manifest["paper_hyperparams"]["total_training_steps"], 1)

    def test_production_shard_uses_four_replica_equivalent_micro_batch(self) -> None:
        command, manifest = build_verl_command(
            ROOT / "configs/runtime/shs_grpo_replica_shard_20260712_v2_realverl.yaml",
            project_root=ROOT,
            verl_root=Path("/tmp/verl-v0.6.1-qwenpatch"),
            model_path=Path("/tmp/shs_deployment_export"),
            data_dir=Path("/tmp/verl_data"),
            run_root=Path("/tmp/runs"),
            checkpoint_dir=Path("/tmp/checkpoint"),
        )
        self.assertIn("data.train_batch_size=128", command)
        self.assertIn("actor_rollout_ref.rollout.n=4", command)
        self.assertIn("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2", command)
        self.assertEqual(manifest["paper_hyperparams"]["ppo_micro_batch_size"], 2)

    def test_two_gpu_weight_sync_gate_has_restart_and_checkpoint_compatible_dataloader(self) -> None:
        command, manifest = build_verl_command(
            ROOT / "configs/runtime/shs_2x5090_actor_rollout_weight_sync_20260712_v1.yaml",
            project_root=ROOT,
            verl_root=Path("/tmp/verl-v0.6.1-qwenpatch"),
            model_path=Path("/tmp/shs_deployment_export"),
            data_dir=Path("/tmp/verl_data"),
            run_root=Path("/tmp/runs"),
            checkpoint_dir=Path("/tmp/checkpoint"),
        )
        self.assertIn("trainer.n_gpus_per_node=2", command)
        self.assertIn("trainer.total_training_steps=11", command)
        self.assertIn("data.dataloader_num_workers=8", command)
        self.assertIn("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2", command)
        self.assertEqual(manifest["paper_hyperparams"]["dataloader_num_workers"], 8)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is required for module-shape tests")
    def test_shs_delta_linear_has_no_scalar_parameters(self) -> None:
        import torch.nn as nn
        from qwen_single_layer_rl.model_surgery.qwen_swiglu_variant_modules import ShuffledHyperGridDeltaLinear

        module = ShuffledHyperGridDeltaLinear(
            nn.Linear(8, 8, bias=False),
            rows=4,
            cols=4,
            shuffle_seed=123,
            mul_init_scale=0.001,
            add_init_scale=0.001,
            add_rank=2,
        )

        scalar_params = [name for name, param in module.named_parameters() if param.dim() == 0]
        self.assertEqual(scalar_params, [])
        self.assertEqual(tuple(module.mul_scale.shape), (1,))
        self.assertEqual(tuple(module.add_scale.shape), (1,))


if __name__ == "__main__":
    unittest.main()
