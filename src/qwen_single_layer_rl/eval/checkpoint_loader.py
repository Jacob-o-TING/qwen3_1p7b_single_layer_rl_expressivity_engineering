from __future__ import annotations

from pathlib import Path
from typing import Any

from qwen_single_layer_rl.config import load_config
from qwen_single_layer_rl.model_surgery import build_variant
from qwen_single_layer_rl.seeding import seed_everything
from qwen_single_layer_rl.sft.checkpoint import load_trainable_state_dict


def load_sft_checkpoint_for_inference(
    *,
    config_path: Path,
    checkpoint_dir: Path | None,
    model_path: Path | None = None,
    device: str = "cuda",
    base_model_only: bool = False,
) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = load_config(config_path)
    model_cfg = cfg.get("model", {})
    configured_path = Path(str(model_cfg.get("local_path", model_cfg.get("name_or_path"))))
    resolved_model_path = model_path or configured_path
    if not resolved_model_path.is_absolute() and not resolved_model_path.exists():
        candidate = config_path.resolve().parents[2] / resolved_model_path
        if candidate.exists():
            resolved_model_path = candidate
    if not resolved_model_path.exists():
        resolved_model_path = Path(str(model_cfg.get("name_or_path")))

    tokenizer = AutoTokenizer.from_pretrained(
        str(resolved_model_path),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(resolved_model_path),
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )
    if not base_model_only:
        if checkpoint_dir is None:
            raise ValueError("checkpoint_dir is required unless base_model_only is true")
        seed = int(
            cfg.get("experiment", {}).get(
                "init_seed", cfg.get("experiment", {}).get("seed", 0)
            )
        )
        seed_everything(seed)
        model = build_variant(cfg).apply(model, cfg)
        trainable_state = torch.load(
            checkpoint_dir / "trainable_state.pt",
            map_location="cpu",
            weights_only=False,
        )
        load_trainable_state_dict(model, trainable_state)
    model.to(device)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer, cfg
