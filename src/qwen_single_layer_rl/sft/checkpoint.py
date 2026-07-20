from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any


def unwrap_model(model: Any) -> Any:
    model = getattr(model, "module", model)
    return getattr(model, "_orig_mod", model)


def trainable_state_dict(model: Any) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in unwrap_model(model).named_parameters()
        if parameter.requires_grad
    }


def load_trainable_state_dict(model: Any, state: dict[str, Any]) -> None:
    named = dict(unwrap_model(model).named_parameters())
    missing = sorted(set(state) - set(named))
    if missing:
        raise KeyError(f"Checkpoint contains unknown trainable parameters: {missing[:20]}")
    for name, tensor in state.items():
        parameter = named[name]
        parameter.data.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))


def capture_rng_state() -> dict[str, Any]:
    import torch

    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np  # type: ignore

        state["numpy"] = np.random.get_state()
    except ModuleNotFoundError:
        pass
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    import torch

    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if "numpy" in state:
        import numpy as np  # type: ignore

        np.random.set_state(state["numpy"])


def _optimizer_to_device(optimizer: Any, device: Any) -> None:
    import torch

    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def save_checkpoint(
    checkpoint_root: Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    trainer_state: dict[str, Any],
    manifest: dict[str, Any],
    keep_last: int,
) -> Path:
    import torch

    step = int(trainer_state["global_step"])
    checkpoint_dir = checkpoint_root / f"step_{step:08d}"
    temporary = checkpoint_root / f".step_{step:08d}.tmp"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    torch.save(trainable_state_dict(model), temporary / "trainable_state.pt")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": capture_rng_state(),
            "trainer_state": trainer_state,
        },
        temporary / "trainer_state.pt",
    )
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(checkpoint_dir)
    (checkpoint_root / "latest.json").write_text(
        json.dumps({"checkpoint": checkpoint_dir.name, "global_step": step}, indent=2),
        encoding="utf-8",
    )

    checkpoints = sorted(path for path in checkpoint_root.glob("step_*") if path.is_dir())
    for old in checkpoints[: max(0, len(checkpoints) - int(keep_last))]:
        shutil.rmtree(old)
    return checkpoint_dir


def load_latest_checkpoint(
    checkpoint_root: Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    device: Any,
) -> dict[str, Any] | None:
    import torch

    latest_path = checkpoint_root / "latest.json"
    if not latest_path.exists():
        return None
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    checkpoint_dir = checkpoint_root / latest["checkpoint"]
    trainable = torch.load(checkpoint_dir / "trainable_state.pt", map_location="cpu", weights_only=False)
    payload = torch.load(checkpoint_dir / "trainer_state.pt", map_location="cpu", weights_only=False)
    load_trainable_state_dict(model, trainable)
    optimizer.load_state_dict(payload["optimizer"])
    _optimizer_to_device(optimizer, device)
    scheduler.load_state_dict(payload["scheduler"])
    restore_rng_state(payload["rng"])
    return dict(payload["trainer_state"])
