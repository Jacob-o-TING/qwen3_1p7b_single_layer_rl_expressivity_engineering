from __future__ import annotations

import os
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SeedReport:
    seed: int
    python_hash_seed: str
    torch_seeded: bool
    numpy_seeded: bool


def seed_everything(seed: int) -> SeedReport:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)

    numpy_seeded = False
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
        numpy_seeded = True
    except ModuleNotFoundError:
        pass

    torch_seeded = False
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch_seeded = True
    except ModuleNotFoundError:
        pass

    return SeedReport(
        seed=seed,
        python_hash_seed=os.environ.get("PYTHONHASHSEED", ""),
        torch_seeded=torch_seeded,
        numpy_seeded=numpy_seeded,
    )
