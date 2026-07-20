from __future__ import annotations

from typing import Any

from qwen_single_layer_rl.rewards.math_reward import binary_math_reward


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = binary_math_reward(solution_str, str(ground_truth))
    return {
        "score": result.reward,
        "predicted": result.predicted,
        "target": result.target,
        "verifier": result.verifier,
        "data_source": data_source,
    }
