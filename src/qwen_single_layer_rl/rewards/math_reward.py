from __future__ import annotations

import re
from dataclasses import dataclass


_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class RewardResult:
    reward: float
    predicted: str
    target: str
    verifier: str


def extract_answer(text: str) -> str:
    boxed = extract_last_boxed(text)
    if boxed:
        return boxed.strip()
    numbers = _NUMBER_RE.findall(text)
    return numbers[-1].strip() if numbers else text.strip()


def extract_last_boxed(text: str) -> str | None:
    marker = r"\boxed{"
    starts: list[int] = []
    search_at = 0
    while True:
        idx = text.find(marker, search_at)
        if idx < 0:
            break
        starts.append(idx + len(marker))
        search_at = idx + len(marker)

    last: str | None = None
    for start in starts:
        depth = 1
        pos = start
        while pos < len(text):
            char = text[pos]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    last = text[start:pos]
                    break
            pos += 1
    return last


def binary_math_reward(completion: str, target_answer: str) -> RewardResult:
    predicted = normalize_answer(extract_answer(completion))
    target = normalize_answer(target_answer)
    verified = _production_math_verify(completion, target_answer)
    return RewardResult(
        reward=verified,
        predicted=predicted,
        target=target,
        verifier="verl_math_verify",
    )


def _production_math_verify(completion: str, target_answer: str) -> float:
    try:
        from verl.utils.reward_score.math_verify import compute_score as verify_math
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Production math reward requires veRL and the math-verify package; "
            "string-exact fallback is forbidden."
        ) from exc

    return float(verify_math(completion, target_answer))


def normalize_answer(text: str) -> str:
    return text.strip().replace(",", "").replace(" ", "")
