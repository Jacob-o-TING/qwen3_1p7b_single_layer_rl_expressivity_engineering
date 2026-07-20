from __future__ import annotations


def layer_contribution(layer_score: float, base_score: float, full_score: float) -> float:
    denom = full_score - base_score
    if denom == 0:
        raise ValueError("full_score and base_score must differ")
    return (layer_score - base_score) / denom


def rank_layers(scores: dict[int, float], base_score: float, full_score: float) -> list[tuple[int, float]]:
    ranked = [
        (layer, layer_contribution(score, base_score=base_score, full_score=full_score))
        for layer, score in scores.items()
    ]
    return sorted(ranked, key=lambda item: item[1], reverse=True)
