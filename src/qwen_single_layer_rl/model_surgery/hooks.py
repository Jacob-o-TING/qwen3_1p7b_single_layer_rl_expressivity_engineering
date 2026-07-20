from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayerSet:
    name: str
    layers: tuple[int, ...]
    source: str


QWEN3_1P7B_LAYER_SETS: dict[str, LayerSet] = {
    "layer10": LayerSet("layer10", (10,), "paper high-contribution single layer"),
    "middle_11_15": LayerSet("middle_11_15", (11, 12, 13, 14, 15), "paper heuristic middle-5"),
    "reported_top5": LayerSet("reported_top5", (10, 12, 9, 2, 13), "paper top-5 LR ablation table"),
}


def get_layer_set(name: str) -> LayerSet:
    try:
        return QWEN3_1P7B_LAYER_SETS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown layer set: {name}") from exc
