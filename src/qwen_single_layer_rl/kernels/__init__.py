"""Inference kernels for architecture variants."""

from .shs_modulated_projection import (
    SHSProjectionResult,
    shs_multiplicative_delta_triton,
    shs_grouped_multiplicative_delta_triton,
    shs_modulated_projection,
    shs_modulated_projection_reference,
)

__all__ = [
    "SHSProjectionResult",
    "shs_multiplicative_delta_triton",
    "shs_grouped_multiplicative_delta_triton",
    "shs_modulated_projection",
    "shs_modulated_projection_reference",
]
