from __future__ import annotations

from math import exp, gamma

import numpy as np


def discretized_gamma_weights(mean: float, std: float, max_lag: int) -> np.ndarray:
    """Create normalized serial interval weights w_1..w_max_lag from a Gamma distribution."""
    if mean <= 0 or std <= 0:
        raise ValueError("mean and std must be positive")
    if max_lag < 1:
        raise ValueError("max_lag must be >= 1")

    shape = (mean / std) ** 2
    scale = (std**2) / mean

    x = np.arange(1, max_lag + 1, dtype=float)
    coeff = 1.0 / (gamma(shape) * (scale**shape))
    pdf = coeff * (x ** (shape - 1.0)) * np.exp(-x / scale)

    s = pdf.sum()
    if s == 0:
        raise ValueError("serial interval weights collapsed to zero")
    return pdf / s
