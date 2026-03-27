from __future__ import annotations

import numpy as np


def compute_infectiousness(incidence: np.ndarray, serial_weights: np.ndarray) -> np.ndarray:
    """Compute Lambda_t = sum_s I_{t-s} w_s for renewal models."""
    T = len(incidence)
    L = len(serial_weights)
    lam = np.zeros(T, dtype=float)

    for t in range(T):
        max_s = min(t, L)
        if max_s == 0:
            lam[t] = 1e-8
            continue
        history = incidence[t - max_s : t][::-1]
        w = serial_weights[:max_s]
        lam[t] = float(np.dot(history, w))

    return np.maximum(lam, 1e-8)


def poisson_log_likelihood(y: np.ndarray, mu: np.ndarray) -> float:
    """Poisson log-likelihood up to additive constant (ignoring log(y!))."""
    mu = np.maximum(mu, 1e-12)
    return float(np.sum(y * np.log(mu) - mu))
