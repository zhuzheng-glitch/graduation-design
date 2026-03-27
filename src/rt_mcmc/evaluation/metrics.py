from __future__ import annotations

import numpy as np


def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    pred = np.asarray(pred)
    truth = np.asarray(truth)
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def coverage_probability(lower: np.ndarray, upper: np.ndarray, truth: np.ndarray) -> float:
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    truth = np.asarray(truth)
    inside = (truth >= lower) & (truth <= upper)
    return float(np.mean(inside))


def summarize_rt_samples(rt_samples: np.ndarray, alpha: float = 0.1) -> dict[str, np.ndarray]:
    q_low = alpha / 2
    q_high = 1 - alpha / 2
    return {
        "mean": np.mean(rt_samples, axis=0),
        "median": np.median(rt_samples, axis=0),
        "lower": np.quantile(rt_samples, q_low, axis=0),
        "upper": np.quantile(rt_samples, q_high, axis=0),
    }


def gelman_rubin_rhat(chains: np.ndarray) -> np.ndarray:
    """Compute Gelman-Rubin R-hat for chains with shape (m, n, d)."""
    arr = np.asarray(chains, dtype=float)
    if arr.ndim == 2:
        arr = arr[:, :, None]

    m, n, d = arr.shape
    if m < 2 or n < 2:
        return np.full(d, np.nan)

    chain_means = np.mean(arr, axis=1)
    grand_mean = np.mean(chain_means, axis=0)

    B = n * np.sum((chain_means - grand_mean) ** 2, axis=0) / (m - 1)
    W = np.mean(np.var(arr, axis=1, ddof=1), axis=0)

    var_hat = ((n - 1) / n) * W + (1 / n) * B
    with np.errstate(divide="ignore", invalid="ignore"):
        rhat = np.sqrt(var_hat / W)

    rhat = np.where(np.isfinite(rhat), rhat, np.nan)
    return rhat
