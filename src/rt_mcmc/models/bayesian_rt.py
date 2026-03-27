from __future__ import annotations

import numpy as np

from rt_mcmc.models.renewal import compute_infectiousness, poisson_log_likelihood


class BayesianRtModel:
    """Bayesian renewal model with optional linear NPI effect on log(R_t)."""

    def __init__(
        self,
        incidence: np.ndarray,
        serial_weights: np.ndarray,
        covariates: np.ndarray | None = None,
        sigma_rw: float = 0.15,
        sigma_beta: float = 1.0,
    ) -> None:
        self.y = np.asarray(incidence, dtype=float)
        self.w = np.asarray(serial_weights, dtype=float)
        self.X = None if covariates is None else np.asarray(covariates, dtype=float)
        self.T = len(self.y)
        self.default_sigma_rw = sigma_rw
        self.sigma_beta = sigma_beta
        self.lam = compute_infectiousness(self.y, self.w)

        if self.X is not None and self.X.shape[0] != self.T:
            raise ValueError("covariates must have same number of rows as incidence length")

    @property
    def p(self) -> int:
        return 0 if self.X is None else self.X.shape[1]

    def initial_log_rt_epiestim(self, a0: float = 2.0, b0: float = 2.0, window: int = 7) -> np.ndarray:
        """Use Cori-style Gamma-Poisson posterior mean as an Rt initialization."""
        window = max(1, int(window))
        rt0 = np.ones(self.T, dtype=float)

        for t in range(self.T):
            left = max(0, t - window + 1)
            i_sum = float(np.sum(self.y[left : t + 1]))
            lam_sum = float(np.sum(self.lam[left : t + 1]))
            rt0[t] = (a0 + i_sum) / max(b0 + lam_sum, 1e-8)

        return np.log(np.maximum(rt0, 1e-6))

    def rw_diff_sum_squares(self, log_rt: np.ndarray) -> float:
        if self.T <= 1:
            return 0.0
        d = np.diff(log_rt)
        return float(np.dot(d, d))

    def log_prior(self, log_rt: np.ndarray, beta: np.ndarray | None, sigma2_rw: float) -> float:
        lp = 0.0
        lp += -0.5 * (log_rt[0] / 0.8) ** 2

        if self.T > 1:
            s2 = max(sigma2_rw, 1e-10)
            lp += -0.5 * (self.T - 1) * np.log(s2)
            lp += -0.5 * self.rw_diff_sum_squares(log_rt) / s2

        if self.p > 0 and beta is not None:
            lp += float(np.sum(-0.5 * (beta / self.sigma_beta) ** 2))

        return float(lp)

    def expected_cases(self, log_rt: np.ndarray, beta: np.ndarray | None) -> np.ndarray:
        eta = log_rt.copy()
        if self.p > 0 and beta is not None:
            eta = eta + self.X @ beta
        rt = np.exp(eta)
        return rt * self.lam

    def log_posterior(self, log_rt: np.ndarray, beta: np.ndarray | None, sigma2_rw: float) -> float:
        mu = self.expected_cases(log_rt, beta)
        ll = poisson_log_likelihood(self.y, mu)
        lp = self.log_prior(log_rt, beta, sigma2_rw=sigma2_rw)
        return float(ll + lp)
