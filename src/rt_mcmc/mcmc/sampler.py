from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rt_mcmc.models.bayesian_rt import BayesianRtModel


@dataclass
class MCMCResult:
    log_rt_samples: np.ndarray
    beta_samples: np.ndarray | None
    sigma2_rw_samples: np.ndarray
    accept_rate_rt: float
    accept_rate_beta: float | None

    @property
    def rt_samples(self) -> np.ndarray:
        return np.exp(self.log_rt_samples)


class MetropolisWithinGibbs:
    """Metropolis-within-Gibbs with conjugate update for sigma_rw^2."""

    def __init__(
        self,
        model: BayesianRtModel,
        n_iter: int = 3000,
        burn_in: int = 1000,
        thin: int = 2,
        step_rt: float = 0.08,
        step_beta: float = 0.05,
        prior_sigma2_alpha: float = 2.0,
        prior_sigma2_beta: float = 0.02,
        init_epiestim_a0: float = 2.0,
        init_epiestim_b0: float = 2.0,
        init_window: int = 7,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.thin = thin
        self.step_rt = step_rt
        self.step_beta = step_beta
        self.prior_sigma2_alpha = prior_sigma2_alpha
        self.prior_sigma2_beta = prior_sigma2_beta
        self.init_epiestim_a0 = init_epiestim_a0
        self.init_epiestim_b0 = init_epiestim_b0
        self.init_window = init_window
        self.rng = np.random.default_rng(seed)

    def _sample_sigma2_rw(self, log_rt: np.ndarray) -> float:
        n = max(self.model.T - 1, 0)
        ssq = self.model.rw_diff_sum_squares(log_rt)

        post_alpha = self.prior_sigma2_alpha + 0.5 * n
        post_beta = self.prior_sigma2_beta + 0.5 * ssq

        # If X ~ InvGamma(a, b), then 1/X ~ Gamma(a, rate=b).
        gamma_draw = self.rng.gamma(shape=post_alpha, scale=1.0 / max(post_beta, 1e-12))
        return 1.0 / max(gamma_draw, 1e-12)

    def run(self) -> MCMCResult:
        T = self.model.T
        p = self.model.p

        # Slide 4 flow: initialize Rt from EpiEstim-style estimator.
        log_rt = self.model.initial_log_rt_epiestim(
            a0=self.init_epiestim_a0,
            b0=self.init_epiestim_b0,
            window=self.init_window,
        )
        beta = np.zeros(p, dtype=float) if p > 0 else None
        sigma2_rw = float(self.model.default_sigma_rw**2)

        accepted_rt = 0
        proposed_rt = 0
        accepted_beta = 0
        proposed_beta = 0

        keep_log_rt: list[np.ndarray] = []
        keep_beta: list[np.ndarray] = []
        keep_sigma2: list[float] = []

        current_lp = self.model.log_posterior(log_rt, beta, sigma2_rw=sigma2_rw)

        for it in range(self.n_iter):
            # 1) Gibbs scan over Rt path with MH steps.
            for t in range(T):
                cand = log_rt.copy()
                eps = self.rng.normal(0.0, self.step_rt)
                cand[t] = log_rt[t] + eps

                cand_lp = self.model.log_posterior(cand, beta, sigma2_rw=sigma2_rw)

                # Asymmetric proposal correction in Rt-space.
                log_q_reverse_minus_forward = cand[t] - log_rt[t]
                log_alpha = cand_lp - current_lp + log_q_reverse_minus_forward

                proposed_rt += 1
                if np.log(self.rng.uniform()) < log_alpha:
                    log_rt = cand
                    current_lp = cand_lp
                    accepted_rt += 1

            # 2) Gibbs update for beta (if covariates exist), still via MH.
            if p > 0 and beta is not None:
                cand_beta = beta + self.rng.normal(0.0, self.step_beta, size=p)
                cand_lp = self.model.log_posterior(log_rt, cand_beta, sigma2_rw=sigma2_rw)
                log_alpha = cand_lp - current_lp
                proposed_beta += 1
                if np.log(self.rng.uniform()) < log_alpha:
                    beta = cand_beta
                    current_lp = cand_lp
                    accepted_beta += 1

            # 3) Conjugate Gibbs update for sigma_rw^2.
            sigma2_rw = self._sample_sigma2_rw(log_rt)
            current_lp = self.model.log_posterior(log_rt, beta, sigma2_rw=sigma2_rw)

            if it >= self.burn_in and ((it - self.burn_in) % self.thin == 0):
                keep_log_rt.append(log_rt.copy())
                keep_sigma2.append(float(sigma2_rw))
                if p > 0 and beta is not None:
                    keep_beta.append(beta.copy())

        log_rt_samples = np.asarray(keep_log_rt)
        beta_samples = np.asarray(keep_beta) if p > 0 else None
        sigma2_samples = np.asarray(keep_sigma2)

        return MCMCResult(
            log_rt_samples=log_rt_samples,
            beta_samples=beta_samples,
            sigma2_rw_samples=sigma2_samples,
            accept_rate_rt=(accepted_rt / proposed_rt) if proposed_rt else 0.0,
            accept_rate_beta=(accepted_beta / proposed_beta) if proposed_beta else None,
        )
