from __future__ import annotations

import numpy as np

from rt_mcmc.data.serial_interval import discretized_gamma_weights
from rt_mcmc.evaluation.metrics import (
    coverage_probability,
    gelman_rubin_rhat,
    rmse,
    summarize_rt_samples,
)
from rt_mcmc.mcmc.sampler import MetropolisWithinGibbs
from rt_mcmc.models.bayesian_rt import BayesianRtModel
from rt_mcmc.models.renewal import compute_infectiousness


def synthetic_rt_curve(T: int, scenario: str = "step") -> np.ndarray:
    if scenario == "stable":
        return np.ones(T) * 1.2

    if scenario == "exp_decay":
        t = np.arange(T, dtype=float)
        return 0.9 + 0.9 * np.exp(-0.03 * t)

    # default: step
    rt = np.ones(T) * 1.15
    rt[T // 4 : T // 2] = 1.45
    rt[T // 2 : (3 * T) // 4] = 0.95
    rt[(3 * T) // 4 :] = 1.20
    return rt


def generate_incidence(rt_true: np.ndarray, serial_weights: np.ndarray, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    T = len(rt_true)
    y = np.zeros(T, dtype=float)
    y[0] = 20

    for t in range(1, T):
        lam = compute_infectiousness(y[: t + 1], serial_weights)[t]
        mu = max(rt_true[t] * lam, 1e-8)
        y[t] = rng.poisson(mu)

    return y


def _run_one_chain(model: BayesianRtModel, config: dict, seed: int) -> dict:
    sampler = MetropolisWithinGibbs(
        model=model,
        n_iter=int(config.get("n_iter", 3500)),
        burn_in=int(config.get("burn_in", 1200)),
        thin=int(config.get("thin", 2)),
        step_rt=float(config.get("step_rt", 0.07)),
        step_beta=float(config.get("step_beta", 0.04)),
        prior_sigma2_alpha=float(config.get("prior_sigma2_alpha", 2.0)),
        prior_sigma2_beta=float(config.get("prior_sigma2_beta", 0.02)),
        init_epiestim_a0=float(config.get("init_epiestim_a0", 2.0)),
        init_epiestim_b0=float(config.get("init_epiestim_b0", 2.0)),
        init_window=int(config.get("init_window", 7)),
        seed=seed,
    )
    res = sampler.run()
    return {
        "rt_samples": res.rt_samples,
        "sigma2_samples": res.sigma2_rw_samples,
        "accept_rate_rt": res.accept_rate_rt,
    }


def run_simulation(config: dict) -> dict:
    T = int(config.get("T", 120))
    si_mean = float(config.get("si_mean", 3.0))
    si_std = float(config.get("si_std", 1.5))
    si_max_lag = int(config.get("si_max_lag", 20))
    alpha = float(config.get("alpha", 0.1))
    n_chains = int(config.get("n_chains", 4))
    scenarios = config.get("scenarios", ["stable", "step", "exp_decay"])

    serial_weights = discretized_gamma_weights(si_mean, si_std, si_max_lag)

    base_data_seed = int(config.get("data_seed", 123))
    base_mcmc_seed = int(config.get("mcmc_seed", 42))

    scenario_results: dict[str, dict] = {}

    for i, scenario in enumerate(scenarios):
        rt_true = synthetic_rt_curve(T, scenario=str(scenario))
        y = generate_incidence(rt_true, serial_weights, seed=base_data_seed + i * 1000)

        chain_runs: list[dict] = []
        for c in range(n_chains):
            model = BayesianRtModel(
                incidence=y,
                serial_weights=serial_weights,
                covariates=None,
                sigma_rw=float(config.get("sigma_rw", 0.15)),
                sigma_beta=float(config.get("sigma_beta", 1.0)),
            )
            chain_seed = base_mcmc_seed + i * 1000 + c * 97
            chain_runs.append(_run_one_chain(model, config, seed=chain_seed))

        rt_chain_samples = np.stack([r["rt_samples"] for r in chain_runs], axis=0)  # (m, n, T)
        pooled_rt_samples = np.concatenate([r["rt_samples"] for r in chain_runs], axis=0)
        sigma2_all = np.concatenate([r["sigma2_samples"] for r in chain_runs], axis=0)

        summary = summarize_rt_samples(pooled_rt_samples, alpha=alpha)
        rhat_t = gelman_rubin_rhat(rt_chain_samples)

        scenario_results[str(scenario)] = {
            "accept_rate_rt_mean": float(np.mean([r["accept_rate_rt"] for r in chain_runs])),
            "rmse": rmse(summary["mean"], rt_true),
            "coverage_90": coverage_probability(summary["lower"], summary["upper"], rt_true),
            "rhat_max": float(np.nanmax(rhat_t)),
            "rhat_mean": float(np.nanmean(rhat_t)),
            "sigma2_rw_mean": float(np.mean(sigma2_all)),
            "true_rt": rt_true,
            "estimated": summary,
            "incidence": y,
        }

    return {
        "scenarios": scenario_results,
        "config": {
            "T": T,
            "n_chains": n_chains,
            "alpha": alpha,
            "scenarios": scenarios,
        },
    }
