from __future__ import annotations

from pathlib import Path

import numpy as np

from rt_mcmc.data.io import load_case_data
from rt_mcmc.data.serial_interval import discretized_gamma_weights
from rt_mcmc.evaluation.metrics import gelman_rubin_rhat, summarize_rt_samples
from rt_mcmc.mcmc.sampler import MetropolisWithinGibbs
from rt_mcmc.models.bayesian_rt import BayesianRtModel


def _run_one_chain(model: BayesianRtModel, config: dict, seed: int):
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
    return sampler.run()


def run_covid_case_study(config: dict) -> dict:
    csv_path = Path(config["data_csv"])
    cov_cols = config.get("covariate_columns")
    n_chains = int(config.get("n_chains", 1))

    y, X, df = load_case_data(csv_path, covariate_columns=cov_cols)

    serial_weights = discretized_gamma_weights(
        mean=float(config.get("si_mean", 3.0)),
        std=float(config.get("si_std", 1.5)),
        max_lag=int(config.get("si_max_lag", 20)),
    )

    base_seed = int(config.get("mcmc_seed", 42))
    chain_results = []
    for c in range(n_chains):
        model = BayesianRtModel(
            incidence=y,
            serial_weights=serial_weights,
            covariates=X,
            sigma_rw=float(config.get("sigma_rw", 0.15)),
            sigma_beta=float(config.get("sigma_beta", 1.0)),
        )
        chain_results.append(_run_one_chain(model, config, seed=base_seed + c * 97))

    pooled_rt = np.concatenate([res.rt_samples for res in chain_results], axis=0)
    summary = summarize_rt_samples(pooled_rt, alpha=float(config.get("alpha", 0.1)))

    output = {
        "dates": df["date"].dt.strftime("%Y-%m-%d").to_list(),
        "incidence": y,
        "rt_mean": summary["mean"],
        "rt_lower": summary["lower"],
        "rt_upper": summary["upper"],
        "accept_rate_rt": float(np.mean([res.accept_rate_rt for res in chain_results])),
        "accept_rate_beta": None,
        "sigma2_rw_mean": float(np.mean(np.concatenate([res.sigma2_rw_samples for res in chain_results]))),
    }

    if chain_results[0].accept_rate_beta is not None:
        output["accept_rate_beta"] = float(np.mean([res.accept_rate_beta for res in chain_results]))

    if n_chains >= 2:
        rt_chain_samples = np.stack([res.rt_samples for res in chain_results], axis=0)
        rhat_t = gelman_rubin_rhat(rt_chain_samples)
        output["rhat_mean"] = float(np.nanmean(rhat_t))
        output["rhat_max"] = float(np.nanmax(rhat_t))

    if chain_results[0].beta_samples is not None:
        pooled_beta = np.concatenate([res.beta_samples for res in chain_results], axis=0)
        output["beta_mean"] = np.mean(pooled_beta, axis=0)
        output["beta_ci_lower"] = np.quantile(pooled_beta, 0.05, axis=0)
        output["beta_ci_upper"] = np.quantile(pooled_beta, 0.95, axis=0)

    return output
