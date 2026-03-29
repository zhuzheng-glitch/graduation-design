from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

EPIESTIM_SRC = ROOT.parent / "rt_epiestim_project" / "src"
if str(EPIESTIM_SRC) not in sys.path:
    sys.path.insert(0, str(EPIESTIM_SRC))

from rt_mcmc.data.io import load_case_data
from rt_mcmc.data.serial_interval import discretized_gamma_weights
from rt_mcmc.evaluation.metrics import gelman_rubin_rhat, summarize_rt_samples
from rt_mcmc.mcmc.sampler import MCMCResult
from rt_mcmc.models.bayesian_rt import BayesianRtModel
from rt_epiestim.models.cori_epiestim import estimate_rt_cori


@dataclass
class ConditionedResult:
    chain_result: MCMCResult
    fixed_log_rt: np.ndarray
    epi_mean: np.ndarray
    prefix_days: int
    epi_start_day: int


def _normalize_float(value: float) -> float | None:
    value = float(value)
    return None if math.isnan(value) else value


def _sample_sigma2_rw(
    log_rt: np.ndarray,
    rng: np.random.Generator,
    prior_alpha: float,
    prior_beta: float,
) -> float:
    n = max(len(log_rt) - 1, 0)
    if n == 0:
        return 1.0 / max(rng.gamma(shape=prior_alpha, scale=1.0 / max(prior_beta, 1e-12)), 1e-12)

    diff = np.diff(log_rt)
    ssq = float(np.dot(diff, diff))
    post_alpha = prior_alpha + 0.5 * n
    post_beta = prior_beta + 0.5 * ssq
    gamma_draw = rng.gamma(shape=post_alpha, scale=1.0 / max(post_beta, 1e-12))
    return 1.0 / max(gamma_draw, 1e-12)


def _build_fixed_prefix(
    model: BayesianRtModel,
    config: dict,
    prefix_days: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    init_a0 = float(config.get("init_epiestim_a0", 2.0))
    init_b0 = float(config.get("init_epiestim_b0", 2.0))
    init_window = int(config.get("init_window", 7))

    baseline_log_rt = model.initial_log_rt_epiestim(
        a0=init_a0,
        b0=init_b0,
        window=init_window,
    )

    epi_window = int(config.get("conditioned_epiestim_window", init_window))
    epi_shape = float(config.get("conditioned_epiestim_prior_shape", init_a0))
    epi_scale = float(config.get("conditioned_epiestim_prior_scale", 1.0 / max(init_b0, 1e-12)))
    alpha = float(config.get("alpha", 0.1))
    posterior_draws = int(config.get("conditioned_epiestim_draws", 4000))
    posterior_seed = int(config.get("conditioned_epiestim_seed", config.get("mcmc_seed", 42) + 2000))

    epi_result = estimate_rt_cori(
        incidence=model.y,
        serial_weights=model.w,
        window=epi_window,
        prior_shape=epi_shape,
        prior_scale=epi_scale,
        alpha=alpha,
        posterior_draws=posterior_draws,
        posterior_seed=posterior_seed,
    )

    fixed_log_rt = baseline_log_rt.copy()
    end = min(prefix_days, model.T)
    for idx in range(epi_result.start_index, end):
        if np.isfinite(epi_result.mean[idx]):
            fixed_log_rt[idx] = math.log(max(float(epi_result.mean[idx]), 1e-6))

    return fixed_log_rt, epi_result.mean.copy(), int(epi_result.start_index)


def _run_one_conditioned_chain(model: BayesianRtModel, config: dict, seed: int, prefix_days: int) -> ConditionedResult:
    rng = np.random.default_rng(seed)
    p = model.p
    T = model.T

    fixed_log_rt, epi_mean, epi_start_day = _build_fixed_prefix(model, config, prefix_days)

    log_rt = fixed_log_rt.copy()
    beta = np.zeros(p, dtype=float) if p > 0 else None
    sigma2_rw = float(model.default_sigma_rw**2)

    n_iter = int(config.get("n_iter", 3000))
    burn_in = int(config.get("burn_in", 1000))
    thin = int(config.get("thin", 2))
    step_rt = float(config.get("step_rt", 0.07))
    step_beta = float(config.get("step_beta", 0.04))
    prior_alpha = float(config.get("prior_sigma2_alpha", 2.0))
    prior_beta = float(config.get("prior_sigma2_beta", 0.02))

    keep_log_rt: list[np.ndarray] = []
    keep_beta: list[np.ndarray] = []
    keep_sigma2: list[float] = []

    accepted_rt = 0
    proposed_rt = 0
    accepted_beta = 0
    proposed_beta = 0

    current_lp = model.log_posterior(log_rt, beta, sigma2_rw=sigma2_rw)

    for it in range(n_iter):
        for t in range(prefix_days, T):
            cand = log_rt.copy()
            cand[t] = log_rt[t] + rng.normal(0.0, step_rt)
            cand_lp = model.log_posterior(cand, beta, sigma2_rw=sigma2_rw)
            log_q_reverse_minus_forward = cand[t] - log_rt[t]
            log_alpha = cand_lp - current_lp + log_q_reverse_minus_forward

            proposed_rt += 1
            if np.log(rng.uniform()) < log_alpha:
                log_rt = cand
                current_lp = cand_lp
                accepted_rt += 1

        if p > 0 and beta is not None:
            cand_beta = beta + rng.normal(0.0, step_beta, size=p)
            cand_lp = model.log_posterior(log_rt, cand_beta, sigma2_rw=sigma2_rw)
            log_alpha = cand_lp - current_lp
            proposed_beta += 1
            if np.log(rng.uniform()) < log_alpha:
                beta = cand_beta
                current_lp = cand_lp
                accepted_beta += 1

        sigma2_rw = _sample_sigma2_rw(log_rt, rng, prior_alpha, prior_beta)
        current_lp = model.log_posterior(log_rt, beta, sigma2_rw=sigma2_rw)

        if it >= burn_in and ((it - burn_in) % thin == 0):
            keep_log_rt.append(log_rt.copy())
            keep_sigma2.append(float(sigma2_rw))
            if p > 0 and beta is not None:
                keep_beta.append(beta.copy())

    chain_result = MCMCResult(
        log_rt_samples=np.asarray(keep_log_rt),
        beta_samples=np.asarray(keep_beta) if p > 0 else None,
        sigma2_rw_samples=np.asarray(keep_sigma2),
        accept_rate_rt=(accepted_rt / proposed_rt) if proposed_rt else 0.0,
        accept_rate_beta=(accepted_beta / proposed_beta) if proposed_beta else None,
    )
    return ConditionedResult(
        chain_result=chain_result,
        fixed_log_rt=fixed_log_rt,
        epi_mean=epi_mean,
        prefix_days=prefix_days,
        epi_start_day=epi_start_day,
    )


def run_conditioned_case_study(config: dict, prefix_days: int | None = None) -> dict:
    csv_path = Path(config["data_csv"])
    cov_cols = config.get("covariate_columns")
    n_chains = int(config.get("n_chains", 1))

    y, X, df = load_case_data(csv_path, covariate_columns=cov_cols)
    serial_weights = discretized_gamma_weights(
        mean=float(config.get("si_mean", 3.0)),
        std=float(config.get("si_std", 1.5)),
        max_lag=int(config.get("si_max_lag", 20)),
    )

    if prefix_days is None:
        prefix_days = int(config.get("conditioned_prefix_days", max(int(config.get("si_max_lag", 20)), int(config.get("init_window", 7)))))
    prefix_days = max(1, min(int(prefix_days), len(y)))

    base_seed = int(config.get("mcmc_seed", 42))
    chain_runs: list[ConditionedResult] = []
    for c in range(n_chains):
        model = BayesianRtModel(
            incidence=y,
            serial_weights=serial_weights,
            covariates=X,
            sigma_rw=float(config.get("sigma_rw", 0.15)),
            sigma_beta=float(config.get("sigma_beta", 1.0)),
        )
        chain_runs.append(_run_one_conditioned_chain(model, config, seed=base_seed + c * 97, prefix_days=prefix_days))

    rt_chain_samples = np.stack([res.chain_result.rt_samples for res in chain_runs], axis=0)
    pooled_rt = np.concatenate([res.chain_result.rt_samples for res in chain_runs], axis=0)
    summary = summarize_rt_samples(pooled_rt, alpha=float(config.get("alpha", 0.1)))

    fixed_rt = np.exp(chain_runs[0].fixed_log_rt)
    epi_mean = chain_runs[0].epi_mean
    conditioned_mean = summary["mean"].copy()
    conditioned_lower = summary["lower"].copy()
    conditioned_upper = summary["upper"].copy()

    conditioned_mean[:prefix_days] = fixed_rt[:prefix_days]
    conditioned_lower[:prefix_days] = fixed_rt[:prefix_days]
    conditioned_upper[:prefix_days] = fixed_rt[:prefix_days]

    rows: list[dict] = []
    for idx, date in enumerate(df["date"].dt.strftime("%Y-%m-%d").to_list()):
        if idx < chain_runs[0].epi_start_day:
            source = "warmup_fixed"
        elif idx < prefix_days:
            source = "epiestim_fixed"
        else:
            source = "mcmc_conditioned"
        rows.append(
            {
                "date": date,
                "incidence": float(y[idx]),
                "rt_mean": float(conditioned_mean[idx]),
                "rt_lower": float(conditioned_lower[idx]),
                "rt_upper": float(conditioned_upper[idx]),
                "source": source,
                "fixed_rt": float(fixed_rt[idx]),
                "epiestim_rt_mean": _normalize_float(epi_mean[idx]) if np.isfinite(epi_mean[idx]) else None,
                "mcmc_rt_mean": float(summary["mean"][idx]),
            }
        )

    output = {
        "dates": df["date"].dt.strftime("%Y-%m-%d").to_list(),
        "incidence": y,
        "rt_mean": conditioned_mean,
        "rt_lower": conditioned_lower,
        "rt_upper": conditioned_upper,
        "prefix_days": prefix_days,
        "epiestim_start_day": chain_runs[0].epi_start_day,
        "fixed_rt_prefix": fixed_rt,
        "rows": rows,
        "accept_rate_rt": float(np.mean([res.chain_result.accept_rate_rt for res in chain_runs])),
        "accept_rate_beta": None,
        "sigma2_rw_mean": float(np.mean(np.concatenate([res.chain_result.sigma2_rw_samples for res in chain_runs]))),
    }

    if chain_runs[0].chain_result.accept_rate_beta is not None:
        output["accept_rate_beta"] = float(np.mean([res.chain_result.accept_rate_beta for res in chain_runs]))

    if n_chains >= 2:
        rhat_t = gelman_rubin_rhat(rt_chain_samples)
        output["rhat_mean"] = float(np.nanmean(rhat_t))
        output["rhat_max"] = float(np.nanmax(rhat_t))

    return output


def _write_conditioned_csv(result: dict, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "date",
                "incidence",
                "rt_mean",
                "rt_lower",
                "rt_upper",
                "source",
                "fixed_rt",
                "epiestim_rt_mean",
                "mcmc_rt_mean",
            ]
        )
        for row in result["rows"]:
            writer.writerow(
                [
                    row["date"],
                    row["incidence"],
                    row["rt_mean"],
                    row["rt_lower"],
                    row["rt_upper"],
                    row["source"],
                    row["fixed_rt"],
                    row["epiestim_rt_mean"],
                    row["mcmc_rt_mean"],
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a conditioned COVID Rt case study: EpiEstim fixed prefix + MCMC suffix updates.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "covid_case.yaml",
        help="Path to the YAML config file. Defaults to configs/covid_case.yaml.",
    )
    parser.add_argument(
        "--prefix-days",
        type=int,
        default=None,
        help="Number of leading days fixed before conditioned MCMC updates begin. Defaults to max(si_max_lag, init_window).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path for exporting the conditioned Rt sequence as CSV.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = run_conditioned_case_study(cfg, prefix_days=args.prefix_days)

    output_csv = args.output_csv
    if output_csv is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = ROOT / "outputs" / f"covid_rt_conditioned_{stamp}.csv"
    _write_conditioned_csv(result, output_csv)

    print("Conditioned case study completed")
    print(f"epiestim_start_day: {result['epiestim_start_day']}")
    print(f"conditioned_mcmc_start_day: {result['prefix_days']}")
    print(f"accept_rate_rt: {result['accept_rate_rt']:.3f}")
    if result.get("rhat_max") is not None:
        print(f"rhat_max: {result['rhat_max']:.4f}")
    print(f"sigma2_rw_mean: {result['sigma2_rw_mean']:.5f}")
    print("First 5 Rt means:", [round(x, 3) for x in result["rt_mean"][:5]])
    print(f"CSV exported: {output_csv}")


if __name__ == "__main__":
    main()
