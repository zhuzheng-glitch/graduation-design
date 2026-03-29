from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import math
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
from rt_mcmc.experiments.covid_case_study import run_covid_case_study
from rt_epiestim.models.cori_epiestim import estimate_rt_cori


def _normalize(value: float) -> float | None:
    value = float(value)
    return None if math.isnan(value) else value


def _write_hybrid_csv(result: dict, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "date",
                "incidence",
                "hybrid_rt_mean",
                "hybrid_rt_lower",
                "hybrid_rt_upper",
                "source",
                "mcmc_rt_mean",
                "epiestim_rt_mean",
            ]
        )
        for row in result["rows"]:
            writer.writerow(
                [
                    row["date"],
                    row["incidence"],
                    row["hybrid_rt_mean"],
                    row["hybrid_rt_lower"],
                    row["hybrid_rt_upper"],
                    row["source"],
                    row["mcmc_rt_mean"],
                    row["epiestim_rt_mean"],
                ]
            )


def run_hybrid_case_study(config: dict, bridge_days: int | None = None) -> dict:
    mcmc_result = run_covid_case_study(config)

    y, _, df = load_case_data(Path(config["data_csv"]), covariate_columns=config.get("covariate_columns"))
    si_mean = float(config.get("si_mean", 3.0))
    si_std = float(config.get("si_std", 1.5))
    si_max_lag = int(config.get("si_max_lag", 20))
    serial_weights = discretized_gamma_weights(si_mean, si_std, si_max_lag)

    init_window = int(config.get("init_window", 7))
    epi_window = int(config.get("hybrid_epiestim_window", init_window))
    epi_shape = float(config.get("hybrid_epiestim_prior_shape", config.get("init_epiestim_a0", 2.0)))
    init_b0 = float(config.get("init_epiestim_b0", 2.0))
    epi_scale = float(config.get("hybrid_epiestim_prior_scale", 1.0 / max(init_b0, 1e-12)))
    alpha = float(config.get("alpha", 0.1))
    posterior_draws = int(config.get("hybrid_epiestim_draws", 4000))
    posterior_seed = int(config.get("hybrid_epiestim_seed", config.get("mcmc_seed", 42) + 1000))

    epi_result = estimate_rt_cori(
        incidence=y,
        serial_weights=serial_weights,
        window=epi_window,
        prior_shape=epi_shape,
        prior_scale=epi_scale,
        alpha=alpha,
        posterior_draws=posterior_draws,
        posterior_seed=posterior_seed,
    )

    if bridge_days is None:
        bridge_days = int(config.get("hybrid_bridge_days", max(si_max_lag, init_window)))
    bridge_days = max(0, min(int(bridge_days), len(y)))

    rows: list[dict] = []
    hybrid_mean: list[float | None] = []
    hybrid_lower: list[float | None] = []
    hybrid_upper: list[float | None] = []

    for idx, (date, incidence, mcmc_mean, mcmc_lower, mcmc_upper) in enumerate(
        zip(
            mcmc_result["dates"],
            mcmc_result["incidence"],
            mcmc_result["rt_mean"],
            mcmc_result["rt_lower"],
            mcmc_result["rt_upper"],
        )
    ):
        epi_mean = _normalize(epi_result.mean[idx])
        epi_lower = _normalize(epi_result.lower[idx])
        epi_upper = _normalize(epi_result.upper[idx])

        if idx < epi_result.start_index:
            source = "warmup"
            chosen_mean = None
            chosen_lower = None
            chosen_upper = None
        elif idx < bridge_days:
            source = "epiestim"
            chosen_mean = epi_mean
            chosen_lower = epi_lower
            chosen_upper = epi_upper
        else:
            source = "mcmc"
            chosen_mean = float(mcmc_mean)
            chosen_lower = float(mcmc_lower)
            chosen_upper = float(mcmc_upper)

        hybrid_mean.append(chosen_mean)
        hybrid_lower.append(chosen_lower)
        hybrid_upper.append(chosen_upper)
        rows.append(
            {
                "date": date,
                "incidence": float(incidence),
                "hybrid_rt_mean": chosen_mean,
                "hybrid_rt_lower": chosen_lower,
                "hybrid_rt_upper": chosen_upper,
                "source": source,
                "mcmc_rt_mean": float(mcmc_mean),
                "epiestim_rt_mean": epi_mean,
            }
        )

    return {
        "dates": df["date"].dt.strftime("%Y-%m-%d").to_list(),
        "incidence": mcmc_result["incidence"],
        "hybrid_rt_mean": hybrid_mean,
        "hybrid_rt_lower": hybrid_lower,
        "hybrid_rt_upper": hybrid_upper,
        "rows": rows,
        "bridge_days": bridge_days,
        "epiestim_start_day": int(epi_result.start_index),
        "mcmc_main_start_day": bridge_days,
        "mcmc_result": mcmc_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a hybrid COVID Rt case study: EpiEstim early bridge + MCMC main phase.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "covid_case.yaml",
        help="Path to the YAML config file. Defaults to configs/covid_case.yaml.",
    )
    parser.add_argument(
        "--bridge-days",
        type=int,
        default=None,
        help="Number of early days to bridge with EpiEstim before switching to MCMC. Defaults to max(si_max_lag, init_window).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path for exporting the hybrid Rt sequence as CSV.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = run_hybrid_case_study(cfg, bridge_days=args.bridge_days)

    output_csv = args.output_csv
    if output_csv is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = ROOT / "outputs" / f"covid_rt_hybrid_{stamp}.csv"
    _write_hybrid_csv(result, output_csv)

    preview = [
        None if value is None else round(value, 3)
        for value in result["hybrid_rt_mean"][:5]
    ]
    print("Hybrid case study completed")
    print(f"epiestim_start_day: {result['epiestim_start_day']}")
    print(f"mcmc_main_start_day: {result['mcmc_main_start_day']}")
    print("First 5 hybrid Rt means:", preview)
    print(f"CSV exported: {output_csv}")


if __name__ == "__main__":
    main()
