from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rt_mcmc.experiments.covid_case_study import run_covid_case_study


def _write_rt_csv(result: dict, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "incidence", "rt_mean", "rt_lower", "rt_upper"])
        for date, incidence, mean, lower, upper in zip(
            result["dates"],
            result["incidence"],
            result["rt_mean"],
            result["rt_lower"],
            result["rt_upper"],
        ):
            writer.writerow([date, incidence, mean, lower, upper])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run COVID Rt case study")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "covid_case.yaml",
        help="Path to the YAML config file. Defaults to configs/covid_case.yaml.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path for exporting the full Rt sequence as CSV.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = run_covid_case_study(cfg)

    output_csv = args.output_csv
    if output_csv is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = ROOT / "outputs" / f"covid_rt_{stamp}.csv"
    _write_rt_csv(result, output_csv)

    print("Case study completed")
    print(f"accept_rate_rt: {result['accept_rate_rt']:.3f}")
    if result.get("accept_rate_beta") is not None:
        print(f"accept_rate_beta: {result['accept_rate_beta']:.3f}")
    if result.get("rhat_max") is not None:
        print(f"rhat_max: {result['rhat_max']:.4f}")
    print(f"sigma2_rw_mean: {result['sigma2_rw_mean']:.5f}")
    print("First 5 Rt means:", [round(x, 3) for x in result["rt_mean"][:5]])
    print(f"CSV exported: {output_csv}")


if __name__ == "__main__":
    main()
