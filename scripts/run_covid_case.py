from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rt_mcmc.experiments.covid_case_study import run_covid_case_study


def main() -> None:
    parser = argparse.ArgumentParser(description="Run COVID Rt case study")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "covid_case.yaml",
        help="Path to the YAML config file. Defaults to configs/covid_case.yaml.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = run_covid_case_study(cfg)

    print("Case study completed")
    print(f"accept_rate_rt: {result['accept_rate_rt']:.3f}")
    if result.get("accept_rate_beta") is not None:
        print(f"accept_rate_beta: {result['accept_rate_beta']:.3f}")
    if result.get("rhat_max") is not None:
        print(f"rhat_max: {result['rhat_max']:.4f}")
    print(f"sigma2_rw_mean: {result['sigma2_rw_mean']:.5f}")
    print("First 5 Rt means:", [round(x, 3) for x in result["rt_mean"][:5]])


if __name__ == "__main__":
    main()

