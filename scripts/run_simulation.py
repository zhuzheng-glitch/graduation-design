from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rt_mcmc.experiments.simulation import run_simulation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Rt simulation study")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = run_simulation(cfg)

    print("Simulation completed")
    for scenario, metrics in result["scenarios"].items():
        print(f"[{scenario}]")
        print(f"  accept_rate_rt_mean: {metrics['accept_rate_rt_mean']:.3f}")
        print(f"  rmse: {metrics['rmse']:.4f}")
        print(f"  coverage_90: {metrics['coverage_90']:.4f}")
        print(f"  rhat_max: {metrics['rhat_max']:.4f}")
        print(f"  sigma2_rw_mean: {metrics['sigma2_rw_mean']:.5f}")


if __name__ == "__main__":
    main()
