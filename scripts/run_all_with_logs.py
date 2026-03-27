from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_and_log(command: list[str], log_path: Path, title: str) -> int:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    lines = [
        f"{title}",
        f"started_at: {timestamp}",
        f"command: {' '.join(str(part) for part in command)}",
        f"return_code: {completed.returncode}",
        "",
        "[stdout]",
        completed.stdout.rstrip(),
        "",
        "[stderr]",
        completed.stderr.rstrip(),
        "",
    ]
    log_path.write_text("\n".join(lines), encoding="utf-8")
    return completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simulation and case-study scripts, then export logs.")
    parser.add_argument(
        "--simulation-config",
        type=Path,
        default=ROOT / "configs" / "simulation.yaml",
        help="Path to the simulation config file.",
    )
    parser.add_argument(
        "--covid-config",
        type=Path,
        default=ROOT / "configs" / "covid_case.yaml",
        help="Path to the COVID case-study config file.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=ROOT / "logs",
        help="Directory where the log files will be written.",
    )
    args = parser.parse_args()

    python_exe = Path(sys.executable)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    simulation_log = args.log_dir / f"simulation_{run_stamp}.log"
    covid_log = args.log_dir / f"covid_case_{run_stamp}.log"

    sim_code = _run_and_log(
        [
            str(python_exe),
            str(ROOT / "scripts" / "run_simulation.py"),
            "--config",
            str(args.simulation_config),
        ],
        simulation_log,
        "Simulation Run",
    )
    covid_code = _run_and_log(
        [
            str(python_exe),
            str(ROOT / "scripts" / "run_covid_case.py"),
            "--config",
            str(args.covid_config),
        ],
        covid_log,
        "COVID Case Study Run",
    )

    print("Logs written:")
    print(f"  {simulation_log}")
    print(f"  {covid_log}")

    if sim_code != 0 or covid_code != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

