from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"date", "cases"}


def load_case_data(csv_path: str | Path, covariate_columns: Iterable[str] | None = None) -> tuple[np.ndarray, np.ndarray | None, pd.DataFrame]:
    """Load COVID incidence data and optional NPI covariates from CSV."""
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if (df["cases"] < 0).any():
        raise ValueError("cases must be non-negative")

    incidence = df["cases"].to_numpy(dtype=float)

    X = None
    if covariate_columns:
        bad = [c for c in covariate_columns if c not in df.columns]
        if bad:
            raise ValueError(f"Covariate columns not found: {bad}")
        X = df[list(covariate_columns)].to_numpy(dtype=float)

    return incidence, X, df
