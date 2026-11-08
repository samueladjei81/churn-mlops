"""Shared test fixtures: fake data with the real CSV's column names, and a fast config."""

import numpy as np
import pandas as pd
import pytest

from churn.data import COLUMN_MAP


def make_raw(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Call  Failure": rng.integers(0, 37, n),
        "Complains": rng.integers(0, 2, n),
        "Subscription  Length": rng.integers(3, 48, n),
        "Charge  Amount": rng.integers(0, 11, n),
        "Seconds of Use": rng.integers(0, 17000, n),
        "Frequency of use": rng.integers(0, 255, n),
        "Frequency of SMS": rng.integers(0, 500, n),
        "Distinct Called Numbers": rng.integers(0, 97, n),
        "Age Group": rng.integers(1, 6, n),
        "Tariff Plan": rng.integers(1, 3, n),
        "Status": rng.integers(1, 3, n),
        "Age": rng.choice([15, 25, 30, 45, 55], n),
        "Customer Value": rng.random(n) * 2000,
    })
    # churn loosely driven by complaints and inactive status, so there is signal to learn
    signal = df["Complains"] + (df["Status"] == 2) + rng.random(n)
    df["Churn"] = (signal > 1.7).astype(int)
    assert list(df.columns) == list(COLUMN_MAP)
    return df


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return make_raw()


@pytest.fixture
def small_config(tmp_path) -> dict:
    """Config pointing everything at a temp folder, with a tiny grid so tests run fast."""
    csv = tmp_path / "churn.csv"
    raw = make_raw(900)
    pd.concat([raw, raw.head(20)]).to_csv(csv, index=False)  # include 20 duplicates
    return {
        "data": {"raw_path": str(csv), "target": "churn"},
        "database": {"path": str(tmp_path / "churn.db")},
        "simulation": {"batches_dir": str(tmp_path / "batches"), "history_fraction": 0.5,
                       "n_months": 3, "drift_start_month": 2, "random_state": 42},
        "split": {"test_size": 0.3, "random_state": 42},
        "training": {"cv_folds": 3, "scoring": "average_precision",
                     "param_grid": {"model__n_estimators": [20, 40], "model__max_depth": [2]}},
        "business": {"offer_cost": 25, "save_rate": 0.3},
        "monitoring": {"psi_threshold": 0.2, "drift_share_threshold": 0.2,
                       "max_pr_auc_drop": 0.10, "min_improvement": 0.0},
        "registry": {"dir": str(tmp_path / "models")},
        "tracking": {"enabled": False},
        "reports": {"dir": str(tmp_path / "reports")},
    }
