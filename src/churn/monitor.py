"""Data drift detection.

Compares each feature's distribution in a new batch against the data the
current model was trained on, using the Population Stability Index (PSI):
    PSI < 0.1   little change
    0.1 - 0.2   moderate change
    > 0.2       significant change (drift)
A Kolmogorov-Smirnov test p-value is included for continuous features.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

EPS = 1e-4


def psi(reference, current, bins: int = 10) -> float:
    ref, cur = np.asarray(reference, float), np.asarray(current, float)
    if len(np.unique(ref)) <= bins:                       # categorical / low-cardinality
        cats = np.union1d(np.unique(ref), np.unique(cur))
        r = np.array([(ref == c).mean() for c in cats])
        c = np.array([(cur == c).mean() for c in cats])
    else:                                                  # continuous: quantile bins from reference
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        r = np.histogram(ref, edges)[0] / len(ref)
        c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def drift_table(reference: pd.DataFrame, current: pd.DataFrame, features: list[str],
                psi_threshold: float = 0.2) -> pd.DataFrame:
    rows = []
    for f in features:
        value = psi(reference[f], current[f])
        continuous = reference[f].nunique() > 10
        rows.append({
            "feature": f,
            "psi": round(value, 4),
            "ks_pvalue": round(float(ks_2samp(reference[f], current[f]).pvalue), 4) if continuous else None,
            "reference_mean": round(float(reference[f].mean()), 3),
            "current_mean": round(float(current[f].mean()), 3),
            "drifted": value > psi_threshold,
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def drift_summary(table: pd.DataFrame, share_threshold: float) -> dict:
    share = float(table["drifted"].mean())
    return {
        "drift_share": round(share, 3),
        "data_drift": share >= share_threshold,
        "drifted_features": table.loc[table["drifted"], "feature"].tolist(),
    }
