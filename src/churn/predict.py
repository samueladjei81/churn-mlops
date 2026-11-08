"""Scoring customers with a trained model."""

import pandas as pd

from churn.data import prepare_features


def score(df: pd.DataFrame, model, threshold: float) -> pd.DataFrame:
    """Churn probability and targeting decision per customer, highest risk first."""
    X = prepare_features(df)
    proba = model.predict_proba(X)[:, 1]
    out = pd.DataFrame(index=df.index)
    if "customer_id" in df.columns:
        out["customer_id"] = df["customer_id"]
    out["churn_probability"] = proba.round(4)
    out["targeted"] = (proba >= threshold).astype(int)
    out["customer_value"] = X["customer_value"].values
    return out.sort_values("churn_probability", ascending=False)


def predict_records(records: list[dict], model, metadata: dict) -> list[dict]:
    """Score a list of customer dicts (used by the API)."""
    df = pd.DataFrame(records)
    scored = score(df, model, metadata["threshold"]).sort_index()
    return [
        {
            "customer_id": records[i].get("customer_id"),
            "churn_probability": float(row.churn_probability),
            "targeted": bool(row.targeted),
            "model_version": metadata["version"],
        }
        for i, row in zip(scored.index, scored.itertuples())
    ]
