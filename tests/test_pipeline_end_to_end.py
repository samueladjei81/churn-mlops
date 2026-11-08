"""Training, scoring and the full monthly simulation, on fake data."""

import json
from pathlib import Path

import pandas as pd

from churn import db
from churn.data import FEATURES, prepare_training_data
from churn.predict import predict_records, score
from churn.registry import ModelRegistry
from churn.simulation import make_batches, run_simulation
from churn.train import fit_model


def test_fit_model_tunes_and_reports(small_config, raw_df):
    X, y = prepare_training_data(raw_df)
    model, meta = fit_model(X, y, small_config)
    assert meta["params"]["model__n_estimators"] in (20, 40)
    assert 0.05 <= meta["threshold"] <= 0.95
    assert meta["test"]["roc_auc"] > 0.7               # fake data has a clear signal
    assert "campaign" in meta["test"]
    assert model.predict_proba(X.head(5)).shape == (5, 2)


def test_fit_model_reuses_given_params(small_config, raw_df):
    X, y = prepare_training_data(raw_df)
    params = {"model__n_estimators": 30, "model__max_depth": 2}
    _, meta = fit_model(X, y, small_config, params=params)
    assert meta["params"] == params and meta["cv_score"] is None


def test_scoring(small_config, raw_df):
    X, y = prepare_training_data(raw_df)
    model, meta = fit_model(X, y, small_config)
    new = raw_df.drop(columns=["Churn"]).head(25).copy()
    new.insert(0, "customer_id", range(1000, 1025))
    scores = score(new, model, meta["threshold"])
    assert len(scores) == 25
    assert scores["churn_probability"].between(0, 1).all()
    assert scores["churn_probability"].is_monotonic_decreasing     # highest risk first

    meta["version"] = 1
    records = X.head(3).assign(customer_id=["a", "b", "c"]).to_dict("records")
    out = predict_records(records, model, meta)
    assert [r["customer_id"] for r in out] == ["a", "b", "c"]
    assert all(isinstance(r["targeted"], bool) for r in out)


def test_batches_split_history_and_months(small_config):
    paths = make_batches(small_config)
    sizes = [len(pd.read_csv(p)) for p in paths]
    assert len(paths) == 4                                           # month 0 + 3 months
    assert sum(sizes) == 900                                         # duplicates removed
    assert sizes[0] == 450
    ids = pd.concat(pd.read_csv(p)["customer_id"] for p in paths)
    assert ids.is_unique


def test_full_simulation(small_config):
    monitoring = run_simulation(small_config)

    # every month was monitored and logged
    assert monitoring["month"].tolist() == [1, 2, 3]
    assert monitoring["pr_auc"].notna().all()

    # drift injected from month 2 is detected and triggers a retrain
    assert monitoring.loc[monitoring["month"] == 1, "data_drift"].item() == 0
    assert monitoring.loc[monitoring["month"] == 2, "data_drift"].item() == 1
    assert monitoring.loc[monitoring["month"] == 2, "action"].item().startswith("Retrained")

    # the registry has more than one version and a champion
    reg = ModelRegistry(small_config["registry"]["dir"])
    assert len(reg.versions()) >= 2
    assert reg.champion_version() is not None

    # every customer ended up in the database with a label, and was scored
    conn = db.connect(small_config["database"]["path"])
    assert len(db.load_customers(conn, labelled_only=True)) == 900
    assert len(db.read_table(conn, "predictions")) >= 450
    events = db.read_table(conn, "model_events")["event"].tolist()
    assert "registered" in events and "promoted" in events

    # reports and target lists were written
    reports = Path(small_config["reports"]["dir"])
    for name in ["summary.html", "monitoring.csv", "month_01.html", "targets_month_03.csv"]:
        assert (reports / name).exists(), name
    targets = pd.read_csv(reports / "targets_month_01.csv")
    assert {"customer_id", "churn_probability", "targeted", "customer_value"} <= set(targets.columns)

    meta = json.loads((Path(small_config["registry"]["dir"]) / "v1" / "metadata.json").read_text())
    assert meta["features"] == FEATURES
