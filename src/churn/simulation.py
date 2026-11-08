"""Replay the dataset as if customers arrived month by month.

The dataset is a single snapshot, so we:
  * give every customer an ID and shuffle them
  * use the first half as history (month 0) for the first model
  * split the rest into monthly batches
  * from `drift_start_month`, change customer behaviour on purpose (more call
    failures, less usage, more complaints that don't lead to churn) so the
    monitoring has something real to catch
"""

import shutil

import numpy as np
import pandas as pd

from churn import db
from churn.config import resolve
from churn.cycle import run_month, train_and_register
from churn.data import load_raw, standardise_columns, validate
from churn.registry import ModelRegistry
from churn.report import summary_report


def inject_drift(batch: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    batch = batch.copy()
    batch["call_failure"] = (batch["call_failure"] * 1.8 + rng.integers(0, 4, len(batch))).round().astype(int)
    batch["seconds_of_use"] = (batch["seconds_of_use"] * 0.7).round().astype(int)
    extra = rng.random(len(batch)) < 0.15                  # new complaints, but these customers don't leave
    batch.loc[extra, "complains"] = 1
    return batch


def make_batches(config: dict) -> list:
    sim = config["simulation"]
    rng = np.random.default_rng(sim["random_state"])
    df = standardise_columns(load_raw(resolve(config["data"]["raw_path"])))
    df = df.drop_duplicates().sample(frac=1, random_state=sim["random_state"]).reset_index(drop=True)
    validate(df, require_target=True)
    df.insert(0, "customer_id", [f"C{i:05d}" for i in range(1, len(df) + 1)])

    n_hist = int(len(df) * sim["history_fraction"])
    edges = np.linspace(n_hist, len(df), sim["n_months"] + 1).round().astype(int)
    chunks = [df.iloc[:n_hist]] + [df.iloc[a:b] for a, b in zip(edges[:-1], edges[1:])]
    out_dir = resolve(sim["batches_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for month, chunk in enumerate(chunks):
        if month >= sim["drift_start_month"]:
            chunk = inject_drift(chunk, rng)
        path = out_dir / f"month_{month:02d}.csv"
        chunk.to_csv(path, index=False)
        paths.append(path)
    return paths


def reset(config: dict) -> None:
    """Start from a clean slate: database, registry, batches and reports."""
    db_path = resolve(config["database"]["path"])
    db_path.unlink(missing_ok=True)
    for key in [("registry", "dir"), ("simulation", "batches_dir"), ("reports", "dir")]:
        shutil.rmtree(resolve(config[key[0]][key[1]]), ignore_errors=True)


def setup(config: dict):
    """Fresh database with month 0 (history, labelled) loaded."""
    reset(config)
    make_batches(config)
    conn = db.connect(resolve(config["database"]["path"]))
    history = pd.read_csv(resolve(config["simulation"]["batches_dir"]) / "month_00.csv")
    db.insert_customers(conn, history, month=0, include_labels=True)
    print(f"Database ready: {len(history)} historical customers loaded (month 0)")
    return conn


def run_simulation(config: dict) -> pd.DataFrame:
    conn = setup(config)
    registry = ModelRegistry(resolve(config["registry"]["dir"]))

    print("Training the first model on historical data (with hyperparameter tuning)...")
    history = db.load_customers(conn, labelled_only=True)
    _, meta = train_and_register(conn, registry, config, history, reason="initial training")
    registry.promote(meta["version"])
    db.log_event(conn, "promoted", meta["version"], {"reason": "first model"})
    print(f"  v{meta['version']} is champion: test PR-AUC {meta['test']['pr_auc']}, "
          f"threshold {meta['threshold']}")

    for month in range(1, config["simulation"]["n_months"] + 1):
        row = run_month(month, conn, registry, config)
        print(f"Month {month}: v{row['model_version']} PR-AUC {row['pr_auc']} | "
              f"drift {row['drift_share']:.0%} | profit {row['campaign_profit']:,.0f} "
              f"(random {row['random_profit']:,.0f}) | {row['action']}")

    monitoring = db.read_table(conn, "monitoring")
    reports = resolve(config["reports"]["dir"])
    monitoring.to_csv(reports / "monitoring.csv", index=False)
    path = summary_report(reports / "summary.html", monitoring, registry.versions(),
                          registry.champion_version(), config["monitoring"]["drift_share_threshold"])
    conn.close()
    print(f"\nSummary report: {path}")
    return monitoring
