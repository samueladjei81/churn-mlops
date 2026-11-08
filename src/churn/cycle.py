"""The monthly MLOps cycle.

For each month:
  1. ingest   new customers arrive (no labels yet) and are validated and stored
  2. score    the champion model scores them; the retention team gets a target list
  3. monitor  their data is compared with the champion's training data (drift)
  4. labels   at month end we learn who actually churned
  5. evaluate how well the predictions and the campaign did
  6. decide   if there is drift or a performance drop, train a challenger on the
              latest data and promote it only if it beats the champion
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from churn import db
from churn.config import resolve
from churn.data import FEATURES, prepare_features
from churn.evaluate import evaluate
from churn.monitor import drift_summary, drift_table
from churn.predict import score
from churn.registry import ModelRegistry
from churn.report import month_report
from churn.tracking import log_run
from churn.train import fit_model


def train_and_register(conn, registry: ModelRegistry, config: dict, data: pd.DataFrame,
                       params: dict | None = None, reason: str = "initial training"):
    model, meta = fit_model(data[FEATURES], data["churn"], config, params=params, reason=reason)
    meta["data_months"] = sorted(int(m) for m in data["month"].unique())
    # Drift reference: the newest month of data the model learned from
    reference = data.loc[data["month"] == data["month"].max(), FEATURES]
    version = registry.register(model, meta, reference=reference)
    meta["version"] = version
    db.log_event(conn, "registered", version, {"reason": reason, "test_pr_auc": meta["test"]["pr_auc"]})
    log_run(config, f"train_v{version}", params=meta["params"],
            metrics={"test": meta["test"], "threshold": meta["threshold"], "cv_score": meta["cv_score"],
                     "n_rows": meta["n_rows"]},
            tags={"reason": reason, "version": version})
    return model, meta


def _retrain(conn, registry: ModelRegistry, config: dict, champion, champion_meta: dict,
             reasons: list[str]) -> str:
    """Train a challenger on all labelled data and compare it with the champion on fresh data."""
    seed = config["split"]["random_state"]
    labelled = db.load_customers(conn, labelled_only=True)

    # Customers who arrived after the champion was trained: it has never seen them.
    # Half are held back from the challenger too, so both models are compared on unseen data.
    fresh = labelled[labelled["month"] > max(champion_meta["data_months"])]
    if fresh["churn"].sum() < 10:
        return "Retraining skipped: not enough new churners since the last model to compare fairly"
    _, holdout = train_test_split(fresh, test_size=0.5, stratify=fresh["churn"], random_state=seed)
    train_data = labelled.drop(index=holdout.index)

    challenger, meta = train_and_register(conn, registry, config, train_data,
                                          params=champion_meta["params"],
                                          reason="retrain: " + ", ".join(reasons))
    y = holdout["churn"]
    champ_score = average_precision_score(y, champion.predict_proba(holdout[FEATURES])[:, 1])
    chall_score = average_precision_score(y, challenger.predict_proba(holdout[FEATURES])[:, 1])
    details = {"champion_pr_auc": round(champ_score, 4), "challenger_pr_auc": round(chall_score, 4)}

    if chall_score >= champ_score + config["monitoring"]["min_improvement"]:
        registry.promote(meta["version"])
        db.log_event(conn, "promoted", meta["version"], details)
        return (f"Retrained ({', '.join(reasons)}): v{meta['version']} promoted "
                f"(PR-AUC {chall_score:.3f} vs v{champion_meta['version']} {champ_score:.3f})")
    db.log_event(conn, "rejected", meta["version"], details)
    return (f"Retrained ({', '.join(reasons)}): v{meta['version']} not promoted "
            f"(PR-AUC {chall_score:.3f} vs v{champion_meta['version']} {champ_score:.3f})")


def run_month(month: int, conn, registry: ModelRegistry, config: dict) -> dict:
    batch_path = resolve(config["simulation"]["batches_dir"]) / f"month_{month:02d}.csv"
    batch = pd.read_csv(batch_path)
    reports_dir = resolve(config["reports"]["dir"])
    mon, biz = config["monitoring"], config["business"]

    # 1. ingest (validation happens in prepare_features)
    features = batch.drop(columns="churn")
    prepare_features(features)
    db.insert_customers(conn, features, month, include_labels=False)

    # 2. score with the champion and hand the target list to the retention team
    model, meta = registry.load()
    scored = score(features, model, meta["threshold"])
    db.save_predictions(conn, scored.assign(month=month, model_version=meta["version"]))
    reports_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(reports_dir / f"targets_month_{month:02d}.csv", index=False)

    # 3. monitor data drift against the champion's training data
    reference = registry.load_reference()
    table = drift_table(reference, features[FEATURES], FEATURES, mon["psi_threshold"])
    drift = drift_summary(table, mon["drift_share_threshold"])

    # 4. labels arrive at the end of the month
    db.add_labels(conn, batch[["customer_id", "churn"]])

    # 5. evaluate this month's predictions
    scored = scored.merge(batch[["customer_id", "churn"]], on="customer_id")
    perf = evaluate(scored["churn"], scored["churn_probability"], meta["threshold"],
                    value=scored["customer_value"], business=biz)

    # 6. decide whether to retrain
    reasons = []
    if drift["data_drift"]:
        reasons.append("data drift")
    expected = meta["test"]["pr_auc"]
    if perf["pr_auc"] is not None and perf["pr_auc"] < expected - mon["max_pr_auc_drop"]:
        reasons.append("performance drop")
    action = _retrain(conn, registry, config, model, meta, reasons) if reasons else "No action: model healthy"

    report = month_report(reports_dir / f"month_{month:02d}.html", month, meta["version"], len(batch),
                          perf, expected, drift, table, reference, features, action)
    row = {
        "month": month, "model_version": meta["version"], "n_customers": len(batch),
        "churn_rate": round(float(batch["churn"].mean()), 4),
        "drift_share": drift["drift_share"], "data_drift": int(drift["data_drift"]),
        "roc_auc": perf["roc_auc"], "pr_auc": perf["pr_auc"],
        "precision": perf["precision"], "recall": perf["recall"],
        "n_targeted": perf["campaign"]["n_targeted"],
        "campaign_profit": perf["campaign"]["campaign_profit"],
        "random_profit": perf["campaign"]["random_profit"],
        "action": action, "report_path": str(report),
    }
    db.save_monitoring(conn, row)
    log_run(config, f"month_{month:02d}", params={"model_version": meta["version"]},
            metrics={k: v for k, v in row.items() if isinstance(v, (int, float))},
            artifacts=[report], tags={"type": "monthly_cycle", "action": action})
    return row
