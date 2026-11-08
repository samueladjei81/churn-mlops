"""Model evaluation metrics (technical and business)."""

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

from churn.business import campaign_summary


def evaluate(y_true, proba, threshold: float = 0.5, value=None, business: dict | None = None) -> dict:
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    both_classes = len(np.unique(y_true)) == 2
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    metrics = {
        "threshold": threshold,
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4) if both_classes else None,
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4) if both_classes else None,
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    if value is not None and business is not None:
        metrics["campaign"] = campaign_summary(y_true, proba, value, threshold,
                                               business["offer_cost"], business["save_rate"])
    return metrics
