"""Training: tune, choose a profit-maximising threshold, evaluate, refit.

`fit_model` is used both for the first model (with hyperparameter tuning)
and for retraining (reusing the champion's parameters, which is much faster).
"""

from datetime import datetime, timezone

import pandas as pd
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_predict,
                                     train_test_split)

from churn import __version__
from churn.business import best_threshold
from churn.data import FEATURES
from churn.evaluate import evaluate
from churn.pipeline import build_pipeline


def _cv(config: dict) -> StratifiedKFold:
    return StratifiedKFold(n_splits=config["training"]["cv_folds"], shuffle=True,
                           random_state=config["split"]["random_state"])


def tune(X: pd.DataFrame, y: pd.Series, config: dict) -> tuple[dict, float]:
    """Grid search over config.training.param_grid. SMOTE runs inside each fold."""
    seed = config["split"]["random_state"]
    search = GridSearchCV(build_pipeline(seed), config["training"]["param_grid"], cv=_cv(config),
                          scoring=config["training"]["scoring"], n_jobs=-1)
    search.fit(X, y)
    return search.best_params_, float(search.best_score_)


def choose_threshold(X: pd.DataFrame, y: pd.Series, params: dict, config: dict) -> tuple[float, float]:
    """Pick the cut-off that maximises campaign profit, using out-of-fold predictions."""
    seed = config["split"]["random_state"]
    oof = cross_val_predict(build_pipeline(seed, params), X, y, cv=_cv(config),
                            method="predict_proba", n_jobs=-1)[:, 1]
    biz = config["business"]
    return best_threshold(y, oof, X["customer_value"], biz["offer_cost"], biz["save_rate"])


def fit_model(X: pd.DataFrame, y: pd.Series, config: dict, params: dict | None = None,
              reason: str = "initial training"):
    """Train a model and return (model, metadata).

    1. hold out a test set
    2. tune (if no params given) and choose a threshold on the training part only
    3. measure on the test set
    4. refit on all the data given, for use in production
    """
    seed = config["split"]["random_state"]
    X, y = X[FEATURES].reset_index(drop=True), pd.Series(y).reset_index(drop=True).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=config["split"]["test_size"],
                                              stratify=y, random_state=seed)
    cv_score = None
    if params is None:
        params, cv_score = tune(X_tr, y_tr, config)
    threshold, oof_profit = choose_threshold(X_tr, y_tr, params, config)

    holdout_model = build_pipeline(seed, params).fit(X_tr, y_tr)
    test = evaluate(y_te, holdout_model.predict_proba(X_te)[:, 1], threshold,
                    value=X_te["customer_value"], business=config["business"])

    final = build_pipeline(seed, params).fit(X, y)
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "package_version": __version__,
        "reason": reason,
        "features": FEATURES,
        "params": params,
        "cv_score": round(cv_score, 4) if cv_score is not None else None,
        "cv_scoring": config["training"]["scoring"],
        "threshold": threshold,
        "n_rows": len(X),
        "churn_rate": round(float(y.mean()), 4),
        "test": test,
    }
    return final, metadata
