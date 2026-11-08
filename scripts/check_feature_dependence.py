"""How much does the model depend on `status` and `customer_value`?

Retrains the model with and without those features and compares the scores.
Uses the best parameters found by `python -m churn.train` so it runs quickly.

Usage (from the project folder):
    make check-features
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score

from churn.config import load_config, resolve
from churn.data import load_raw, prepare_training_data, standardise_columns
from churn.pipeline import build_pipeline

# Best parameters from your last `python -m churn.train` run
BEST_PARAMS = {
    "model__learning_rate": 0.1,
    "model__max_depth": 4,
    "model__n_estimators": 300,
    "model__subsample": 1.0,
}

VARIANTS = {
    "all features": [],
    "without status": ["status"],
    "without customer_value": ["customer_value"],
    "without both": ["status", "customer_value"],
}


def main() -> None:
    config = load_config()
    seed = config["split"]["random_state"]
    raw = load_raw(resolve(config["data"]["raw_path"]))
    X, y = prepare_training_data(raw)

    # 1. How strongly does status alone separate churners?
    df = standardise_columns(raw).drop_duplicates()
    table = pd.crosstab(df["status"].map({1: "active", 2: "inactive"}), df["churn"],
                        margins=True, normalize="index").round(3)
    table.columns = ["% stayed", "% churned"]
    counts = df["status"].map({1: "active", 2: "inactive"}).value_counts()
    counts["All"] = len(df)
    print("\nChurn rate by status (first 9 months):")
    print(table.assign(customers=counts.astype(int)).to_string())

    # 2. Retrain with and without the features
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["split"]["test_size"], stratify=y, random_state=seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    rows = []
    for name, drop in VARIANTS.items():
        cols = [c for c in X.columns if c not in drop]
        model = build_pipeline(seed).set_params(**BEST_PARAMS)
        cv_res = cross_validate(model, X_train[cols], y_train, cv=cv,
                                scoring=["average_precision", "roc_auc"], n_jobs=-1)
        model.fit(X_train[cols], y_train)
        proba = model.predict_proba(X_test[cols])[:, 1]
        rows.append({
            "variant": name,
            "CV PR-AUC": cv_res["test_average_precision"].mean(),
            "CV PR-AUC std": cv_res["test_average_precision"].std(),
            "Test PR-AUC": average_precision_score(y_test, proba),
            "Test ROC-AUC": roc_auc_score(y_test, proba),
        })
        print(f"done: {name}")

    results = pd.DataFrame(rows).set_index("variant").round(3)
    base = results.loc["all features", "Test PR-AUC"]
    results["PR-AUC change"] = (results["Test PR-AUC"] - base).round(3)
    print("\n", results.to_string())

    out = resolve(config["reports"]["dir"]) / "feature_dependence.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
