"""Loading, cleaning and validating customer data.

Every route into the model (training, batch scoring, and later the API)
goes through these functions, so the data is always prepared the same way.
"""

from pathlib import Path

import pandas as pd

# Original CSV headers (some contain double spaces) -> clean snake_case names
COLUMN_MAP = {
    "Call  Failure": "call_failure",
    "Complains": "complains",
    "Subscription  Length": "subscription_length",
    "Charge  Amount": "charge_amount",
    "Seconds of Use": "seconds_of_use",
    "Frequency of use": "frequency_of_use",
    "Frequency of SMS": "frequency_of_sms",
    "Distinct Called Numbers": "distinct_called_numbers",
    "Age Group": "age_group",
    "Tariff Plan": "tariff_plan",
    "Status": "status",
    "Age": "age",
    "Customer Value": "customer_value",
    "Churn": "churn",
}

# Features the model uses, in order, after the correlated columns are dropped
FEATURES = [
    "call_failure",
    "complains",
    "subscription_length",
    "charge_amount",
    "seconds_of_use",
    "distinct_called_numbers",
    "age_group",
    "tariff_plan",
    "status",
    "customer_value",
]

# Simple sanity rules: allowed values for categorical columns
ALLOWED_VALUES = {
    "complains": {0, 1},
    "tariff_plan": {1, 2},
    "status": {1, 2},
    "age_group": {1, 2, 3, 4, 5},
}


class DataValidationError(ValueError):
    """Raised when incoming data does not match what the model expects."""


def load_raw(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename original headers to snake_case. Already-clean names are left alone."""
    return df.rename(columns=COLUMN_MAP)


def validate(df: pd.DataFrame, require_target: bool = False, target: str = "churn") -> None:
    """Check columns, missing values and value ranges. Raises DataValidationError."""
    required = FEATURES + ([target] if require_target else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing columns: {missing}")

    nulls = df[required].isnull().sum()
    if nulls.any():
        raise DataValidationError(f"Missing values found: {nulls[nulls > 0].to_dict()}")

    non_numeric = [c for c in required if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise DataValidationError(f"Non-numeric columns: {non_numeric}")

    negative = [c for c in FEATURES if (df[c] < 0).any()]
    if negative:
        raise DataValidationError(f"Negative values in: {negative}")

    for col, allowed in ALLOWED_VALUES.items():
        bad = set(df[col].unique()) - allowed
        if bad:
            raise DataValidationError(f"Unexpected values in '{col}': {sorted(bad)}")

    if require_target:
        bad = set(df[target].unique()) - {0, 1}
        if bad:
            raise DataValidationError(f"Target '{target}' must be 0/1, found {sorted(bad)}")


def prepare_training_data(df: pd.DataFrame, target: str = "churn") -> tuple[pd.DataFrame, pd.Series]:
    """Clean raw training data and split it into features (X) and target (y)."""
    df = standardise_columns(df)
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Dropped {n_before - len(df)} duplicate rows ({len(df)} remain)")
    validate(df, require_target=True, target=target)
    return df[FEATURES], df[target]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean new (unlabelled) customer data for scoring."""
    df = standardise_columns(df)
    validate(df, require_target=False)
    return df[FEATURES]
