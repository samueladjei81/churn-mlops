import pandas as pd
import pytest

from churn.data import (FEATURES, DataValidationError, prepare_features,
                        prepare_training_data, standardise_columns, validate)


def test_columns_are_renamed_to_snake_case(raw_df):
    df = standardise_columns(raw_df)
    assert "call_failure" in df.columns
    assert "Call  Failure" not in df.columns


def test_training_data_drops_duplicates(raw_df):
    with_dups = pd.concat([raw_df, raw_df.head(10)])
    X, y = prepare_training_data(with_dups)
    assert len(X) == len(raw_df.drop_duplicates())
    assert len(X) == len(y)


def test_training_data_returns_only_model_features(raw_df):
    X, _ = prepare_training_data(raw_df)
    assert list(X.columns) == FEATURES
    for dropped in ["age", "frequency_of_use", "frequency_of_sms", "churn"]:
        assert dropped not in X.columns


def test_prepare_features_works_without_target(raw_df):
    X = prepare_features(raw_df.drop(columns=["Churn"]))
    assert list(X.columns) == FEATURES


def test_missing_column_is_rejected(raw_df):
    with pytest.raises(DataValidationError, match="Missing columns"):
        prepare_features(raw_df.drop(columns=["Status"]))


def test_missing_values_are_rejected(raw_df):
    df = standardise_columns(raw_df)
    df.loc[0, "seconds_of_use"] = None
    with pytest.raises(DataValidationError, match="Missing values"):
        validate(df)


def test_unexpected_category_is_rejected(raw_df):
    df = standardise_columns(raw_df)
    df.loc[0, "tariff_plan"] = 3
    with pytest.raises(DataValidationError, match="tariff_plan"):
        validate(df)


def test_negative_values_are_rejected(raw_df):
    df = standardise_columns(raw_df)
    df.loc[0, "call_failure"] = -1
    with pytest.raises(DataValidationError, match="Negative"):
        validate(df)
