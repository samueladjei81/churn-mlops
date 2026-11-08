import pandas as pd
import pytest

from churn import db
from churn.data import FEATURES, prepare_training_data
from churn.registry import ModelRegistry, NoChampionError


@pytest.fixture
def customers(raw_df):
    X, y = prepare_training_data(raw_df)
    df = X.assign(churn=y)
    df.insert(0, "customer_id", [f"C{i:05d}" for i in range(len(df))])
    return df


def test_database_stores_customers_without_labels_then_adds_them(tmp_path, customers):
    conn = db.connect(tmp_path / "t.db")
    db.insert_customers(conn, customers.head(50), month=1, include_labels=False)
    assert db.load_customers(conn, labelled_only=True).empty
    assert len(db.load_customers(conn, months=[1])) == 50

    db.add_labels(conn, customers.head(50)[["customer_id", "churn"]])
    labelled = db.load_customers(conn, labelled_only=True)
    assert len(labelled) == 50
    assert labelled["churn"].tolist() == customers.head(50)["churn"].tolist()
    assert list(labelled.columns) == ["customer_id", "month", *FEATURES, "churn"]


def test_database_monitoring_and_events(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.save_monitoring(conn, {"month": 1, "model_version": 1, "pr_auc": 0.8, "action": "No action"})
    db.save_monitoring(conn, {"month": 1, "model_version": 2, "pr_auc": 0.9, "action": "x"})  # replaces
    mon = db.read_table(conn, "monitoring")
    assert len(mon) == 1 and mon.loc[0, "model_version"] == 2
    db.log_event(conn, "promoted", 2, {"why": "test"})
    assert db.read_table(conn, "model_events").loc[0, "event"] == "promoted"


def test_registry_register_promote_load(tmp_path):
    reg = ModelRegistry(tmp_path / "models")
    with pytest.raises(NoChampionError):
        reg.load()
    ref = pd.DataFrame({"a": [1, 2]})
    v1 = reg.register({"fake": "model"}, {"test": {"pr_auc": 0.8}}, ref)
    v2 = reg.register({"fake": "model2"}, {"test": {"pr_auc": 0.9}}, ref)
    assert (v1, v2) == (1, 2)
    reg.promote(1)
    model, meta = reg.load()
    assert model == {"fake": "model"} and meta["version"] == 1
    reg.promote(2)
    assert reg.champion_version() == 2
    assert [v["champion"] for v in reg.versions()] == [False, True]
    assert reg.load_reference().equals(ref)
    with pytest.raises(ValueError):
        reg.promote(99)
