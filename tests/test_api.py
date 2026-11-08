"""API tests. Skipped automatically if FastAPI is not installed."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

import churn.api as api  # noqa: E402
from churn.cycle import train_and_register  # noqa: E402
from churn import db  # noqa: E402
from churn.data import prepare_training_data  # noqa: E402
from churn.registry import ModelRegistry  # noqa: E402

CUSTOMER = {"customer_id": "C1", "call_failure": 8, "complains": 1, "subscription_length": 32,
            "charge_amount": 0, "seconds_of_use": 1200, "distinct_called_numbers": 10,
            "age_group": 3, "tariff_plan": 1, "status": 2, "customer_value": 150.5}


@pytest.fixture
def client(small_config, raw_df, monkeypatch):
    monkeypatch.setattr(api, "load_config", lambda: small_config)
    api._state.clear()
    return TestClient(api.app)


@pytest.fixture
def trained(small_config, raw_df):
    X, y = prepare_training_data(raw_df)
    data = X.assign(churn=y, month=0)
    conn = db.connect(small_config["database"]["path"])
    reg = ModelRegistry(small_config["registry"]["dir"])
    _, meta = train_and_register(conn, reg, small_config, data)
    reg.promote(meta["version"])


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_predict_without_model_returns_503(client):
    assert client.post("/predict", json=CUSTOMER).status_code == 503


def test_predict(client, trained):
    r = client.post("/predict", json=CUSTOMER)
    assert r.status_code == 200
    body = r.json()
    assert body["customer_id"] == "C1" and 0 <= body["churn_probability"] <= 1
    assert body["model_version"] == 1


def test_predict_batch_and_model_info(client, trained):
    r = client.post("/predict/batch", json=[CUSTOMER, {**CUSTOMER, "customer_id": "C2", "status": 1}])
    assert r.status_code == 200 and len(r.json()) == 2
    assert client.get("/model").json()["version"] == 1


def test_invalid_input_is_rejected(client, trained):
    assert client.post("/predict", json={**CUSTOMER, "tariff_plan": 3}).status_code == 422
    missing = {k: v for k, v in CUSTOMER.items() if k != "status"}
    assert client.post("/predict", json=missing).status_code == 422
