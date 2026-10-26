import json
import os

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from interpret.glassbox import ExplainableBoostingRegressor

from src.features import apply_spec, build_features, fit_spec, standardise_names
from tests.conftest import make_rows


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # Train a tiny EBM on synthetic ads so the API can be tested without the real model.
    df = build_features(standardise_names(make_rows(300)))
    spec = fit_spec(df)
    X = apply_spec(df, spec)
    model = ExplainableBoostingRegressor(
        max_rounds=50, outer_bags=1, interactions=0, early_stopping_rounds=10, random_state=0
    )
    model.fit(X, np.log(df["price_lkr"]))
    folder = tmp_path_factory.mktemp("models")
    joblib.dump(model, folder / "ebm.joblib")
    (folder / "spec.json").write_text(json.dumps(spec))
    (folder / "conformal.json").write_text(json.dumps({"alpha": 0.1, "q_log": 0.25}))
    os.environ["MODELS_DIR"] = str(folder)

    import importlib

    import app.main

    importlib.reload(app.main)  # pick up MODELS_DIR
    with TestClient(app.main.app) as test_client:
        yield test_client


@pytest.fixture
def payload():
    row = make_rows(1).drop(columns=["price_lkr", "seller_name"]).iloc[0].to_dict()
    return {k: (None if v != v else v) for k, v in row.items()}  # NaN -> null


def test_health(client):
    assert client.get("/health").json()["status"] == "healthy"


def test_predict_returns_a_price_inside_its_range(client, payload):
    body = client.post("/predict", json=payload).json()
    assert body["lower_lkr"] <= body["predicted_price_lkr"] <= body["upper_lkr"]
    assert body["predicted_price_lkr"] > 0
    assert body["request_id"]


def test_predict_explains_itself(client, payload):
    body = client.post("/predict", json=payload).json()
    assert len(body["top_terms"]) == 5
    assert all({"term", "contribution_log"} <= set(t) for t in body["top_terms"])
    assert body["model_support"] >= 0


def test_contract_rejects_engine_in_litres(client, payload):
    payload["engine_cc"] = 1.5
    assert client.post("/predict", json=payload).status_code == 422


def test_contract_rejects_unknown_condition(client, payload):
    payload["condition"] = "scrap"
    assert client.post("/predict", json=payload).status_code == 422


def test_feedback(client, payload):
    request_id = client.post("/predict", json=payload).json()["request_id"]
    response = client.post(
        "/feedback", json={"request_id": request_id, "actual_price_lkr": 9_500_000}
    )
    assert response.status_code == 200
