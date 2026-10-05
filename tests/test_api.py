import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.train import train


@pytest.fixture(scope="module")
def client():
    # The context manager runs the lifespan (loads / trains the model).
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_setosa(client):
    r = client.post(
        "/predict",
        json={"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
    )
    assert r.status_code == 200
    assert r.json()["prediction"] == "setosa"


def test_predict_virginica(client):
    r = client.post(
        "/predict",
        json={"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.8, "petal_width": 2.2},
    )
    assert r.status_code == 200
    assert r.json()["prediction"] == "virginica"


def test_predict_response_format(client):
    r = client.post(
        "/predict",
        json={"sepal_length": 5.9, "sepal_width": 3.0, "petal_length": 4.2, "petal_width": 1.5},
    )
    body = r.json()
    assert set(body) == {"prediction", "confidence", "probabilities"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert abs(sum(body["probabilities"].values()) - 1.0) < 0.01


def test_predict_missing_field_returns_422(client):
    r = client.post("/predict", json={"sepal_length": 5.1})
    assert r.status_code == 422


def test_predict_invalid_value_returns_422(client):
    r = client.post(
        "/predict",
        json={"sepal_length": -1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
    )
    assert r.status_code == 422


def test_model_accuracy_is_reasonable(tmp_path):
    acc = train(tmp_path / "m.joblib")
    assert acc >= 0.85
