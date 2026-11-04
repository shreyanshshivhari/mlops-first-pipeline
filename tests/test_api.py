
# Import TestClient
from fastapi.testclient import TestClient

# Import application factory
from main import create_app

import pytest


# ---------------------------------------
# FAKE MODEL
# ---------------------------------------

class FakeIrisModel:

    def predict(self, features):
        # Always return class 0
        return [0]


# ---------------------------------------
# MOCK LOADERS
# ---------------------------------------

def fake_model_loader():
    return FakeIrisModel()


def fake_version_loader():
    return "test-version"


# ---------------------------------------
# CLIENT FIXTURE
# ---------------------------------------

@pytest.fixture
def client():

    app = create_app(
        model_loader=fake_model_loader,
        version_loader=fake_version_loader
    )

    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------
# TEST 1: HEALTH ENDPOINT
# ---------------------------------------

def test_health_endpoint(client):

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy",
        "model_version": "test-version"
    }


# ---------------------------------------
# TEST 2: VALID PREDICTION
# ---------------------------------------

def test_valid_iris_prediction(client):

    payload = {
        "sepal_length": 3.4,
        "sepal_width": 1.4,
        "petal_length": 3.4,
        "petal_width": 0.4
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "class_name" in result
    assert "model_version" in result

    assert result["prediction"] == 0
    assert result["class_name"] == "setosa"
    assert result["model_version"] == "test-version"
