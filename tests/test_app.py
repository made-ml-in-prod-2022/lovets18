"""module for testing app server"""
import os
import pytest
from fastapi.testclient import TestClient

from tests.constants import CONFIG  # pylint:disable=E0401, E0611

os.environ["PATH_TO_CONFIG"] = CONFIG
from online_inference.app import app  # pylint:disable=C0413, E0401


@pytest.fixture
def client():
    """client gen"""
    with TestClient(app) as client:  # pylint:disable=W0621
        yield client


def test_index(client):  # pylint:disable=W0621
    """test index page"""
    response = client.get("/")
    assert 200 == response.status_code
    assert response.json() == "HM2"


def test_health(client):  # pylint:disable=W0621
    """test health func"""
    response = client.get("/health")
    assert response.status_code == 200


def test_predict(client):  # pylint:disable=W0621
    """test predict correct data"""
    request_data = [41, 1, 1, 135, 203, 0, 0, 132, 0, 0.0, 1, 0, 1]
    request_features = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    ]

    response = client.get(
        "/predict",
        json={"data": [request_data], "features": request_features},
    )
    assert response.status_code == 200
    assert response.json()[0] == {"target": 0}


def test_predict_no_data(client):  # pylint:disable=W0621
    """test predict incorrect data"""
    request_data = []
    request_features = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    ]

    response = client.get(
        "/predict",
        json={"data": [request_data], "features": request_features},
    )
    assert response.status_code == 200
    assert response.json()[0] == {"target": -1}


if __name__ == "__main__":
    pytest.main()
