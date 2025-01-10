import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def client():
    api_client = TestClient(app)
    return api_client

def test_get(client):
    r = client.get("/")
    assert r.status_code == 200

def test_below(client):
    r = client.post("/predict/", json={
                            "age":54,
                            "workclass":"Private",
                            "education":"HS-grad",
                            "marital-status":"Widowed",
                            "occupation":"Exec-managerial",
                            "relationship": "Unmarried",
                            "race":"White",
                            "sex":"Female",
                            "hours-per-week":45,
                            "native-country":"United-States"
                        }
                    )
    assert r.status_code == 200
    assert r.json() == {"predictions": "<=50K"}

def test_above(client):
    r = client.post("/predict/", json={
                            "age":43,
                            "workclass":"Federal-gov",
                            "education":"Bachelors",
                            "marital-status":"Never-married",
                            "occupation":"Exec-managerial",
                            "relationship": "Not-in-family",
                            "race":"White",
                            "sex":"Male",
                            "hours-per-week":40,
                            "native-country":"United-States"
                        }
                    )
    assert r.status_code == 200
    assert r.json() == {"predictions": ">50K"}