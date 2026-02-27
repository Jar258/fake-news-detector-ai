import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_fake():
    response = client.post("/predict", json={"content": "Trump arrested!"})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] in ["FAKE", "REAL"]
    assert 0 <= data["confidence"] <= 100

def test_bad_input():
    response = client.post("/predict", json={"content": ""})
    assert response.status_code == 422  # Pydantic validation fail
