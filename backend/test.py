import pytest
from app import app  # Import our Flask app
import json

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict_valid_data(client):
    data = {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50,
    }
    response = client.post('/predict', json=data)
    assert response.status_code == 200
    assert 'prediction' in json.loads(response.data)

def test_predict_invalid_data(client):
    data = {
        "Pregnancies": "invalid",
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50,
    }
    response = client.post('/predict', json=data)
    assert response.status_code == 400
    assert 'error' in json.loads(response.data)