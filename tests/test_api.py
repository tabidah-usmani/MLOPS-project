import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

MODEL_EXISTS = os.path.exists("models/model.pkl")

@pytest.fixture
def client():
    if not MODEL_EXISTS:
        pytest.skip("Model file not found - skipping API tests in CI")
    from api.app import app
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_check(client):
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'healthy'

def test_predict_fake_news(client):
    payload = {"text": "BREAKING: Aliens have landed in Washington DC!"}
    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert 'label' in data
    assert data['label'] in ['FAKE', 'REAL']
    assert 'confidence' in data
    assert 0.0 <= data['confidence'] <= 1.0

def test_predict_missing_text(client):
    response = client.post('/predict', json={})
    assert response.status_code == 400

def test_predict_empty_text(client):
    response = client.post('/predict', json={"text": ""})
    assert response.status_code == 400

def test_metrics_endpoint(client):
    response = client.get('/metrics')
    assert response.status_code == 200