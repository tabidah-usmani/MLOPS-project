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
    payload = {"text": "BREAKING NEWS Scientists have discovered that the government has been secretly poisoning the water supply with chemicals designed to control the population and keep citizens docile and compliant according to multiple whistleblower sources who have come forward with shocking documents proving this agenda has been ongoing for decades"}
    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert 'label' in data
    assert data['label'] in ['FAKE', 'REAL']
    assert 'confidence' in data
    assert 'word_count' in data
    assert data['word_count'] >= 30

def test_predict_short_text_warning(client):
    payload = {"text": "BREAKING: Aliens have landed!"}
    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert 'warning' in data
    assert data['word_count'] < 30
    
def test_predict_missing_text(client):
    response = client.post('/predict', json={})
    assert response.status_code == 400

def test_predict_empty_text(client):
    response = client.post('/predict', json={"text": ""})
    assert response.status_code == 400

def test_metrics_endpoint(client):
    response = client.get('/metrics')
    assert response.status_code == 200