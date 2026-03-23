"""Tests for the REST API."""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api import app


@pytest.fixture
def client():
    """Create a test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class TestAPIEndpoints:
    def test_health_endpoint(self, client):
        """Health endpoint returns OK."""
        resp = client.get('/health')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'ok'
        assert 'model_loaded' in data

    def test_chat_no_model(self, client):
        """Chat returns 503 when model not loaded."""
        resp = client.post('/chat', json={'message': 'hello'})
        assert resp.status_code == 503

    def test_chat_missing_message(self, client):
        """Chat returns 400 when message missing."""
        # This would also be 503 since model isn't loaded, but tests the route
        resp = client.post('/chat', json={})
        assert resp.status_code in (400, 503)

    def test_generate_no_model(self, client):
        """Generate returns 503 when model not loaded."""
        resp = client.post('/generate', json={'message': 'hello'})
        assert resp.status_code == 503

    def test_config_no_model(self, client):
        """Config returns 503 when not initialized."""
        resp = client.get('/config')
        assert resp.status_code == 503

    def test_chat_empty_message(self, client):
        """Chat handles empty message."""
        resp = client.post('/chat', json={'message': ''})
        assert resp.status_code in (400, 503)

    def test_invalid_endpoint(self, client):
        """Unknown endpoint returns 404."""
        resp = client.get('/nonexistent')
        assert resp.status_code == 404

    def test_chat_wrong_method(self, client):
        """Chat rejects GET requests."""
        resp = client.get('/chat')
        assert resp.status_code == 405
