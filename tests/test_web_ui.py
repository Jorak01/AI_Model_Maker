"""Tests for web_ui.py — Web UI Flask endpoints."""

import json
import pytest

try:
    from flask import Flask
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

if HAS_FLASK:
    from web_ui import create_web_app, WEB_UI_HTML


@pytest.mark.skipif(not HAS_FLASK, reason="Flask not installed")
class TestWebUI:
    @pytest.fixture
    def client(self):
        app = create_web_app()
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client

    def test_index_returns_html(self, client):
        resp = client.get('/')
        assert resp.status_code == 200
        assert b'AI Model Suite' in resp.data

    def test_index_content_type(self, client):
        resp = client.get('/')
        assert 'text/html' in resp.content_type

    def test_api_chat(self, client):
        payload = {
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.8,
            "max_tokens": 64,
        }
        resp = client.post('/api/chat',
                           data=json.dumps(payload),
                           content_type='application/json')
        assert resp.status_code == 200
        data = resp.get_json()
        assert "response" in data

    def test_api_status(self, client):
        resp = client.get('/api/status')
        assert resp.status_code == 200
        data = resp.get_json()
        assert "status" in data
        assert data["status"] == "Online"
        assert "device" in data
        assert "uptime" in data
        assert "model_count" in data

    def test_api_models(self, client):
        resp = client.get('/api/models')
        assert resp.status_code == 200
        data = resp.get_json()
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_html_has_panels(self):
        assert 'panel-chat' in WEB_UI_HTML
        assert 'panel-status' in WEB_UI_HTML
        assert 'panel-models' in WEB_UI_HTML
        assert 'panel-settings' in WEB_UI_HTML

    def test_html_has_nav_buttons(self):
        assert 'Chat' in WEB_UI_HTML
        assert 'Status' in WEB_UI_HTML
        assert 'Models' in WEB_UI_HTML
        assert 'Settings' in WEB_UI_HTML
