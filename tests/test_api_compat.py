"""Tests for api_compat.py — OpenAI-compatible API endpoints."""

import json
import pytest

try:
    from flask import Flask
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

if HAS_FLASK:
    from api_compat import create_app


@pytest.mark.skipif(not HAS_FLASK, reason="Flask not installed")
class TestApiCompat:
    @pytest.fixture
    def client(self):
        app = create_app()
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client

    # Health
    def test_health(self, client):
        resp = client.get('/health')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"

    def test_v1_health(self, client):
        resp = client.get('/v1/health')
        assert resp.status_code == 200

    # Models
    def test_list_models(self, client):
        resp = client.get('/v1/models')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        # Should always have at least the local-model entry
        assert any(m["id"] == "local-model" for m in data["data"])

    # Chat completions
    def test_chat_completions(self, client):
        payload = {
            "model": "local-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "max_tokens": 50,
        }
        resp = client.post('/v1/chat/completions',
                           data=json.dumps(payload),
                           content_type='application/json')
        assert resp.status_code == 200
        data = resp.get_json()
        assert "choices" in data
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "usage" in data
        assert data["object"] == "chat.completion"

    def test_chat_completions_with_system(self, client):
        payload = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ],
        }
        resp = client.post('/v1/chat/completions',
                           data=json.dumps(payload),
                           content_type='application/json')
        assert resp.status_code == 200

    def test_chat_completions_empty_messages(self, client):
        payload = {"messages": []}
        resp = client.post('/v1/chat/completions',
                           data=json.dumps(payload),
                           content_type='application/json')
        assert resp.status_code == 200

    # Embeddings
    def test_embeddings_string(self, client):
        payload = {"input": "Hello world", "model": "local-model"}
        resp = client.post('/v1/embeddings',
                           data=json.dumps(payload),
                           content_type='application/json')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert len(data["data"][0]["embedding"]) == 128

    def test_embeddings_list(self, client):
        payload = {"input": ["Hello", "World"], "model": "local-model"}
        resp = client.post('/v1/embeddings',
                           data=json.dumps(payload),
                           content_type='application/json')
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data["data"]) == 2

    # Image generations
    def test_image_generations(self, client):
        payload = {"prompt": "a sunset", "n": 1, "size": "512x512"}
        resp = client.post('/v1/images/generations',
                           data=json.dumps(payload),
                           content_type='application/json')
        assert resp.status_code == 200
        data = resp.get_json()
        assert "data" in data
        assert "created" in data

    # Response structure
    def test_chat_response_has_id(self, client):
        payload = {"messages": [{"role": "user", "content": "test"}]}
        resp = client.post('/v1/chat/completions',
                           data=json.dumps(payload),
                           content_type='application/json')
        data = resp.get_json()
        assert "id" in data
        assert data["id"].startswith("chatcmpl-")
