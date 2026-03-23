"""Tests for the external API client."""

import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from external_api import (
    ExternalAPIClient, PROVIDERS, list_providers, api_call,
    is_provider_configured, find_available_provider,
    refresh_provider_models, _load_api_cache, _save_api_cache,
    _api_cache_is_stale, _fetch_openai_models, _fetch_ollama_models,
    _fetch_google_models, _fetch_deepseek_models,
)


class TestProviders:
    def test_providers_exist(self):
        assert "openai" in PROVIDERS
        assert "anthropic" in PROVIDERS
        assert "google" in PROVIDERS
        assert "deepseek" in PROVIDERS
        assert "ollama" in PROVIDERS
        assert "custom" in PROVIDERS

    def test_all_six_providers(self):
        assert len(PROVIDERS) == 6

    def test_provider_structure(self):
        for key, info in PROVIDERS.items():
            assert "name" in info
            assert "base_url" in info
            assert "models" in info
            assert isinstance(info["models"], list)

    def test_provider_models_not_empty(self):
        """All providers except custom should have at least one default model."""
        for key, info in PROVIDERS.items():
            if key != "custom":
                assert len(info["models"]) > 0, f"{key} has no default models"

    def test_openai_has_latest_models(self):
        models = PROVIDERS["openai"]["models"]
        assert "gpt-4o" in models
        assert "gpt-4o-mini" in models

    def test_anthropic_has_latest_models(self):
        models = PROVIDERS["anthropic"]["models"]
        # Should have Claude 3.5+ models
        assert any("3.5" in m or "3.7" in m or "sonnet-4" in m for m in models)

    def test_google_has_gemini_models(self):
        models = PROVIDERS["google"]["models"]
        assert any("gemini" in m for m in models)

    def test_deepseek_has_models(self):
        models = PROVIDERS["deepseek"]["models"]
        assert "deepseek-chat" in models

    def test_ollama_has_local_models(self):
        models = PROVIDERS["ollama"]["models"]
        assert any("llama" in m for m in models)

    def test_list_providers(self, capsys):
        list_providers()
        captured = capsys.readouterr()
        assert "openai" in captured.out.lower() or "OpenAI" in captured.out
        assert "anthropic" in captured.out.lower() or "Anthropic" in captured.out
        assert "google" in captured.out.lower() or "Google" in captured.out
        assert "deepseek" in captured.out.lower() or "DeepSeek" in captured.out
        assert "ollama" in captured.out.lower() or "Ollama" in captured.out


class TestIsProviderConfigured:
    def test_openai_not_configured_by_default(self):
        if not os.environ.get("OPENAI_API_KEY"):
            assert is_provider_configured("openai") is False

    def test_anthropic_not_configured_by_default(self):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            assert is_provider_configured("anthropic") is False

    def test_google_not_configured_by_default(self):
        if not os.environ.get("GOOGLE_API_KEY"):
            assert is_provider_configured("google") is False

    def test_deepseek_not_configured_by_default(self):
        if not os.environ.get("DEEPSEEK_API_KEY"):
            assert is_provider_configured("deepseek") is False

    @patch('external_api._get_api_key', return_value="test-key-123")
    def test_openai_configured_with_key(self, mock_key):
        assert is_provider_configured("openai") is True

    @patch('external_api._get_api_key', return_value="test-key-123")
    def test_google_configured_with_key(self, mock_key):
        assert is_provider_configured("google") is True

    @patch('external_api.requests.get')
    def test_ollama_configured_when_running(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp
        assert is_provider_configured("ollama") is True

    @patch('external_api.requests.get', side_effect=Exception("Connection refused"))
    def test_ollama_not_configured_when_down(self, mock_get):
        assert is_provider_configured("ollama") is False

    @patch('external_api._get_api_key', return_value=None)
    def test_custom_needs_key_and_url(self, mock_key):
        assert is_provider_configured("custom") is False


class TestFindAvailableProvider:
    @patch('external_api.is_provider_configured', return_value=False)
    def test_no_providers_available(self, mock_configured, capsys):
        result = find_available_provider()
        assert result is None
        captured = capsys.readouterr()
        assert "Skipping" in captured.out

    @patch('external_api.is_provider_configured')
    def test_finds_first_configured(self, mock_configured, capsys):
        mock_configured.side_effect = lambda p: p == "anthropic"
        result = find_available_provider(["openai", "anthropic", "ollama"])
        assert result == "anthropic"

    @patch('external_api.is_provider_configured')
    def test_respects_preferred_order(self, mock_configured):
        mock_configured.side_effect = lambda p: p in ("openai", "ollama")
        result = find_available_provider(["ollama", "openai"])
        assert result == "ollama"

    @patch('external_api.is_provider_configured', return_value=False)
    def test_skips_all_unconfigured(self, mock_configured, capsys):
        result = find_available_provider(["openai", "anthropic"])
        assert result is None
        captured = capsys.readouterr()
        assert "Skipping openai" in captured.out
        assert "Skipping anthropic" in captured.out

    def test_ignores_unknown_providers(self):
        with patch('external_api.is_provider_configured', return_value=False):
            result = find_available_provider(["nonexistent", "fake"])
            assert result is None


class TestAPICache:
    def test_load_empty_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr("external_api._API_CACHE_FILE",
                            str(tmp_path / "nonexistent.json"))
        cache = _load_api_cache()
        assert cache == {}

    def test_save_and_load_cache(self, tmp_path, monkeypatch):
        cache_file = str(tmp_path / "cache.json")
        cache_dir = str(tmp_path)
        monkeypatch.setattr("external_api._API_CACHE_FILE", cache_file)
        monkeypatch.setattr("external_api._API_CACHE_DIR", cache_dir)

        data = {"last_updated": "2026-01-01T00:00:00", "providers": {"openai": ["gpt-4o"]}}
        _save_api_cache(data)
        loaded = _load_api_cache()
        assert loaded["providers"]["openai"] == ["gpt-4o"]

    def test_stale_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr("external_api._API_CACHE_FILE",
                            str(tmp_path / "nonexistent.json"))
        assert _api_cache_is_stale() is True


class TestFetchModels:
    @patch('external_api.requests.get')
    def test_fetch_openai_models(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {"id": "gpt-4o"},
                {"id": "gpt-4o-mini"},
                {"id": "gpt-3.5-turbo"},
                {"id": "dall-e-3"},  # Should be filtered out
            ]
        }
        mock_get.return_value = mock_resp

        models = _fetch_openai_models("key", "https://api.openai.com/v1")
        assert "gpt-4o" in models
        assert "gpt-3.5-turbo" in models
        # dall-e should be excluded by chat filter
        assert "dall-e-3" not in models

    @patch('external_api.requests.get')
    def test_fetch_ollama_models(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [
                {"name": "llama3.2:latest"},
                {"name": "mistral:latest"},
            ]
        }
        mock_get.return_value = mock_resp

        models = _fetch_ollama_models("http://localhost:11434")
        assert "llama3.2" in models
        assert "mistral" in models

    @patch('external_api.requests.get')
    def test_fetch_google_models(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [
                {"name": "models/gemini-2.0-flash", "supportedGenerationMethods": ["generateContent"]},
                {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},
            ]
        }
        mock_get.return_value = mock_resp

        models = _fetch_google_models("key", "https://generativelanguage.googleapis.com/v1beta")
        assert "gemini-2.0-flash" in models
        # Embedding model should be filtered out
        assert "text-embedding-004" not in models

    @patch('external_api.requests.get', side_effect=Exception("timeout"))
    def test_fetch_openai_models_timeout(self, mock_get):
        models = _fetch_openai_models("key", "https://api.openai.com/v1")
        assert models == []

    @patch('external_api.requests.get')
    def test_fetch_deepseek_models(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [{"id": "deepseek-chat"}, {"id": "deepseek-reasoner"}]
        }
        mock_get.return_value = mock_resp

        models = _fetch_deepseek_models("key", "https://api.deepseek.com/v1")
        assert "deepseek-chat" in models
        assert "deepseek-reasoner" in models

    @patch('external_api.requests.get', side_effect=Exception("timeout"))
    def test_fetch_deepseek_models_timeout(self, mock_get):
        models = _fetch_deepseek_models("key", "https://api.deepseek.com/v1")
        assert models == []

    @patch('external_api.requests.get', side_effect=Exception("timeout"))
    def test_fetch_ollama_models_timeout(self, mock_get):
        models = _fetch_ollama_models("http://localhost:11434")
        assert models == []


class TestRefreshProviderModels:
    @patch('external_api.is_provider_configured', return_value=False)
    @patch('external_api._get_api_key', return_value=None)
    def test_refresh_with_no_providers(self, mock_key, mock_configured,
                                       tmp_path, monkeypatch):
        cache_file = str(tmp_path / "cache.json")
        cache_dir = str(tmp_path)
        monkeypatch.setattr("external_api._API_CACHE_FILE", cache_file)
        monkeypatch.setattr("external_api._API_CACHE_DIR", cache_dir)

        results = refresh_provider_models(force=True, verbose=False)
        assert isinstance(results, dict)


class TestExternalAPIClient:
    def test_create_client(self):
        client = ExternalAPIClient("openai", model="gpt-4o", api_key="test-key")
        assert client.provider == "openai"
        assert client.model == "gpt-4o"
        assert client.api_key == "test-key"

    def test_create_client_default_model(self):
        client = ExternalAPIClient("openai", api_key="test-key")
        assert client.model == PROVIDERS["openai"]["models"][0]

    def test_create_google_client(self):
        client = ExternalAPIClient("google", model="gemini-2.0-flash", api_key="test-key")
        assert client.provider == "google"
        assert client.model == "gemini-2.0-flash"

    def test_create_deepseek_client(self):
        client = ExternalAPIClient("deepseek", model="deepseek-chat", api_key="test-key")
        assert client.provider == "deepseek"
        assert client.model == "deepseek-chat"

    def test_invalid_provider(self):
        with pytest.raises(ValueError):
            ExternalAPIClient("nonexistent")

    def test_conversation_history(self):
        client = ExternalAPIClient("openai", api_key="test-key")
        assert client.conversation_history == []

    def test_clear_history(self):
        client = ExternalAPIClient("openai", api_key="test-key")
        client.conversation_history = [{"role": "user", "content": "test"}]
        client.clear_history()
        assert client.conversation_history == []

    def test_ollama_no_key_warning(self, capsys):
        """Ollama should not warn about missing API key."""
        client = ExternalAPIClient("ollama", model="llama3.3")
        captured = capsys.readouterr()
        assert "Warning" not in captured.out

    @patch('external_api.requests.post')
    def test_openai_chat(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Hello from GPT!"}}]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = ExternalAPIClient("openai", model="gpt-4o", api_key="test-key")
        response = client.chat("Hello")

        assert response == "Hello from GPT!"
        assert len(client.conversation_history) == 2
        assert client.conversation_history[0]["role"] == "user"
        assert client.conversation_history[1]["role"] == "assistant"
        mock_post.assert_called_once()

    @patch('external_api.requests.post')
    def test_anthropic_chat(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "content": [{"text": "Hello from Claude!"}]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = ExternalAPIClient("anthropic", model="claude-3.5-sonnet-20241022",
                                    api_key="test-key")
        response = client.chat("Hello")

        assert response == "Hello from Claude!"
        mock_post.assert_called_once()

    @patch('external_api.requests.post')
    def test_google_chat(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello from Gemini!"}]
                }
            }]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = ExternalAPIClient("google", model="gemini-2.0-flash", api_key="test-key")
        response = client.chat("Hello")

        assert response == "Hello from Gemini!"
        mock_post.assert_called_once()

    @patch('external_api.requests.post')
    def test_deepseek_chat(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Hello from DeepSeek!"}}]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = ExternalAPIClient("deepseek", model="deepseek-chat", api_key="test-key")
        response = client.chat("Hello")

        assert response == "Hello from DeepSeek!"
        mock_post.assert_called_once()

    @patch('external_api.requests.post')
    def test_ollama_chat(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"content": "Hello from Llama!"}
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = ExternalAPIClient("ollama", model="llama3.3")
        response = client.chat("Hello")

        assert response == "Hello from Llama!"
        mock_post.assert_called_once()

    @patch('external_api.requests.post')
    def test_chat_with_system_prompt(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "response"}}]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = ExternalAPIClient("openai", model="gpt-4o", api_key="test-key")
        client.chat("Hello", system_prompt="You are helpful")

        call_args = mock_post.call_args
        messages = call_args[1]["json"]["messages"] if "json" in call_args[1] else call_args[0]
        assert any(m["role"] == "system" for m in messages)

    def test_chat_api_error(self):
        client = ExternalAPIClient("openai", model="gpt-4o", api_key="bad-key",
                                    base_url="http://localhost:1")
        response = client.chat("Hello")
        assert "[API Error" in response

    @patch('external_api.requests.post')
    def test_multi_turn_conversation(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "response"}}]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = ExternalAPIClient("openai", model="gpt-4o", api_key="test-key")
        client.chat("First message")
        client.chat("Second message")

        assert len(client.conversation_history) == 4  # 2 user + 2 assistant

    @patch('external_api.requests.post')
    def test_custom_provider_chat(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Custom response"}}]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = ExternalAPIClient("custom", model="my-model", api_key="key",
                                    base_url="http://my-server.com/v1")
        response = client.chat("Hello")
        assert response == "Custom response"


class TestApiCall:
    @patch('external_api.requests.post')
    def test_single_call(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Answer"}}]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result = api_call("openai", "gpt-4o", "What is AI?", api_key="test-key")
        assert result == "Answer"

    @patch('external_api.requests.post')
    def test_single_call_google(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "Gemini answer"}]
                }
            }]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result = api_call("google", "gemini-2.0-flash", "What is AI?", api_key="test-key")
        assert result == "Gemini answer"
