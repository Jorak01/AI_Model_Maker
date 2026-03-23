"""Tests for the external API client."""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from external_api import (
    ExternalAPIClient, PROVIDERS, list_providers, api_call,
    is_provider_configured, find_available_provider
)


class TestProviders:
    def test_providers_exist(self):
        assert "openai" in PROVIDERS
        assert "anthropic" in PROVIDERS
        assert "ollama" in PROVIDERS
        assert "custom" in PROVIDERS

    def test_provider_structure(self):
        for key, info in PROVIDERS.items():
            assert "name" in info
            assert "base_url" in info
            assert "models" in info
            assert isinstance(info["models"], list)

    def test_list_providers(self, capsys):
        list_providers()
        captured = capsys.readouterr()
        assert "openai" in captured.out.lower() or "OpenAI" in captured.out
        assert "anthropic" in captured.out.lower() or "Anthropic" in captured.out
        assert "ollama" in captured.out.lower() or "Ollama" in captured.out


class TestIsProviderConfigured:
    def test_openai_not_configured_by_default(self):
        """OpenAI should not be configured without a key."""
        # Unless user has OPENAI_API_KEY set in env
        if not os.environ.get("OPENAI_API_KEY"):
            assert is_provider_configured("openai") is False

    def test_anthropic_not_configured_by_default(self):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            assert is_provider_configured("anthropic") is False

    @patch('external_api._get_api_key', return_value="test-key-123")
    def test_openai_configured_with_key(self, mock_key):
        assert is_provider_configured("openai") is True

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


class TestExternalAPIClient:
    def test_create_client(self):
        client = ExternalAPIClient("openai", model="gpt-4", api_key="test-key")
        assert client.provider == "openai"
        assert client.model == "gpt-4"
        assert client.api_key == "test-key"

    def test_create_client_default_model(self):
        client = ExternalAPIClient("openai", api_key="test-key")
        assert client.model == "gpt-4"  # First in the list

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
        client = ExternalAPIClient("ollama", model="llama3")
        captured = capsys.readouterr()
        assert "Warning" not in captured.out

    @patch('external_api.requests.post')
    def test_openai_chat(self, mock_post):
        """Test OpenAI API call with mocked response."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Hello from GPT!"}}]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = ExternalAPIClient("openai", model="gpt-4", api_key="test-key")
        response = client.chat("Hello")

        assert response == "Hello from GPT!"
        assert len(client.conversation_history) == 2
        assert client.conversation_history[0]["role"] == "user"
        assert client.conversation_history[1]["role"] == "assistant"
        mock_post.assert_called_once()

    @patch('external_api.requests.post')
    def test_anthropic_chat(self, mock_post):
        """Test Anthropic API call with mocked response."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "content": [{"text": "Hello from Claude!"}]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = ExternalAPIClient("anthropic", model="claude-3-haiku-20240307",
                                    api_key="test-key")
        response = client.chat("Hello")

        assert response == "Hello from Claude!"
        mock_post.assert_called_once()

    @patch('external_api.requests.post')
    def test_ollama_chat(self, mock_post):
        """Test Ollama API call with mocked response."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {"content": "Hello from Llama!"}
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = ExternalAPIClient("ollama", model="llama3")
        response = client.chat("Hello")

        assert response == "Hello from Llama!"
        mock_post.assert_called_once()

    @patch('external_api.requests.post')
    def test_chat_with_system_prompt(self, mock_post):
        """Test that system prompt is included in the request."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "response"}}]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = ExternalAPIClient("openai", model="gpt-4", api_key="test-key")
        client.chat("Hello", system_prompt="You are helpful")

        call_args = mock_post.call_args
        messages = call_args[1]["json"]["messages"] if "json" in call_args[1] else call_args[0]
        assert any(m["role"] == "system" for m in messages)

    def test_chat_api_error(self):
        """Test that API errors are caught gracefully."""
        client = ExternalAPIClient("openai", model="gpt-4", api_key="bad-key",
                                    base_url="http://localhost:1")
        response = client.chat("Hello")
        assert "[API Error" in response

    @patch('external_api.requests.post')
    def test_multi_turn_conversation(self, mock_post):
        """Test that conversation history builds up."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "response"}}]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = ExternalAPIClient("openai", model="gpt-4", api_key="test-key")
        client.chat("First message")
        client.chat("Second message")

        assert len(client.conversation_history) == 4  # 2 user + 2 assistant

    @patch('external_api.requests.post')
    def test_custom_provider_chat(self, mock_post):
        """Test custom provider API call."""
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
        """Test the convenience api_call function."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "Answer"}}]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result = api_call("openai", "gpt-4", "What is AI?", api_key="test-key")
        assert result == "Answer"
