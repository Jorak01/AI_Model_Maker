"""External API Client - Call remote model APIs (OpenAI, Anthropic, Ollama, etc.)

This module is separate from the local runtime and provides a unified interface
to interact with external AI model providers via their APIs.
"""

import os
import json
import yaml
import requests
from typing import Optional, Dict, List, Any


# Supported external providers
PROVIDERS = {
    "openai": {
        "name": "OpenAI (GPT-4, GPT-3.5)",
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "models": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"],
    },
    "anthropic": {
        "name": "Anthropic (Claude)",
        "base_url": "https://api.anthropic.com/v1",
        "env_key": "ANTHROPIC_API_KEY",
        "models": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
    },
    "ollama": {
        "name": "Ollama (Local)",
        "base_url": "http://localhost:11434",
        "env_key": None,
        "models": ["llama3", "mistral", "codellama", "phi3", "gemma2"],
    },
    "custom": {
        "name": "Custom API Endpoint",
        "base_url": "",
        "env_key": "CUSTOM_API_KEY",
        "models": [],
    },
}


def load_config(path: str = 'config.yaml') -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def is_provider_configured(provider: str) -> bool:
    """Check if a provider is configured and ready to use.

    For API-key-based providers: checks if a key is set.
    For Ollama: checks if the base URL is reachable.
    """
    if provider == "ollama":
        # Ollama doesn't need a key, but check if server is reachable
        try:
            url = _get_base_url("ollama")
            resp = requests.get(f"{url}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    if provider == "custom":
        base_url = _get_base_url("custom")
        return bool(_get_api_key("custom") and base_url)

    return bool(_get_api_key(provider))


def find_available_provider(preferred_order: Optional[List[str]] = None) -> Optional[str]:
    """Find the first configured provider, skipping unconfigured ones.

    Args:
        preferred_order: Order of providers to try. Defaults to all providers.

    Returns:
        The first configured provider name, or None if none are available.
    """
    order = preferred_order or list(PROVIDERS.keys())
    for provider in order:
        if provider not in PROVIDERS:
            continue
        if is_provider_configured(provider):
            return provider
        print(f"  Skipping {provider}: not configured")
    return None


def list_providers():
    """Print available external API providers."""
    print("\n  External API Providers:")
    print("  " + "-" * 60)
    for key, info in PROVIDERS.items():
        configured = is_provider_configured(key)
        if key == "ollama":
            status = "running" if configured else "not running"
        else:
            status = "configured" if configured else "no key set"
        print(f"  {key:<12} {info['name']:<35} [{status}]")
    print()
    print("  Set API keys via environment variables or config.yaml:")
    print("    OPENAI_API_KEY, ANTHROPIC_API_KEY, CUSTOM_API_KEY")
    print()


def _get_api_key(provider: str) -> Optional[str]:
    """Get API key from environment or config."""
    info = PROVIDERS.get(provider, {})
    env_key = info.get("env_key")

    # Check environment first
    if env_key:
        key = os.environ.get(env_key)
        if key:
            return key

    # Check config.yaml
    try:
        config = load_config()
        ext = config.get("external_api", {})
        key = ext.get(provider, {}).get("api_key", "")
        if key and key != "your-api-key-here":
            return key
    except Exception:
        pass

    return None


def _get_base_url(provider: str) -> str:
    """Get the base URL for a provider."""
    try:
        config = load_config()
        ext = config.get("external_api", {})
        url = ext.get(provider, {}).get("base_url", "")
        if url:
            return url
    except Exception:
        pass
    return PROVIDERS.get(provider, {}).get("base_url", "")


class ExternalAPIClient:
    """Unified client for calling external model APIs."""

    def __init__(self, provider: str, model: Optional[str] = None,
                 api_key: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize the external API client.

        Args:
            provider: One of 'openai', 'anthropic', 'ollama', 'custom'
            model: Model name to use (e.g., 'gpt-4', 'claude-3-sonnet')
            api_key: API key (overrides env/config)
            base_url: Custom base URL (overrides default)
        """
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}. Options: {list(PROVIDERS.keys())}")

        self.provider = provider
        self.model = model or (PROVIDERS[provider]["models"][0] if PROVIDERS[provider]["models"] else "")
        self.api_key = api_key or _get_api_key(provider)
        self.base_url = base_url or _get_base_url(provider)
        self.conversation_history: List[Dict[str, str]] = []

        if provider not in ("ollama",) and not self.api_key:
            print(f"  Warning: No API key set for {provider}.")
            print(f"  Set {PROVIDERS[provider]['env_key']} or add it to config.yaml")

    def chat(self, message: str, system_prompt: Optional[str] = None,
             temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """Send a chat message and get a response.

        Args:
            message: User message
            system_prompt: System prompt (optional)
            temperature: Sampling temperature
            max_tokens: Max response tokens

        Returns:
            The model's response text
        """
        self.conversation_history.append({"role": "user", "content": message})

        try:
            if self.provider == "openai":
                response = self._call_openai(system_prompt, temperature, max_tokens)
            elif self.provider == "anthropic":
                response = self._call_anthropic(system_prompt, temperature, max_tokens)
            elif self.provider == "ollama":
                response = self._call_ollama(system_prompt, temperature, max_tokens)
            elif self.provider == "custom":
                response = self._call_custom(system_prompt, temperature, max_tokens)
            else:
                response = f"Provider '{self.provider}' not implemented."
        except Exception as e:
            response = f"[API Error: {e}]"

        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

    def _call_openai(self, system_prompt: Optional[str],
                     temperature: float, max_tokens: int) -> str:
        """Call OpenAI ChatCompletion API."""
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(self.conversation_history)

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _call_anthropic(self, system_prompt: Optional[str],
                        temperature: float, max_tokens: int) -> str:
        """Call Anthropic Messages API."""
        messages = [m for m in self.conversation_history
                    if m["role"] in ("user", "assistant")]

        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system_prompt:
            body["system"] = system_prompt

        resp = requests.post(
            f"{self.base_url}/messages",
            headers={
                "x-api-key": self.api_key or "",
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]

    def _call_ollama(self, system_prompt: Optional[str],
                     temperature: float, max_tokens: int) -> str:
        """Call Ollama local API."""
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(self.conversation_history)

        resp = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")

    def _call_custom(self, system_prompt: Optional[str],
                     temperature: float, max_tokens: int) -> str:
        """Call a custom OpenAI-compatible API endpoint."""
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(self.conversation_history)

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def interactive_external_chat():
    """Interactive chat session with an external API model.

    If a chosen provider is not configured, it skips and checks further providers.
    If no providers are configured at all, returns to the menu.
    """
    print("\n" + "=" * 55)
    print("       External Model API Chat")
    print("=" * 55)

    list_providers()

    try:
        print("  Choose provider (openai/anthropic/ollama/custom/auto):")
        provider = input("  > ").strip().lower()

        # Auto mode: find first available provider
        if provider == "auto" or provider == "":
            print("\n  Auto-detecting available providers...")
            provider = find_available_provider()
            if provider is None:
                print("\n  No external APIs are configured.")
                print("  Set an API key in config.yaml or environment variables,")
                print("  or start Ollama locally.")
                print("  Returning to menu.\n")
                return
            print(f"  Found available provider: {provider}")

        elif provider not in PROVIDERS:
            print(f"  Unknown provider: {provider}")
            return

        else:
            # Check if selected provider is configured; if not, try others
            if not is_provider_configured(provider):
                print(f"\n  {provider} is not configured. Checking other providers...")
                fallback = find_available_provider(
                    [p for p in PROVIDERS if p != provider]
                )
                if fallback is None:
                    print("\n  No external APIs are configured.")
                    print("  Set an API key in config.yaml or environment variables,")
                    print("  or start Ollama locally.")
                    print("  Returning to menu.\n")
                    return
                print(f"  Using fallback provider: {fallback}")
                provider = fallback

        # Model selection
        models = PROVIDERS[provider]["models"]
        if models:
            print(f"\n  Available models for {provider}:")
            for i, m in enumerate(models, 1):
                print(f"    {i}. {m}")
            choice = input(f"  Choose model [1-{len(models)}, or type name]: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(models):
                model = models[int(choice) - 1]
            elif choice:
                model = choice
            else:
                model = models[0]
        else:
            model = input("  Enter model name: ").strip()

        # Custom base URL
        if provider == "custom":
            url = input("  Enter API base URL: ").strip()
        else:
            url = None

        # System prompt
        print("\n  System prompt (optional, press Enter to skip):")
        system_prompt = input("  > ").strip() or None

    except (KeyboardInterrupt, EOFError):
        print("\n  Cancelled.")
        return

    client = ExternalAPIClient(provider, model=model, base_url=url)

    print(f"\n  Connected to {provider}/{model}")
    print("  Type 'quit' to exit, 'clear' to reset history\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Returning to menu...")
            break

        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit', 'stop'):
            print("  Returning to menu...")
            break
        if user_input.lower() == 'clear':
            client.clear_history()
            print("  [History cleared]\n")
            continue

        response = client.chat(user_input, system_prompt=system_prompt)
        print(f"AI: {response}\n")


def api_call(provider: str, model: str, message: str,
             system_prompt: Optional[str] = None,
             api_key: Optional[str] = None,
             base_url: Optional[str] = None,
             temperature: float = 0.7,
             max_tokens: int = 1024) -> str:
    """Programmatic single-call to an external API.

    Example:
        response = api_call("openai", "gpt-4", "What is AI?")
    """
    client = ExternalAPIClient(provider, model=model, api_key=api_key, base_url=base_url)
    return client.chat(message, system_prompt=system_prompt,
                       temperature=temperature, max_tokens=max_tokens)


if __name__ == '__main__':
    interactive_external_chat()
