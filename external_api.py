"""External API Client - Call remote model APIs (OpenAI, Anthropic, Ollama, Google, DeepSeek, etc.)

Unified interface to interact with external AI model providers via their APIs.
Provider model lists are kept up-to-date through live API queries and a local cache.
"""

import os
import json
import yaml
import requests
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)

# ── Cache settings ──────────────────────────────────────────────────────
_API_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".model_cache")
_API_CACHE_FILE = os.path.join(_API_CACHE_DIR, "api_provider_models.json")
_API_CACHE_TTL_HOURS = 12

# ── Supported external providers ────────────────────────────────────────
# "models" is the static fallback; refresh_provider_models() replaces it live.

PROVIDERS: Dict[str, dict] = {
    "openai": {
        "name": "OpenAI (GPT-4o, o3, o4-mini)",
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "models": [
            "gpt-4o", "gpt-4o-mini",
            "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
            "o3", "o3-mini", "o4-mini",
            "gpt-4.5-preview",
            "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo",
        ],
    },
    "anthropic": {
        "name": "Anthropic (Claude 3.7, 3.5)",
        "base_url": "https://api.anthropic.com/v1",
        "env_key": "ANTHROPIC_API_KEY",
        "models": [
            "claude-sonnet-4-20250514",
            "claude-3.7-sonnet-20250219",
            "claude-3.5-sonnet-20241022",
            "claude-3.5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-haiku-20240307",
        ],
    },
    "google": {
        "name": "Google (Gemini 2.5, 2.0)",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "env_key": "GOOGLE_API_KEY",
        "models": [
            "gemini-2.5-pro", "gemini-2.5-flash",
            "gemini-2.0-flash", "gemini-2.0-flash-lite",
            "gemini-1.5-pro", "gemini-1.5-flash",
        ],
    },
    "deepseek": {
        "name": "DeepSeek (Chat, Reasoner)",
        "base_url": "https://api.deepseek.com/v1",
        "env_key": "DEEPSEEK_API_KEY",
        "models": ["deepseek-chat", "deepseek-reasoner"],
    },
    "ollama": {
        "name": "Ollama (Local)",
        "base_url": "http://localhost:11434",
        "env_key": None,
        "models": [
            "llama3.3", "llama3.2", "llama3.1",
            "mistral", "mistral-nemo",
            "phi4", "phi3",
            "gemma2", "gemma3",
            "qwen2.5", "qwen3",
            "deepseek-r1", "codellama", "command-r",
        ],
    },
    "custom": {
        "name": "Custom API Endpoint",
        "base_url": "",
        "env_key": "CUSTOM_API_KEY",
        "models": [],
    },
}


# ── Config / key helpers ────────────────────────────────────────────────

def load_config(path: str = 'config.yaml') -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def _get_api_key(provider: str) -> Optional[str]:
    info = PROVIDERS.get(provider, {})
    env_key = info.get("env_key")
    if env_key:
        key = os.environ.get(env_key)
        if key:
            return key
    try:
        config = load_config()
        key = config.get("external_api", {}).get(provider, {}).get("api_key", "")
        if key and key != "your-api-key-here":
            return key
    except Exception:
        pass
    return None


def _get_base_url(provider: str) -> str:
    try:
        config = load_config()
        url = config.get("external_api", {}).get(provider, {}).get("base_url", "")
        if url:
            return url
    except Exception:
        pass
    return PROVIDERS.get(provider, {}).get("base_url", "")


def is_provider_configured(provider: str) -> bool:
    if provider == "ollama":
        try:
            resp = requests.get(f"{_get_base_url('ollama')}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False
    if provider == "custom":
        return bool(_get_api_key("custom") and _get_base_url("custom"))
    return bool(_get_api_key(provider))


def find_available_provider(preferred_order: Optional[List[str]] = None) -> Optional[str]:
    for provider in (preferred_order or list(PROVIDERS.keys())):
        if provider not in PROVIDERS:
            continue
        if is_provider_configured(provider):
            return provider
        print(f"  Skipping {provider}: not configured")
    return None


def list_providers():
    print("\n  External API Providers:")
    print("  " + "-" * 65)
    for key, info in PROVIDERS.items():
        configured = is_provider_configured(key)
        status = ("running" if configured else "not running") if key == "ollama" \
            else ("configured" if configured else "no key set")
        print(f"  {key:<12} {info['name']:<40} [{status}] ({len(info['models'])} models)")
    print()
    print("  Set API keys via environment variables or config.yaml:")
    print("    OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY,")
    print("    DEEPSEEK_API_KEY, CUSTOM_API_KEY")
    print()
    print("  Tip: Use 'refresh-models' to query providers for their latest model lists")
    print()


# ── Live model-list refresh ────────────────────────────────────────────

def _load_api_cache() -> dict:
    if os.path.exists(_API_CACHE_FILE):
        try:
            with open(_API_CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _save_api_cache(cache: dict):
    os.makedirs(_API_CACHE_DIR, exist_ok=True)
    with open(_API_CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)


def _api_cache_is_stale() -> bool:
    last = _load_api_cache().get("last_updated")
    if not last:
        return True
    try:
        return datetime.now() - datetime.fromisoformat(last) > timedelta(hours=_API_CACHE_TTL_HOURS)
    except (ValueError, TypeError):
        return True


def _fetch_openai_models(api_key: str, base_url: str, timeout: int = 10) -> List[str]:
    """Query an OpenAI-compatible /v1/models endpoint."""
    try:
        resp = requests.get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            chat = sorted(
                [m["id"] for m in data
                 if any(k in m["id"] for k in ("gpt-", "o1", "o3", "o4", "chatgpt"))],
                reverse=True,
            )
            return chat if chat else [m["id"] for m in data]
    except Exception as e:
        logger.debug(f"OpenAI models query failed: {e}")
    return []


def _fetch_ollama_models(base_url: str, timeout: int = 5) -> List[str]:
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=timeout)
        if resp.status_code == 200:
            return [m["name"].split(":")[0] for m in resp.json().get("models", [])]
    except Exception as e:
        logger.debug(f"Ollama models query failed: {e}")
    return []


def _fetch_google_models(api_key: str, base_url: str, timeout: int = 10) -> List[str]:
    try:
        resp = requests.get(f"{base_url}/models?key={api_key}", timeout=timeout)
        if resp.status_code == 200:
            return [
                m["name"].replace("models/", "")
                for m in resp.json().get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])
            ]
    except Exception as e:
        logger.debug(f"Google models query failed: {e}")
    return []


def _fetch_deepseek_models(api_key: str, base_url: str, timeout: int = 10) -> List[str]:
    """Query DeepSeek /v1/models endpoint."""
    try:
        resp = requests.get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        if resp.status_code == 200:
            return [m["id"] for m in resp.json().get("data", [])]
    except Exception as e:
        logger.debug(f"DeepSeek models query failed: {e}")
    return []


def refresh_provider_models(force: bool = False, verbose: bool = True) -> Dict[str, List[str]]:
    """Query each configured provider for its current model list (concurrent).

    Results cached locally for _API_CACHE_TTL_HOURS.
    """
    if not force and not _api_cache_is_stale():
        cache = _load_api_cache()
        if verbose:
            print(f"\n  API model cache is up to date (last refreshed: {cache.get('last_updated', 'unknown')})")
        for prov, models in cache.get("providers", {}).items():
            if prov in PROVIDERS and models:
                PROVIDERS[prov]["models"] = models
        return cache.get("providers", {})

    if verbose:
        print("\n  Refreshing external API model lists...")
        print("  " + "-" * 55)

    # Build fetch tasks: (provider_name, callable) pairs
    tasks: List[tuple] = []

    openai_key = _get_api_key("openai")
    if openai_key:
        tasks.append(("openai", lambda k=openai_key: _fetch_openai_models(k, _get_base_url("openai"))))

    google_key = _get_api_key("google")
    if google_key:
        tasks.append(("google", lambda k=google_key: _fetch_google_models(k, _get_base_url("google"))))

    deepseek_key = _get_api_key("deepseek")
    if deepseek_key:
        tasks.append(("deepseek", lambda k=deepseek_key: _fetch_deepseek_models(k, _get_base_url("deepseek"))))

    if is_provider_configured("ollama"):
        tasks.append(("ollama", lambda: _fetch_ollama_models(_get_base_url("ollama"))))

    results: Dict[str, List[str]] = {}

    # Concurrent fetch
    if tasks:
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = {pool.submit(fn): name for name, fn in tasks}
            for future in as_completed(futures):
                prov = futures[future]
                try:
                    models = future.result()
                    if models:
                        results[prov] = models
                        PROVIDERS[prov]["models"] = models
                        if verbose:
                            print(f"  ✓ {prov}: {len(models)} models fetched")
                    elif verbose:
                        print(f"  ✗ {prov}: query failed, keeping defaults")
                except Exception as e:
                    if verbose:
                        print(f"  ✗ {prov}: {e}")

    # Log providers we skipped / couldn't query
    if verbose:
        for prov in ("openai", "google", "deepseek"):
            if not _get_api_key(prov) and prov not in results:
                print(f"  - {prov}: no API key set, keeping defaults")
        if not is_provider_configured("ollama") and "ollama" not in results:
            print(f"  - ollama: server not running, keeping defaults")
        print(f"  - anthropic: using curated model list ({len(PROVIDERS['anthropic']['models'])} models)")

    cache = {"last_updated": datetime.now().isoformat(), "providers": results}
    _save_api_cache(cache)

    if verbose:
        print("  " + "-" * 55)
        total = sum(len(v) for v in results.values())
        print(f"  Fetched {total} models across {len(results)} providers")
        print(f"  Cache saved → {_API_CACHE_FILE}\n")

    return results


# ── API Client ──────────────────────────────────────────────────────────

class ExternalAPIClient:
    """Unified client for calling external model APIs."""

    def __init__(self, provider: str, model: Optional[str] = None,
                 api_key: Optional[str] = None, base_url: Optional[str] = None):
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
        self.conversation_history.append({"role": "user", "content": message})
        try:
            dispatch = {
                "anthropic": self._call_anthropic,
                "google": self._call_google,
                "ollama": self._call_ollama,
            }
            handler = dispatch.get(self.provider, self._call_openai_compat)
            response = handler(system_prompt, temperature, max_tokens)
        except Exception as e:
            response = f"[API Error: {e}]"
        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def clear_history(self):
        self.conversation_history = []

    # ── Provider-specific call methods ──────────────────────────────────

    def _call_openai_compat(self, system_prompt: Optional[str],
                            temperature: float, max_tokens: int) -> str:
        """Call any OpenAI-compatible API (OpenAI, DeepSeek, Custom, etc.)."""
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
        return resp.json()["choices"][0]["message"]["content"]

    def _call_anthropic(self, system_prompt: Optional[str],
                        temperature: float, max_tokens: int) -> str:
        messages = [m for m in self.conversation_history if m["role"] in ("user", "assistant")]
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
        return resp.json()["content"][0]["text"]

    def _call_google(self, system_prompt: Optional[str],
                     temperature: float, max_tokens: int) -> str:
        contents: List[dict] = []
        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": f"[System instruction]: {system_prompt}"}]})
            contents.append({"role": "model", "parts": [{"text": "Understood. I will follow those instructions."}]})
        for msg in self.conversation_history:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        resp = requests.post(
            f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": contents,
                "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
            },
            timeout=60,
        )
        resp.raise_for_status()
        candidates = resp.json().get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            return parts[0]["text"] if parts else ""
        return ""

    def _call_ollama(self, system_prompt: Optional[str],
                     temperature: float, max_tokens: int) -> str:
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
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")


# ── Interactive session ─────────────────────────────────────────────────

def interactive_external_chat():
    print("\n" + "=" * 55)
    print("       External Model API Chat")
    print("=" * 55)
    list_providers()

    try:
        print("  Choose provider (openai/anthropic/google/deepseek/ollama/custom/auto):")
        provider = input("  > ").strip().lower()

        if not provider or provider == "auto":
            print("\n  Auto-detecting available providers...")
            provider = find_available_provider()
            if not provider:
                print("\n  No external APIs are configured.")
                print("  Set an API key in config.yaml or environment variables,")
                print("  or start Ollama locally.")
                print("  Returning to menu.\n")
                return
            print(f"  Found available provider: {provider}")
        elif provider not in PROVIDERS:
            print(f"  Unknown provider: {provider}")
            return
        elif not is_provider_configured(provider):
            print(f"\n  {provider} is not configured. Checking other providers...")
            fallback = find_available_provider([p for p in PROVIDERS if p != provider])
            if not fallback:
                print("\n  No external APIs are configured.")
                print("  Set an API key in config.yaml or environment variables,")
                print("  or start Ollama locally.")
                print("  Returning to menu.\n")
                return
            print(f"  Using fallback provider: {fallback}")
            provider = fallback

        models = PROVIDERS[provider]["models"]
        if models:
            print(f"\n  Available models for {provider}:")
            for i, m in enumerate(models, 1):
                print(f"    {i}. {m}")
            choice = input(f"  Choose model [1-{len(models)}, or type name]: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(models):
                model = models[int(choice) - 1]
            else:
                model = choice or models[0]
        else:
            model = input("  Enter model name: ").strip()

        url = input("  Enter API base URL: ").strip() if provider == "custom" else None
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
        print(f"AI: {client.chat(user_input, system_prompt=system_prompt)}\n")


# ── Convenience function ────────────────────────────────────────────────

def api_call(provider: str, model: str, message: str,
             system_prompt: Optional[str] = None,
             api_key: Optional[str] = None,
             base_url: Optional[str] = None,
             temperature: float = 0.7,
             max_tokens: int = 1024) -> str:
    """Programmatic single-call to an external API.

    Example:
        response = api_call("openai", "gpt-4o", "What is AI?")
        response = api_call("google", "gemini-2.5-flash", "Explain transformers")
    """
    client = ExternalAPIClient(provider, model=model, api_key=api_key, base_url=base_url)
    return client.chat(message, system_prompt=system_prompt,
                       temperature=temperature, max_tokens=max_tokens)


if __name__ == '__main__':
    interactive_external_chat()
