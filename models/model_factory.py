"""Model factory - create and load different baseline models.

Supports a wide range of pretrained HuggingFace models with auto-update
capability to always reflect the latest available model versions.
"""

import os
import json
import torch
import torch.nn as nn
import logging
import requests as _requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

# ── Cache for auto-updated model metadata ────────────────────────────────
MODEL_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".model_cache")
MODEL_CACHE_FILE = os.path.join(MODEL_CACHE_DIR, "model_versions.json")
CACHE_TTL_HOURS = 24

# ── Baseline models: (short_name → HuggingFace repo ID) ─────────────────
PRETRAINED_MODELS: Dict[str, str] = {
    # GPT-2
    "gpt2": "gpt2", "distilgpt2": "distilgpt2", "gpt2-medium": "gpt2-medium",
    "gpt2-large": "gpt2-large", "gpt2-xl": "gpt2-xl",
    # Pythia
    "pythia-160m": "EleutherAI/pythia-160m", "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1b": "EleutherAI/pythia-1b", "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "pythia-2.8b": "EleutherAI/pythia-2.8b",
    # Phi
    "phi-2": "microsoft/phi-2", "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "phi-3.5-mini": "microsoft/Phi-3.5-mini-instruct", "phi-4-mini": "microsoft/Phi-4-mini-instruct",
    # Llama
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B", "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
    # Mistral
    "mistral-7b": "mistralai/Mistral-7B-v0.3",
    # Gemma
    "gemma-2-2b": "google/gemma-2-2b",
    # Qwen
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B", "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B",
    # TinyLlama
    "tinyllama-1.1b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # SmolLM
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M", "smollm2-360m": "HuggingFaceTB/SmolLM2-360M",
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B",
    # StableLM / OLMo / BLOOM / Falcon / DeepSeek
    "stablelm-2-1.6b": "stabilityai/stablelm-2-1_6b", "olmo-1b": "allenai/OLMo-1B-hf",
    "bloom-560m": "bigscience/bloom-560m", "bloom-1.1b": "bigscience/bloom-1b1",
    "falcon-rw-1b": "tiiuae/falcon-rw-1B",
    "deepseek-r1-distill-qwen-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
}

# ── Model metadata: (short_name → params, desc, family) ─────────────────
# "custom" is special (not pretrained); all PRETRAINED_MODELS entries are pretrained.
_MODEL_META = {
    "custom":       ("~2M",  "Custom transformer (train from scratch)", "custom"),
    "gpt2":         ("124M", "OpenAI GPT-2 Small", "gpt2"),
    "distilgpt2":   ("82M",  "Distilled GPT-2 (smaller, faster)", "gpt2"),
    "gpt2-medium":  ("355M", "OpenAI GPT-2 Medium", "gpt2"),
    "gpt2-large":   ("774M", "OpenAI GPT-2 Large", "gpt2"),
    "gpt2-xl":      ("1.5B", "OpenAI GPT-2 XL", "gpt2"),
    "pythia-160m":  ("160M", "EleutherAI Pythia 160M", "pythia"),
    "pythia-410m":  ("410M", "EleutherAI Pythia 410M", "pythia"),
    "pythia-1b":    ("1B",   "EleutherAI Pythia 1B", "pythia"),
    "pythia-1.4b":  ("1.4B", "EleutherAI Pythia 1.4B", "pythia"),
    "pythia-2.8b":  ("2.8B", "EleutherAI Pythia 2.8B", "pythia"),
    "phi-2":        ("2.7B", "Microsoft Phi-2", "phi"),
    "phi-3-mini":   ("3.8B", "Microsoft Phi-3 Mini 4K Instruct", "phi"),
    "phi-3.5-mini": ("3.8B", "Microsoft Phi-3.5 Mini Instruct", "phi"),
    "phi-4-mini":   ("3.8B", "Microsoft Phi-4 Mini Instruct", "phi"),
    "llama-3.2-1b": ("1B",   "Meta Llama 3.2 1B", "llama"),
    "llama-3.2-3b": ("3B",   "Meta Llama 3.2 3B", "llama"),
    "mistral-7b":   ("7.2B", "Mistral 7B v0.3", "mistral"),
    "gemma-2-2b":   ("2.6B", "Google Gemma 2 2B", "gemma"),
    "qwen2.5-0.5b": ("0.5B", "Alibaba Qwen 2.5 0.5B", "qwen"),
    "qwen2.5-1.5b": ("1.5B", "Alibaba Qwen 2.5 1.5B", "qwen"),
    "qwen2.5-3b":   ("3B",   "Alibaba Qwen 2.5 3B", "qwen"),
    "tinyllama-1.1b":("1.1B","TinyLlama 1.1B Chat v1.0", "llama"),
    "smollm2-135m": ("135M", "HuggingFace SmolLM2 135M", "smollm"),
    "smollm2-360m": ("360M", "HuggingFace SmolLM2 360M", "smollm"),
    "smollm2-1.7b": ("1.7B", "HuggingFace SmolLM2 1.7B", "smollm"),
    "stablelm-2-1.6b":("1.6B","Stability AI StableLM 2 1.6B", "stablelm"),
    "olmo-1b":      ("1B",   "AI2 OLMo 1B", "olmo"),
    "bloom-560m":   ("560M", "BigScience BLOOM 560M (multilingual)", "bloom"),
    "bloom-1.1b":   ("1.1B", "BigScience BLOOM 1.1B", "bloom"),
    "falcon-rw-1b": ("1B",   "TII Falcon RW 1B", "falcon"),
    "deepseek-r1-distill-qwen-1.5b": ("1.5B", "DeepSeek R1 Distill Qwen 1.5B", "deepseek"),
}

# Build MODEL_INFO from the compact _MODEL_META table
MODEL_INFO: Dict[str, dict] = {
    name: {"params": p, "desc": d, "pretrained": name != "custom", "family": f}
    for name, (p, d, f) in _MODEL_META.items()
}

PIPELINE_INFO = {
    "scratch":  {"desc": "Train custom model from scratch (no pretrained weights)"},
    "finetune": {"desc": "Fine-tune ALL layers of a pretrained model"},
    "freeze":   {"desc": "Freeze base layers, only train top N layers"},
}

# ── Cache helpers ────────────────────────────────────────────────────────

def _load_model_cache() -> dict:
    if os.path.exists(MODEL_CACHE_FILE):
        try:
            with open(MODEL_CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _save_model_cache(cache: dict):
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    with open(MODEL_CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)


def _cache_is_stale() -> bool:
    cache = _load_model_cache()
    last = cache.get("last_updated")
    if not last:
        return True
    try:
        return datetime.now() - datetime.fromisoformat(last) > timedelta(hours=CACHE_TTL_HOURS)
    except (ValueError, TypeError):
        return True


# HuggingFace API URL is configurable via environment variable (see .env)
_HUGGINGFACE_API_URL = os.environ.get("HUGGINGFACE_API_URL", "https://huggingface.co/api")


def _fetch_hf_model_info(repo_id: str, timeout: int = 8) -> Optional[dict]:
    """Fetch model metadata from the HuggingFace Hub API."""
    try:
        resp = _requests.get(f"{_HUGGINGFACE_API_URL}/models/{repo_id}", timeout=timeout)
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


def refresh_models(force: bool = False, verbose: bool = True) -> dict:
    """Check HuggingFace Hub for latest info on all registered models (concurrent)."""
    if not force and not _cache_is_stale():
        cache = _load_model_cache()
        if verbose:
            print(f"\n  Model cache is up to date (last refreshed: {cache.get('last_updated', '?')})")
        return cache.get("models", {})

    if verbose:
        print("\n  Refreshing model registry from HuggingFace Hub...")
        print("  " + "-" * 55)

    results: Dict[str, dict] = {}
    unavailable: List[str] = []

    def _check(name_id):
        name, hf_id = name_id
        info = _fetch_hf_model_info(hf_id)
        return name, hf_id, info

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_check, item): item for item in PRETRAINED_MODELS.items()}
        for future in as_completed(futures):
            name, hf_id, info = future.result()
            if info:
                results[name] = {
                    "hf_id": hf_id, "available": True,
                    "last_modified": info.get("lastModified", ""),
                    "downloads": info.get("downloads", 0),
                }
                if verbose:
                    print(f"  ✓ {name:<35} {info.get('lastModified', '?')[:10]}  ({info.get('downloads', 0):>12,} dl)")
            else:
                unavailable.append(name)
                results[name] = {"hf_id": hf_id, "available": False}
                if verbose:
                    print(f"  ✗ {name:<35} (unavailable)")

    _save_model_cache({
        "last_updated": datetime.now().isoformat(),
        "total": len(PRETRAINED_MODELS), "available": len(PRETRAINED_MODELS) - len(unavailable),
        "unavailable": unavailable, "models": results,
    })

    if verbose:
        print("  " + "-" * 55)
        print(f"  {len(PRETRAINED_MODELS) - len(unavailable)}/{len(PRETRAINED_MODELS)} verified")
        if unavailable:
            print(f"  Unavailable: {', '.join(unavailable)}")
        print()

    return results


def get_model_latest_info(short_name: str) -> Optional[dict]:
    """Get cached metadata for a specific model."""
    cache = _load_model_cache()
    models = cache.get("models", {})
    if short_name in models and not _cache_is_stale():
        return models[short_name]
    hf_id = PRETRAINED_MODELS.get(short_name)
    return _fetch_hf_model_info(hf_id) if hf_id else None


def list_model_families() -> Dict[str, List[str]]:
    """Group models by family."""
    families: Dict[str, List[str]] = {}
    for name, info in MODEL_INFO.items():
        families.setdefault(info.get("family", "other"), []).append(name)
    return families


# ── Display ──────────────────────────────────────────────────────────────

def list_models(by_family: bool = False):
    """Print available baseline models."""
    if by_family:
        for family, names in list_model_families().items():
            print(f"\n  ── {family.upper()}")
            for n in names:
                i = MODEL_INFO[n]
                tag = "pretrained" if i["pretrained"] else "scratch"
                print(f"     {n:<35} {i['params']:>6}  [{tag}]  {i['desc']}")
    else:
        print("\n  Available Baseline Models:")
        print("  " + "-" * 80)
        for name, info in MODEL_INFO.items():
            tag = "[pretrained]" if info["pretrained"] else "[scratch]"
            print(f"  {name:<35} {info['params']:>6}  {tag:<14} {info['desc']}")
        print(f"\n  Total: {len(MODEL_INFO)} models  |  'refresh-models' to verify on HuggingFace Hub")
    print()


def list_pipelines():
    """Print available training pipelines."""
    print("\n  Training Pipelines:")
    for name, info in PIPELINE_INFO.items():
        print(f"    {name:<12} {info['desc']}")
    print(f"\n    scratch → custom only  |  finetune/freeze → any pretrained model\n")


# ── Pretrained wrapper ───────────────────────────────────────────────────

class PretrainedWrapper(nn.Module):
    """Unified wrapper around HuggingFace AutoModelForCausalLM models."""

    def __init__(self, model_name: str, max_seq_length: int = 128, pad_token_id: int = 0):
        super().__init__()
        from transformers import AutoModelForCausalLM

        self.model_name = model_name
        self.hf_name = PRETRAINED_MODELS[model_name]
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id

        print(f"Loading pretrained model: {self.hf_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(self.hf_name, trust_remote_code=True)
        cfg = self.model.config
        self.vocab_size = cfg.vocab_size
        self.embedding_dim = getattr(cfg, 'n_embd', None) or getattr(cfg, 'hidden_size', None) or 256

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids).logits

    def generate(self, input_ids: torch.Tensor, max_length: int = 100,
                 temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9,
                 repetition_penalty: float = 1.2, eos_token_id: Optional[int] = None) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self.model.generate(
                input_ids, max_new_tokens=max_length, temperature=temperature,
                top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                eos_token_id=eos_token_id, do_sample=True, pad_token_id=self.pad_token_id
            )

    def freeze_layers(self, num_layers: int = 4):
        """Freeze embeddings + bottom N transformer layers (multi-architecture)."""
        for p in self.model.get_input_embeddings().parameters():
            p.requires_grad = False

        # Find layer container across architectures
        layers = None
        m = self.model
        if hasattr(m, 'transformer'):
            layers = getattr(m.transformer, 'h', None) or getattr(m.transformer, 'layers', None)
        elif hasattr(m, 'gpt_neox'):
            layers = m.gpt_neox.layers
        elif hasattr(m, 'model'):
            layers = getattr(m.model, 'layers', None)
            if layers is None and hasattr(m.model, 'decoder'):
                layers = getattr(m.model.decoder, 'layers', None)

        if layers:
            for i, layer in enumerate(layers):
                if i < num_layers:
                    for p in layer.parameters():
                        p.requires_grad = False

        frozen = sum(1 for p in self.parameters() if not p.requires_grad)
        total = sum(1 for p in self.parameters())
        print(f"Frozen {frozen}/{total} parameter groups ({num_layers} layers)")

    def save(self, path: str):
        torch.save({
            'state_dict': self.model.state_dict(), 'model_name': self.model_name,
            'max_seq_length': self.max_seq_length, 'pad_token_id': self.pad_token_id,
            'type': 'pretrained'
        }, path)

    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'PretrainedWrapper':
        ckpt = torch.load(path, map_location=device)
        wrapper = cls(ckpt['model_name'], ckpt.get('max_seq_length', 128), ckpt.get('pad_token_id', 0))
        wrapper.model.load_state_dict(ckpt['state_dict'])
        return wrapper.to(device)


# ── Factory ──────────────────────────────────────────────────────────────

def create_model(config: dict) -> nn.Module:
    """Create a model based on config settings."""
    base = config['model']['base_model']
    pipeline = config['training']['pipeline']
    max_len = config['model']['max_seq_length']

    if base == "custom":
        if pipeline != "scratch":
            print(f"Warning: Custom model only supports 'scratch' pipeline. Switching from '{pipeline}'.")
        from models.model import ConversationalModel
        return ConversationalModel(
            vocab_size=config['model']['vocab_size'], embedding_dim=config['model']['embedding_dim'],
            hidden_dim=config['model']['hidden_dim'], num_layers=config['model']['num_layers'],
            num_heads=config['model']['num_heads'], max_seq_length=max_len,
            dropout=config['model']['dropout'], pad_token_id=0
        )

    if base not in PRETRAINED_MODELS:
        raise ValueError(f"Unknown model: {base}. Options: {list(MODEL_INFO.keys())}")

    if pipeline == "scratch":
        print(f"Warning: Pretrained model '{base}' should use 'finetune' or 'freeze'. Switching to 'finetune'.")

    wrapper = PretrainedWrapper(base, max_len)
    if pipeline == "freeze":
        wrapper.freeze_layers(config['training'].get('freeze_layers', 4))
    return wrapper


def load_model(path: str, device: str = 'cpu') -> nn.Module:
    """Load any model from checkpoint (auto-detects type)."""
    ckpt = torch.load(path, map_location=device)
    if ckpt.get('type') == 'pretrained':
        return PretrainedWrapper.load(path, device)
    from models.model import ConversationalModel
    return ConversationalModel.load(path, device)
