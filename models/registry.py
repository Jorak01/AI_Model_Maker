"""Model Registry - Track, name, and access trained models by intent."""

import os
import json
import shutil
from datetime import datetime
from typing import Optional, Dict, List


REGISTRY_DIR = "trained_models"
REGISTRY_FILE = os.path.join(REGISTRY_DIR, "registry.json")


def _load_registry() -> Dict:
    """Load the registry from disk."""
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"models": {}}


def _save_registry(registry: Dict):
    """Save the registry to disk."""
    os.makedirs(REGISTRY_DIR, exist_ok=True)
    with open(REGISTRY_FILE, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=2)


def _sanitize_name(name: str) -> str:
    """Convert a name to a safe directory name."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name.lower().strip())
    return safe.strip("_")[:64]


def register_model(name: str, intent: str, base_model: str, pipeline: str,
                    checkpoint_dir: str, data_sources: Optional[List[str]] = None,
                    notes: str = "") -> str:
    """Register a trained model in the registry.

    Args:
        name: Human-readable name for the model (e.g. "customer-support-bot")
        intent: Description of what the model is trained for
        base_model: Which base model was used (custom, gpt2, etc.)
        pipeline: Training pipeline used (scratch, finetune, freeze)
        checkpoint_dir: Source checkpoint directory to copy from
        data_sources: List of training data files used
        notes: Additional notes about the model

    Returns:
        The model's registry directory path
    """
    registry = _load_registry()
    safe_name = _sanitize_name(name)

    if not safe_name:
        raise ValueError("Invalid model name")

    model_dir = os.path.join(REGISTRY_DIR, safe_name)
    os.makedirs(model_dir, exist_ok=True)

    # Copy best model and tokenizer
    best_src = os.path.join(checkpoint_dir, "best_model.pt")
    tok_src = os.path.join(checkpoint_dir, "tokenizer.pkl")

    if os.path.exists(best_src):
        shutil.copy2(best_src, os.path.join(model_dir, "model.pt"))
    else:
        # Find latest checkpoint
        ckpts = sorted([f for f in os.listdir(checkpoint_dir)
                        if f.startswith("model_epoch_") and f.endswith(".pt")])
        if ckpts:
            shutil.copy2(os.path.join(checkpoint_dir, ckpts[-1]),
                         os.path.join(model_dir, "model.pt"))
        else:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    if os.path.exists(tok_src):
        shutil.copy2(tok_src, os.path.join(model_dir, "tokenizer.pkl"))

    # Copy training data if present
    if data_sources:
        data_dir = os.path.join(model_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        for src in data_sources:
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(data_dir, os.path.basename(src)))

    # Register entry
    registry["models"][safe_name] = {
        "name": name,
        "safe_name": safe_name,
        "intent": intent,
        "base_model": base_model,
        "pipeline": pipeline,
        "created": datetime.now().isoformat(),
        "path": model_dir,
        "notes": notes,
        "data_sources": data_sources or [],
    }
    _save_registry(registry)

    print(f"\n  Model registered: '{name}'")
    print(f"  Directory: {model_dir}")
    print(f"  Intent: {intent}")
    return model_dir


def list_registered_models():
    """Print all registered models."""
    registry = _load_registry()
    models = registry.get("models", {})

    if not models:
        print("\n  No registered models yet.")
        print("  Use 'prompt-train' to create and register a custom model.")
        return

    print(f"\n  Registered Models ({len(models)}):")
    print("  " + "-" * 70)
    print(f"  {'Name':<25} {'Base':<14} {'Pipeline':<10} {'Intent'}")
    print("  " + "-" * 70)
    for key, info in models.items():
        print(f"  {info['name']:<25} {info['base_model']:<14} {info['pipeline']:<10} {info['intent'][:40]}")
    print()


def get_model_info(name: str) -> Optional[Dict]:
    """Get info about a registered model by name."""
    registry = _load_registry()
    safe = _sanitize_name(name)
    return registry["models"].get(safe)


def get_model_path(name: str) -> Optional[str]:
    """Get the model checkpoint path for a registered model."""
    info = get_model_info(name)
    if info:
        model_path = os.path.join(info["path"], "model.pt")
        if os.path.exists(model_path):
            return model_path
    return None


def get_tokenizer_path(name: str) -> Optional[str]:
    """Get the tokenizer path for a registered model."""
    info = get_model_info(name)
    if info:
        tok_path = os.path.join(info["path"], "tokenizer.pkl")
        if os.path.exists(tok_path):
            return tok_path
    return None


def delete_model(name: str) -> bool:
    """Delete a registered model."""
    registry = _load_registry()
    safe = _sanitize_name(name)
    if safe not in registry["models"]:
        print(f"  Model '{name}' not found.")
        return False

    info = registry["models"][safe]
    model_dir = info["path"]
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    del registry["models"][safe]
    _save_registry(registry)
    print(f"  Model '{name}' deleted.")
    return True


def load_registered_model(name: str, device: str = 'cpu'):
    """Load a registered model and tokenizer by name."""
    from models.model_factory import load_model
    from models.tokenizer import Tokenizer

    model_path = get_model_path(name)
    tok_path = get_tokenizer_path(name)

    if not model_path:
        print(f"  Error: Model '{name}' not found in registry.")
        return None, None

    model = load_model(model_path, device=device)
    tokenizer = Tokenizer.load(tok_path) if tok_path else None
    return model, tokenizer
