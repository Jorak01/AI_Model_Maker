"""Model Loader - Load any model by number, support custom/untrained models, and train loaded models.

Provides a unified numbered listing of all available models:
  - Registered/custom models (already trained, from model registry)
  - Local checkpoints (from checkpoints/ directory)
  - Untrained baseline models (fresh from model factory)

Any loaded model can be chatted with or trained further.
"""

import os
import yaml
import torch
import torch.nn as nn
from datetime import datetime
from typing import Optional, Tuple, List, Dict

from models.model_factory import MODEL_INFO, PRETRAINED_MODELS, create_model, load_model as factory_load_model
from models.tokenizer import Tokenizer
from model_registry import _load_registry, get_model_path, get_tokenizer_path

# Back-navigation sentinel
_BACK_WORDS = frozenset(('back', 'quit', 'exit', 'q', '0'))


def _is_back(text: str) -> bool:
    """Return True if the user typed a 'go back' keyword."""
    return text.strip().lower() in _BACK_WORDS


# ── Metadata helpers ─────────────────────────────────────────────────────

def _format_size(size_bytes: int) -> str:
    """Format byte count to human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def _get_file_metadata(path: str) -> Dict:
    """Get file size and modification time for a path."""
    if path and os.path.exists(path):
        stat = os.stat(path)
        return {
            "file_size": stat.st_size,
            "file_size_str": _format_size(stat.st_size),
            "last_modified": datetime.fromtimestamp(stat.st_mtime),
            "last_modified_str": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
        }
    return {"file_size": None, "file_size_str": "—", "last_modified": None, "last_modified_str": "—"}


def _is_hf_model_installed(model_name: str) -> bool:
    """Check if a HuggingFace model is already downloaded/cached locally."""
    hf_id = PRETRAINED_MODELS.get(model_name)
    if not hf_id:
        return False

    # Check common HuggingFace cache locations
    cache_dirs = []

    # Default HF cache
    hf_home = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    hub_cache = os.path.join(hf_home, "hub")
    cache_dirs.append(hub_cache)

    # Legacy cache location
    legacy = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "transformers")
    cache_dirs.append(legacy)

    # TRANSFORMERS_CACHE env
    env_cache = os.environ.get("TRANSFORMERS_CACHE")
    if env_cache:
        cache_dirs.append(env_cache)

    # Look for the model directory in cache (HF uses models--org--name format)
    safe_id = hf_id.replace("/", "--")
    model_dir_name = f"models--{safe_id}"

    for cache_dir in cache_dirs:
        model_dir = os.path.join(cache_dir, model_dir_name)
        if os.path.isdir(model_dir):
            # Check for actual snapshot files (not just empty dirs)
            snapshots = os.path.join(model_dir, "snapshots")
            if os.path.isdir(snapshots) and os.listdir(snapshots):
                return True
    return False


# ── Catalog builders ─────────────────────────────────────────────────────

def _get_registered_models() -> List[Dict]:
    """Return list of registered (custom-trained) models with metadata."""
    registry = _load_registry()
    models = registry.get("models", {})
    entries = []
    for key, info in models.items():
        model_path = os.path.join(info["path"], "model.pt")
        tok_path = os.path.join(info["path"], "tokenizer.pkl")
        meta = _get_file_metadata(model_path)
        # Parse original creation date from registry
        created_str = info.get("created", "")
        try:
            created_dt = datetime.fromisoformat(created_str) if created_str else None
        except (ValueError, TypeError):
            created_dt = None
        entries.append({
            "source": "registered",
            "key": key,
            "name": info.get("name", key),
            "desc": info.get("intent", "Custom trained model"),
            "base_model": info.get("base_model", "unknown"),
            "pipeline": info.get("pipeline", "unknown"),
            "model_path": model_path if os.path.exists(model_path) else None,
            "tokenizer_path": tok_path if os.path.exists(tok_path) else None,
            "installed": os.path.exists(model_path),
            "file_size": meta["file_size"],
            "file_size_str": meta["file_size_str"],
            "last_trained": meta["last_modified"],
            "last_trained_str": meta["last_modified_str"],
            "created": created_dt,
            "created_str": created_dt.strftime("%Y-%m-%d %H:%M") if created_dt else "—",
        })
    return entries


def _get_checkpoint_models(config: Optional[dict] = None) -> List[Dict]:
    """Return list of local checkpoint models."""
    if config is None:
        try:
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
        except Exception:
            return []

    assert config is not None
    ckpt_dir: str = config.get('checkpoint', {}).get('save_dir', 'checkpoints')
    tok_path = os.path.join(ckpt_dir, 'tokenizer.pkl')
    entries = []

    if not os.path.exists(ckpt_dir):
        return entries

    # Best model
    best_path = os.path.join(ckpt_dir, 'best_model.pt')
    if os.path.exists(best_path):
        meta = _get_file_metadata(best_path)
        entries.append({
            "source": "checkpoint",
            "key": "best_model",
            "name": "Best Checkpoint",
            "desc": "Best trained model from checkpoints/",
            "base_model": config.get('model', {}).get('base_model', 'custom'),
            "pipeline": config.get('training', {}).get('pipeline', 'scratch'),
            "model_path": best_path,
            "tokenizer_path": tok_path if os.path.exists(tok_path) else None,
            "installed": True,
            "file_size": meta["file_size"],
            "file_size_str": meta["file_size_str"],
            "last_trained": meta["last_modified"],
            "last_trained_str": meta["last_modified_str"],
        })

    # Epoch checkpoints
    ckpts = sorted([f for f in os.listdir(ckpt_dir)
                    if f.startswith('model_epoch_') and f.endswith('.pt')])
    for ckpt in ckpts:
        full_path = os.path.join(ckpt_dir, ckpt)
        epoch = ckpt.replace('model_epoch_', '').replace('.pt', '')
        meta = _get_file_metadata(full_path)
        entries.append({
            "source": "checkpoint",
            "key": ckpt,
            "name": f"Epoch {epoch} Checkpoint",
            "desc": f"Checkpoint from training epoch {epoch}",
            "base_model": config.get('model', {}).get('base_model', 'custom'),
            "pipeline": config.get('training', {}).get('pipeline', 'scratch'),
            "model_path": full_path,
            "tokenizer_path": tok_path if os.path.exists(tok_path) else None,
            "installed": True,
            "file_size": meta["file_size"],
            "file_size_str": meta["file_size_str"],
            "last_trained": meta["last_modified"],
            "last_trained_str": meta["last_modified_str"],
        })

    return entries


def _get_untrained_models() -> List[Dict]:
    """Return list of untrained baseline models from model factory."""
    entries = []
    for name, info in MODEL_INFO.items():
        is_pretrained = info.get("pretrained", False)
        installed = _is_hf_model_installed(name) if is_pretrained else True  # scratch always "available"
        entries.append({
            "source": "untrained",
            "key": name,
            "name": name,
            "desc": info.get("desc", ""),
            "params": info.get("params", "?"),
            "family": info.get("family", "other"),
            "pretrained": is_pretrained,
            "base_model": name,
            "pipeline": "finetune" if is_pretrained else "scratch",
            "model_path": None,
            "tokenizer_path": None,
            "installed": installed,
            "file_size": None,
            "file_size_str": "—",
            "last_trained": None,
            "last_trained_str": "—",
        })
    return entries


# ── Unified catalog ──────────────────────────────────────────────────────

def build_model_catalog(config: Optional[dict] = None) -> List[Dict]:
    """Build a unified numbered catalog of all available models.

    Returns a list of model entries, each with:
        number, source, key, name, desc, base_model, pipeline,
        model_path, tokenizer_path, and (for untrained) params/family/pretrained.

    Ordering: registered → checkpoints → untrained baselines.
    """
    catalog = []

    # 1. Registered custom models
    catalog.extend(_get_registered_models())

    # 2. Local checkpoints
    catalog.extend(_get_checkpoint_models(config))

    # 3. Untrained baseline models
    catalog.extend(_get_untrained_models())

    # Assign numbers (1-indexed)
    for i, entry in enumerate(catalog):
        entry["number"] = i + 1

    return catalog


def display_catalog(catalog: Optional[List[Dict]] = None, config: Optional[dict] = None):
    """Print the numbered model catalog to the console.

    Args:
        catalog: Pre-built catalog, or None to build fresh.
        config: Config dict (used if catalog is None).

    Returns:
        The catalog list (for reuse).
    """
    if catalog is None:
        catalog = build_model_catalog(config)

    if not catalog:
        print("\n  No models available.")
        return catalog

    # Group by source
    registered = [e for e in catalog if e["source"] == "registered"]
    checkpoints = [e for e in catalog if e["source"] == "checkpoint"]
    untrained = [e for e in catalog if e["source"] == "untrained"]

    print("\n" + "=" * 95)
    print("       Model Catalog — Select by Number")
    print("=" * 95)

    if registered:
        print(f"\n  ── Registered/Custom Models ({len(registered)})")
        print("  " + "-" * 90)
        print(f"  {'#':<5} {'Name':<22} {'Base':<12} {'Installed':<10} "
              f"{'Last Trained':<18} {'Size':<10} {'Intent'}")
        print("  " + "-" * 90)
        for e in registered:
            inst = "✓ yes" if e.get("installed") else "✗ no"
            trained = e.get("last_trained_str", "—")
            size = e.get("file_size_str", "—")
            print(f"  {e['number']:<5} {e['name']:<22} {e['base_model']:<12} "
                  f"{inst:<10} {trained:<18} {size:<10} {e['desc'][:25]}")

    if checkpoints:
        print(f"\n  ── Local Checkpoints ({len(checkpoints)})")
        print("  " + "-" * 90)
        print(f"  {'#':<5} {'Name':<22} {'Base':<12} {'Installed':<10} "
              f"{'Last Trained':<18} {'Size':<10} {'Description'}")
        print("  " + "-" * 90)
        for e in checkpoints:
            inst = "✓ yes" if e.get("installed") else "✗ no"
            trained = e.get("last_trained_str", "—")
            size = e.get("file_size_str", "—")
            print(f"  {e['number']:<5} {e['name']:<22} {e['base_model']:<12} "
                  f"{inst:<10} {trained:<18} {size:<10} {e['desc'][:25]}")

    if untrained:
        print(f"\n  ── Untrained Baseline Models ({len(untrained)})")
        print("  " + "-" * 90)
        print(f"  {'#':<5} {'Name':<22} {'Params':>6}  {'Type':<12} "
              f"{'Cached':<10} {'Description'}")
        print("  " + "-" * 90)
        for e in untrained:
            tag = "pretrained" if e.get("pretrained") else "scratch"
            params = e.get("params", "?")
            cached = "✓ yes" if e.get("installed") else "✗ no"
            # For scratch models, show "n/a" instead of installed status
            if not e.get("pretrained"):
                cached = "n/a"
            print(f"  {e['number']:<5} {e['name']:<22} {params:>6}  [{tag}]"
                  f"{'':>4}{cached:<10} {e['desc'][:30]}")

    print(f"\n  Total: {len(catalog)} models available")
    print()

    return catalog


def get_entry_by_number(catalog: List[Dict], number: int) -> Optional[Dict]:
    """Look up a catalog entry by its assigned number."""
    for entry in catalog:
        if entry["number"] == number:
            return entry
    return None


def get_entry_by_name(catalog: List[Dict], name: str) -> Optional[Dict]:
    """Look up a catalog entry by name or key (case-insensitive)."""
    name_lower = name.lower().strip()
    for entry in catalog:
        if entry["key"].lower() == name_lower or entry["name"].lower() == name_lower:
            return entry
    return None


# ── Model loading ────────────────────────────────────────────────────────

def _load_config() -> dict:
    """Load config.yaml with sensible defaults."""
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except Exception:
        return {
            'model': {
                'base_model': 'custom', 'vocab_size': 10000, 'embedding_dim': 256,
                'hidden_dim': 512, 'num_layers': 4, 'num_heads': 8,
                'max_seq_length': 128, 'dropout': 0.1,
            },
            'training': {
                'pipeline': 'scratch', 'batch_size': 32, 'learning_rate': 0.0001,
                'num_epochs': 10, 'gradient_clip': 1.0, 'warmup_steps': 500,
                'weight_decay': 0.01, 'freeze_layers': 4,
            },
            'checkpoint': {'save_dir': 'checkpoints', 'save_every': 1, 'keep_last': 5},
            'generation': {
                'max_length': 100, 'temperature': 0.8,
                'top_k': 50, 'top_p': 0.9, 'repetition_penalty': 1.2,
            },
            'device': {'use_cuda': False, 'cuda_device': 0},
            'performance': {'mixed_precision': True, 'compile_model': True, 'num_workers': 0},
        }


def _get_device(config: dict) -> str:
    """Get the torch device string from config."""
    if config.get('device', {}).get('use_cuda', False) and torch.cuda.is_available():
        return f"cuda:{config['device'].get('cuda_device', 0)}"
    return 'cpu'


def load_model_by_number(number: int, catalog: Optional[List[Dict]] = None,
                         config: Optional[dict] = None,
                         device: Optional[str] = None
                         ) -> Tuple[Optional[nn.Module], Optional[Tokenizer], Optional[Dict]]:
    """Load a model by its catalog number.

    Args:
        number: The catalog number (1-indexed).
        catalog: Pre-built catalog (builds one if None).
        config: Config dict (loads from file if None).
        device: Torch device string (auto-detected if None).

    Returns:
        (model, tokenizer, entry) — tokenizer may be None for untrained pretrained models.
    """
    if config is None:
        config = _load_config()
    if device is None:
        device = _get_device(config)
    if catalog is None:
        catalog = build_model_catalog(config)

    entry = get_entry_by_number(catalog, number)
    if entry is None:
        print(f"  Error: No model with number {number}.")
        return None, None, None

    return _load_entry(entry, config, device)


def load_model_by_name(name: str, catalog: Optional[List[Dict]] = None,
                       config: Optional[dict] = None,
                       device: Optional[str] = None
                       ) -> Tuple[Optional[nn.Module], Optional[Tokenizer], Optional[Dict]]:
    """Load a model by name or key.

    Args:
        name: Model name or key (case-insensitive).
        catalog: Pre-built catalog (builds one if None).
        config: Config dict (loads from file if None).
        device: Torch device string (auto-detected if None).

    Returns:
        (model, tokenizer, entry) — tokenizer may be None for untrained pretrained models.
    """
    if config is None:
        config = _load_config()
    if device is None:
        device = _get_device(config)
    if catalog is None:
        catalog = build_model_catalog(config)

    entry = get_entry_by_name(catalog, name)
    if entry is None:
        print(f"  Error: No model named '{name}'.")
        return None, None, None

    return _load_entry(entry, config, device)


def _load_entry(entry: Dict, config: dict, device: str
                ) -> Tuple[Optional[nn.Module], Optional[Tokenizer], Optional[Dict]]:
    """Internal: load model + tokenizer from a catalog entry."""
    source = entry["source"]
    name = entry["name"]

    print(f"\n  Loading #{entry['number']}: {name}  [{source}]")
    print(f"  Base: {entry['base_model']}  |  Pipeline: {entry['pipeline']}")

    try:
        if source in ("registered", "checkpoint"):
            # Load existing trained model from checkpoint
            model_path = entry.get("model_path")
            tok_path = entry.get("tokenizer_path")

            if not model_path or not os.path.exists(model_path):
                print(f"  Error: Model file not found at {model_path}")
                return None, None, entry

            model = factory_load_model(model_path, device=device)
            tokenizer = Tokenizer.load(tok_path) if tok_path and os.path.exists(tok_path) else None

            print(f"  ✓ Model loaded from {model_path}")
            if tokenizer:
                print(f"  ✓ Tokenizer loaded ({len(tokenizer)} tokens)")
            else:
                print(f"  ⚠ No tokenizer found (build one before training)")

            return model, tokenizer, entry

        elif source == "untrained":
            # Create a fresh model from the factory
            base_model = entry["key"]
            pipeline = entry["pipeline"]

            # Build config for model creation
            model_config = dict(config)
            model_config['model'] = dict(config.get('model', {}))
            model_config['training'] = dict(config.get('training', {}))
            model_config['model']['base_model'] = base_model
            model_config['training']['pipeline'] = pipeline

            if base_model == "custom":
                # For custom untrained, we need a tokenizer first
                tok_path = os.path.join(
                    config.get('checkpoint', {}).get('save_dir', 'checkpoints'),
                    'tokenizer.pkl'
                )
                if os.path.exists(tok_path):
                    tokenizer = Tokenizer.load(tok_path)
                    model_config['model']['vocab_size'] = len(tokenizer)
                    print(f"  ✓ Existing tokenizer loaded ({len(tokenizer)} tokens)")
                else:
                    tokenizer = None
                    print(f"  ⚠ No tokenizer yet — one will be built when training starts")
            else:
                tokenizer = None  # Pretrained models have their own tokenizer

            print(f"  Creating fresh {base_model} model...")
            model = create_model(model_config)
            model = model.to(device)

            print(f"  ✓ Untrained model created ({base_model})")
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {total_params:,}")

            return model, tokenizer, entry

        else:
            print(f"  Error: Unknown source '{source}'")
            return None, None, entry

    except Exception as e:
        print(f"  Error loading model: {e}")
        return None, None, entry


# ── Training on loaded models ────────────────────────────────────────────

def train_loaded_model(model: nn.Module, tokenizer: Optional[Tokenizer],
                       entry: Dict, config: Optional[dict] = None,
                       device: Optional[str] = None,
                       train_data: Optional[str] = None,
                       test_data: Optional[str] = None,
                       epochs: Optional[int] = None,
                       checkpoint_dir: Optional[str] = None):
    """Train (or continue training) a loaded model.

    Works with any model source: registered, checkpoint, or untrained.

    Args:
        model: The loaded model (nn.Module).
        tokenizer: Tokenizer (may be None — will be built if needed).
        entry: Catalog entry dict for the model.
        config: Config dict (loads from file if None).
        device: Torch device string.
        train_data: Path to training data JSON. Defaults to config path.
        test_data: Path to test data JSON. Defaults to config path.
        epochs: Number of epochs. Defaults to config value.
        checkpoint_dir: Where to save checkpoints. Defaults to model-specific dir.
    """
    from utils.data_loader import create_data_loaders, load_and_prepare_data
    from utils.trainer import Trainer

    if config is None:
        config = _load_config()
    if device is None:
        device = _get_device(config)

    train_path = train_data or config.get('data', {}).get('train_path', 'data/train.json')
    test_path = test_data or config.get('data', {}).get('test_path', 'data/test.json')
    num_epochs = epochs or config.get('training', {}).get('num_epochs', 10)
    perf = config.get('performance', {})

    # Determine checkpoint directory
    if checkpoint_dir is None:
        source = entry.get("source", "unknown")
        key = entry.get("key", "model")
        safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
        if source == "registered":
            checkpoint_dir = os.path.join("checkpoints", f"retrain_{safe_key}")
        elif source == "untrained":
            checkpoint_dir = os.path.join("checkpoints", f"trained_{safe_key}")
        else:
            checkpoint_dir = str(config.get('checkpoint', {}).get('save_dir', 'checkpoints'))

    ckpt_dir: str = checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    tok_path = os.path.join(ckpt_dir, 'tokenizer.pkl')

    print(f"\n  ── Training Setup ──")
    print(f"  Model: #{entry.get('number', '?')} {entry.get('name', 'unknown')}")
    print(f"  Source: {entry.get('source', 'unknown')}")
    print(f"  Train data: {train_path}")
    print(f"  Test data: {test_path}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Checkpoints: {checkpoint_dir}")

    # Ensure we have training data
    if not os.path.exists(train_path):
        print(f"\n  Error: Training data not found at {train_path}")
        print("  Provide training data or use 'prompt-train' to create some.")
        return

    # Build or load tokenizer
    if tokenizer is None:
        print("\n  Building tokenizer from training data...")
        tokenizer = Tokenizer(
            config.get('model', {}).get('vocab_size', 10000), method='word'
        )
        texts = load_and_prepare_data(train_path)
        if os.path.exists(test_path):
            texts += load_and_prepare_data(test_path)
        tokenizer.build_vocab(texts)
        tokenizer.save(tok_path)
        print(f"  ✓ Tokenizer built ({len(tokenizer)} tokens), saved to {tok_path}")

        # For custom untrained models, we may need to resize the embedding
        if entry.get("source") == "untrained" and entry.get("key") == "custom":
            # Recreate with correct vocab size
            from models.model import ConversationalModel
            if isinstance(model, ConversationalModel) and model.vocab_size != len(tokenizer):
                print(f"  Resizing model vocab: {model.vocab_size} → {len(tokenizer)}")
                model_config = dict(config)
                model_config['model'] = dict(config.get('model', {}))
                model_config['training'] = dict(config.get('training', {}))
                model_config['model']['base_model'] = 'custom'
                model_config['training']['pipeline'] = 'scratch'
                model_config['model']['vocab_size'] = len(tokenizer)
                model = create_model(model_config)
                model = model.to(device)
    else:
        # Save tokenizer to the new checkpoint dir
        tokenizer.save(tok_path)

    # Create data loaders
    num_workers = perf.get('num_workers', 0)
    print(f"\n  Creating data loaders (workers={num_workers})...")
    train_loader, test_loader = create_data_loaders(
        train_path, test_path, tokenizer,
        config.get('training', {}).get('batch_size', 32),
        config.get('model', {}).get('max_seq_length', 128),
        num_workers=num_workers,
    )

    # Train
    print("  Starting training...\n")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        learning_rate=config.get('training', {}).get('learning_rate', 0.0001),
        device=device,
        checkpoint_dir=checkpoint_dir,
        gradient_clip=config.get('training', {}).get('gradient_clip', 1.0),
        warmup_steps=config.get('training', {}).get('warmup_steps', 500),
        weight_decay=config.get('training', {}).get('weight_decay', 0.01),
        pad_token_id=tokenizer.pad_token_id,
        use_amp=perf.get('mixed_precision', True),
        compile_model=perf.get('compile_model', True),
    )
    trainer.train(
        num_epochs=num_epochs,
        save_every=config.get('checkpoint', {}).get('save_every', 1),
        keep_last=config.get('checkpoint', {}).get('keep_last', 5),
    )

    print(f"\n  ✓ Training complete!")
    print(f"  Checkpoints saved to: {checkpoint_dir}")

    # Offer to register the model
    return checkpoint_dir


# ── Interactive loader (for run.py) ──────────────────────────────────────

def interactive_load_and_act():
    """Interactive flow: show catalog → pick by number → chat or train.

    This is the main entry point for the 'load' command in run.py.
    """
    config = _load_config()
    device = _get_device(config)

    # Build and display catalog
    catalog = build_model_catalog(config)
    catalog = display_catalog(catalog, config)

    if not catalog:
        print("  No models available. Train a model first or check your configuration.")
        return

    # Get selection
    try:
        print("  Type 'back' or '0' to return to the main menu.")
        selection = input("  Enter model number (or name): ").strip()
        if not selection or _is_back(selection):
            return
    except (KeyboardInterrupt, EOFError):
        print("\n  Cancelled.")
        return

    # Check for 'delete <number or name>'
    if selection.lower().startswith('delete '):
        target = selection[7:].strip()
        _interactive_delete_from_catalog(target, catalog)
        return

    # Load by number or name
    try:
        number = int(selection)
        model, tokenizer, entry = load_model_by_number(number, catalog, config, device)
    except ValueError:
        model, tokenizer, entry = load_model_by_name(selection, catalog, config, device)

    if model is None or entry is None:
        return

    # Choose action
    _post_load_menu(model, tokenizer, entry, config, device)


def _post_load_menu(model: nn.Module, tokenizer: Optional[Tokenizer],
                    entry: Dict, config: dict, device: str):
    """After loading a model, let the user choose: chat, train, or info."""

    while True:
        print(f"\n  ── Model Loaded: #{entry.get('number', '?')} {entry.get('name', '?')} ──")
        print("  " + "-" * 45)
        print("  1   chat       Chat with this model")
        print("  2   train      Train (or continue training) this model")
        print("  3   info       Show model details")
        print("  4   delete     Delete this model")
        print("  5   verify     Verify model integrity")
        print("  6   repair     Repair/reinstall broken model")
        print("  0   back       Return to main menu")
        print()

        try:
            choice = input("  action>> ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n  Returning to menu...")
            break

        if choice in ('0', 'back', 'quit', 'exit', 'q'):
            break

        if choice in ('1', 'chat'):
            if tokenizer is None:
                print("  ⚠ No tokenizer available for this model.")
                if entry.get("source") == "untrained" and entry.get("pretrained", False):
                    print("  Pretrained models require their HuggingFace tokenizer.")
                    print("  Train the model first to create a compatible tokenizer.")
                else:
                    print("  Train the model first to build a tokenizer.")
                continue
            from chat import interactive_chat
            interactive_chat(model, tokenizer, config, device)

        elif choice in ('2', 'train'):
            _interactive_train(model, tokenizer, entry, config, device)

        elif choice in ('3', 'info'):
            _show_model_info(model, entry)

        elif choice in ('4', 'delete'):
            deleted = delete_model_entry(entry)
            if deleted:
                print("  Returning to main menu...")
                break  # model is gone, exit post-load menu

        elif choice in ('5', 'verify'):
            verify_model_entry(entry, device)

        elif choice in ('6', 'repair'):
            repair_model_entry(entry, config, device)

        else:
            print(f"  Unknown option: '{choice}'")


def _interactive_train(model: nn.Module, tokenizer: Optional[Tokenizer],
                       entry: Dict, config: dict, device: str):
    """Interactive training sub-menu for a loaded model."""
    train_path = config.get('data', {}).get('train_path', 'data/train.json')
    test_path = config.get('data', {}).get('test_path', 'data/test.json')
    default_epochs = config.get('training', {}).get('num_epochs', 10)

    print(f"\n  ── Training Configuration ──")

    # Training data
    try:
        print(f"  (Type 'back' at any prompt to cancel)")
        print(f"  Training data path [{train_path}]:")
        custom_train = input("  > ").strip()
        if _is_back(custom_train):
            print("  Returning to model menu...")
            return
        if custom_train:
            train_path = custom_train

        print(f"  Test data path [{test_path}]:")
        custom_test = input("  > ").strip()
        if _is_back(custom_test):
            print("  Returning to model menu...")
            return
        if custom_test:
            test_path = custom_test

        print(f"  Number of epochs [{default_epochs}]:")
        ep_input = input("  > ").strip()
        if _is_back(ep_input):
            print("  Returning to model menu...")
            return
        epochs = int(ep_input) if ep_input else default_epochs

    except (KeyboardInterrupt, EOFError):
        print("\n  Training cancelled.")
        return
    except ValueError:
        print("  Invalid number, using default.")
        epochs = default_epochs

    # Confirm
    try:
        print(f"\n  Ready to train:")
        print(f"    Model:  #{entry.get('number', '?')} {entry.get('name', '?')}")
        print(f"    Data:   {train_path}")
        print(f"    Epochs: {epochs}")
        confirm = input("  Start training? [Y/n]: ").strip().lower()
        if confirm == 'n':
            print("  Training cancelled.")
            return
    except (KeyboardInterrupt, EOFError):
        print("\n  Training cancelled.")
        return

    try:
        ckpt_dir = train_loaded_model(
            model=model, tokenizer=tokenizer, entry=entry,
            config=config, device=device,
            train_data=train_path, test_data=test_path,
            epochs=epochs,
        )

        # After training, ask if they want to register the model
        if ckpt_dir and entry.get("source") in ("untrained", "checkpoint"):
            _offer_registration(entry, ckpt_dir, train_path)

    except KeyboardInterrupt:
        print("\n  Training interrupted.")
    except Exception as e:
        print(f"\n  Training error: {e}")


def _offer_registration(entry: Dict, checkpoint_dir: str, data_path: str):
    """Offer to register a newly trained model."""
    try:
        print(f"\n  Would you like to register this trained model?")
        register = input("  [y/N]: ").strip().lower()
        if register != 'y':
            return

        name = input("  Model name: ").strip()
        if not name:
            print("  Registration cancelled (no name provided).")
            return

        intent = input("  Intent/purpose: ").strip()
        if not intent:
            intent = f"Trained from {entry.get('name', 'unknown')}"

        from model_registry import register_model
        register_model(
            name=name,
            intent=intent,
            base_model=entry.get("base_model", "custom"),
            pipeline=entry.get("pipeline", "scratch"),
            checkpoint_dir=checkpoint_dir,
            data_sources=[data_path] if data_path else None,
            notes=f"Trained via model loader from {entry.get('source', 'unknown')} "
                  f"model '{entry.get('name', '?')}'",
        )
        print(f"  ✓ Model '{name}' registered!")

    except (KeyboardInterrupt, EOFError):
        print("\n  Registration skipped.")


def _show_model_info(model: nn.Module, entry: Dict):
    """Display detailed info about a loaded model."""
    print(f"\n  ── Model Details ──")
    print(f"  " + "-" * 55)
    print(f"  Number:       #{entry.get('number', '?')}")
    print(f"  Name:         {entry.get('name', '?')}")
    print(f"  Source:       {entry.get('source', '?')}")
    print(f"  Base Model:   {entry.get('base_model', '?')}")
    print(f"  Pipeline:     {entry.get('pipeline', '?')}")

    # Installed / availability status
    installed = entry.get("installed")
    if installed is not None:
        status = "✓ Yes" if installed else "✗ No"
        print(f"  Installed:    {status}")

    # File paths
    if entry.get("model_path"):
        print(f"  Model Path:   {entry['model_path']}")
    if entry.get("tokenizer_path"):
        print(f"  Tokenizer:    {entry['tokenizer_path']}")

    # Dates
    if entry.get("created_str") and entry["created_str"] != "—":
        print(f"  Created:      {entry['created_str']}")
    if entry.get("last_trained_str") and entry["last_trained_str"] != "—":
        print(f"  Last Trained: {entry['last_trained_str']}")

    # File size
    if entry.get("file_size_str") and entry["file_size_str"] != "—":
        print(f"  File Size:    {entry['file_size_str']}")

    # Description / metadata
    if entry.get("desc"):
        print(f"  Description:  {entry['desc']}")
    if entry.get("params"):
        print(f"  Est. Params:  {entry['params']}")
    if entry.get("family"):
        print(f"  Family:       {entry['family']}")

    # Actual parameter count
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters:   {total:,} total, {trainable:,} trainable")
    print(f"  Training:     {'enabled (all params)' if total == trainable else f'{trainable:,} trainable / {total - trainable:,} frozen'}")
    print()


# ── Model deletion ───────────────────────────────────────────────────────

def delete_model_entry(entry: Dict, confirm: bool = True) -> bool:
    """Delete an installed model (registered or checkpoint).

    Args:
        entry: Catalog entry dict for the model.
        confirm: If True, prompt for user confirmation before deleting.

    Returns:
        True if the model was deleted, False otherwise.
    """
    source = entry.get("source", "unknown")
    name = entry.get("name", "unknown")

    # Untrained baseline models can't be deleted (they're built-in)
    if source == "untrained":
        print(f"  ✗ Cannot delete untrained baseline model '{name}'.")
        print("    Built-in models are part of the model factory.")
        return False

    # Show what will be deleted
    print(f"\n  ── Delete Model ──")
    print(f"  Name:   {name}")
    print(f"  Source: {source}")
    if entry.get("model_path"):
        print(f"  File:   {entry['model_path']}")
    if entry.get("file_size_str") and entry["file_size_str"] != "—":
        print(f"  Size:   {entry['file_size_str']}")

    # Confirm deletion
    if confirm:
        try:
            answer = input(f"\n  Are you sure you want to delete '{name}'? [y/N]: ").strip().lower()
            if answer != 'y':
                print("  Deletion cancelled.")
                return False
        except (KeyboardInterrupt, EOFError):
            print("\n  Deletion cancelled.")
            return False

    if source == "registered":
        # Delete via model_registry
        from model_registry import delete_model
        key = entry.get("key", "")
        result = delete_model(key)
        if result:
            print(f"  ✓ Registered model '{name}' deleted (files + registry entry removed).")
        return result

    elif source == "checkpoint":
        # Delete the .pt checkpoint file
        model_path = entry.get("model_path")
        if model_path and os.path.exists(model_path):
            os.remove(model_path)
            print(f"  ✓ Checkpoint file deleted: {model_path}")
            return True
        else:
            print(f"  ✗ Checkpoint file not found: {model_path}")
            return False

    else:
        print(f"  ✗ Unknown source '{source}' — cannot delete.")
        return False


def verify_model_entry(entry: Dict, device: str = "cpu", verbose: bool = True) -> Dict:
    """Verify an installed model is intact and loadable.

    Checks:
      - File exists and size > 0
      - Checkpoint can be loaded by torch
      - Model parameters can be enumerated
      - Tokenizer (if present) loads correctly

    Args:
        entry: Catalog entry dict for the model.
        device: Torch device for loading test.
        verbose: Print progress to console.

    Returns:
        Dict with keys: ok (bool), checks (list of dicts), errors (list of str)
    """
    checks: List[Dict] = []
    errors: List[str] = []
    source = entry.get("source", "unknown")
    name = entry.get("name", "unknown")

    if verbose:
        print(f"\n  ── Verifying: {name} [{source}] ──")

    # 1. Untrained models: just check availability
    if source == "untrained":
        is_pretrained = entry.get("pretrained", False)
        if is_pretrained:
            installed = _is_hf_model_installed(entry.get("key", ""))
            checks.append({"check": "HF cache", "ok": installed,
                           "detail": "Cached locally" if installed else "Not cached"})
            if not installed:
                errors.append(f"Pretrained model '{name}' is not downloaded/cached.")
        else:
            checks.append({"check": "Built-in", "ok": True, "detail": "Scratch model always available"})
        if verbose:
            for c in checks:
                mark = "✓" if c["ok"] else "✗"
                print(f"  {mark} {c['check']}: {c['detail']}")
        return {"ok": len(errors) == 0, "checks": checks, "errors": errors}

    # 2. Check model file exists
    model_path = entry.get("model_path")
    if model_path and os.path.exists(model_path):
        size = os.path.getsize(model_path)
        ok = size > 0
        checks.append({"check": "File exists", "ok": ok,
                        "detail": f"{model_path} ({_format_size(size)})" if ok else "File is empty (0 bytes)"})
        if not ok:
            errors.append(f"Model file exists but is empty: {model_path}")
    else:
        checks.append({"check": "File exists", "ok": False, "detail": f"Not found: {model_path}"})
        errors.append(f"Model file not found: {model_path}")
        if verbose:
            for c in checks:
                mark = "✓" if c["ok"] else "✗"
                print(f"  {mark} {c['check']}: {c['detail']}")
        return {"ok": False, "checks": checks, "errors": errors}

    # 3. Try loading the checkpoint
    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        checks.append({"check": "Loadable", "ok": True, "detail": "Checkpoint loads without error"})

        # Check for expected keys
        if isinstance(ckpt, dict):
            has_state = "state_dict" in ckpt or "model_state_dict" in ckpt
            checks.append({"check": "State dict", "ok": has_state,
                           "detail": f"Keys: {list(ckpt.keys())[:5]}"})
            if not has_state:
                errors.append("Checkpoint missing 'state_dict' key")
    except Exception as e:
        checks.append({"check": "Loadable", "ok": False, "detail": f"Error: {e}"})
        errors.append(f"Checkpoint corrupt or incompatible: {e}")
        if verbose:
            for c in checks:
                mark = "✓" if c["ok"] else "✗"
                print(f"  {mark} {c['check']}: {c['detail']}")
        return {"ok": False, "checks": checks, "errors": errors}

    # 4. Try full model load
    try:
        model = factory_load_model(model_path, device=device)
        total_params = sum(p.numel() for p in model.parameters())
        checks.append({"check": "Model loads", "ok": True,
                        "detail": f"{total_params:,} parameters"})
    except Exception as e:
        checks.append({"check": "Model loads", "ok": False, "detail": f"Error: {e}"})
        errors.append(f"Model failed to instantiate: {e}")

    # 5. Check tokenizer
    tok_path = entry.get("tokenizer_path")
    if tok_path:
        if os.path.exists(tok_path):
            try:
                tok = Tokenizer.load(tok_path)
                checks.append({"check": "Tokenizer", "ok": True,
                                "detail": f"Loaded ({len(tok)} tokens)"})
            except Exception as e:
                checks.append({"check": "Tokenizer", "ok": False, "detail": f"Error: {e}"})
                errors.append(f"Tokenizer corrupt: {e}")
        else:
            checks.append({"check": "Tokenizer", "ok": False, "detail": "File not found"})
            errors.append(f"Tokenizer not found: {tok_path}")

    if verbose:
        for c in checks:
            mark = "✓" if c["ok"] else "✗"
            print(f"  {mark} {c['check']}: {c['detail']}")
        status = "PASS" if len(errors) == 0 else f"FAIL ({len(errors)} issue(s))"
        print(f"\n  Result: {status}")

    return {"ok": len(errors) == 0, "checks": checks, "errors": errors}


def repair_model_entry(entry: Dict, config: Optional[dict] = None,
                       device: str = "cpu") -> bool:
    """Attempt to repair/reinstall a broken model.

    For registered models: re-copies from original checkpoint dir if available.
    For checkpoints: offers to retrain from data.
    For untrained pretrained: re-downloads from HuggingFace.
    For untrained scratch: always available, nothing to repair.

    Args:
        entry: Catalog entry dict for the model.
        config: Config dict.
        device: Torch device.

    Returns:
        True if repair succeeded, False otherwise.
    """
    source = entry.get("source", "unknown")
    name = entry.get("name", "unknown")

    print(f"\n  ── Repair/Reinstall: {name} [{source}] ──")

    if source == "untrained":
        if not entry.get("pretrained", False):
            print("  Scratch models don't need repair (they're generated on the fly).")
            return True

        # Re-download pretrained model from HuggingFace
        model_key = entry.get("key", "")
        hf_id = PRETRAINED_MODELS.get(model_key)
        if not hf_id:
            print(f"  ✗ Unknown pretrained model key: {model_key}")
            return False

        print(f"  Downloading {hf_id} from HuggingFace Hub...")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            AutoModelForCausalLM.from_pretrained(hf_id, trust_remote_code=True)
            AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
            print(f"  ✓ Model {hf_id} downloaded and cached successfully.")
            return True
        except Exception as e:
            print(f"  ✗ Download failed: {e}")
            return False

    elif source == "checkpoint":
        model_path = entry.get("model_path")
        print(f"  Checkpoint: {model_path}")
        print("  Checkpoints are created during training.")
        print("  To repair, retrain the model using 'train' (option 1).")
        print("  Or delete this checkpoint and retrain from scratch.")
        return False

    elif source == "registered":
        # Try to verify and rebuild from registry info
        print("  Registered models are copies of training checkpoints.")
        print("  To repair, retrain the model and re-register it.")
        print("  Use 'prompt-train' or 'load → train' to create a new version.")
        return False

    else:
        print(f"  ✗ Unknown source '{source}' — cannot repair.")
        return False


def _interactive_delete_from_catalog(target: str, catalog: List[Dict]):
    """Handle 'delete <number or name>' from the catalog view."""
    # Try to find the entry by number or name
    entry = None
    try:
        number = int(target)
        entry = get_entry_by_number(catalog, number)
    except ValueError:
        entry = get_entry_by_name(catalog, target)

    if entry is None:
        print(f"  Error: No model matching '{target}' found in catalog.")
        return

    delete_model_entry(entry, confirm=True)
