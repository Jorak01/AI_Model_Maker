"""
AI Model Runtime - Interactive entry point.

Run with: python run.py
All functions are accessible from the interactive menu.
Type 'stop' or 'quit' at any time to exit.
"""

import sys
import os
import signal
import subprocess
import importlib
import re

# Global flag for graceful shutdown
_running = True

# Path to .env file
_ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')

# Global active model state — set by 'load' command, used by 'chat'
_active_model = None
_active_tokenizer = None
_active_device = None
_active_model_name = None


def signal_handler(sig, frame):
    global _running
    _running = False
    print("\n\nShutting down...")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


# =========================================================================
# Dependency checker — runs once at launch
# =========================================================================

# Map of pip package names to their Python import names (where they differ)
_IMPORT_NAME_MAP = {
    'pyyaml': 'yaml',
    'pillow': 'PIL',
    'flask-cors': 'flask_cors',
    'beautifulsoup4': 'bs4',
    'scikit-learn': 'sklearn',
}


def _parse_requirements(filepath='requirements.txt'):
    """Parse requirements.txt and return list of (pip_name, import_name) tuples.

    Skips comments, blank lines, and commented-out optional packages.
    """
    packages = []
    req_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath)
    if not os.path.exists(req_path):
        return packages

    with open(req_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines, comments, and commented-out optionals
            if not line or line.startswith('#'):
                continue
            # Extract package name (strip version specifiers like >=, ==, ~=, etc.)
            pip_name = re.split(r'[><=!~;]', line)[0].strip()
            if not pip_name:
                continue
            # Determine the import name
            import_name = _IMPORT_NAME_MAP.get(pip_name.lower(), pip_name.lower().replace('-', '_'))
            packages.append((pip_name, import_name))
    return packages


def check_and_install_dependencies():
    """Check all required packages from requirements.txt and install any that are missing."""
    print("\n  Checking dependencies...")

    packages = _parse_requirements()
    if not packages:
        print("  No requirements.txt found or it is empty — skipping.")
        return

    missing = []
    for pip_name, import_name in packages:
        try:
            importlib.import_module(str(import_name))
        except ImportError:
            missing.append(pip_name)

    if not missing:
        print(f"  All {len(packages)} required packages are installed. ✓")
        return

    print(f"\n  Missing {len(missing)} package(s): {', '.join(missing)}")
    print("  Installing missing packages...\n")

    req_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'requirements.txt')
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-r', req_path],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"  Dependencies installed successfully. ✓")
        else:
            print(f"  pip install finished with warnings/errors (exit code {result.returncode}):")
            # Show only meaningful error lines
            for line in result.stderr.splitlines():
                if line.strip():
                    print(f"    {line.strip()}")

            # Verify which packages are still missing after the attempt
            still_missing = []
            for pip_name in missing:
                import_name = _IMPORT_NAME_MAP.get(pip_name.lower()) or pip_name.lower().replace('-', '_')
                try:
                    importlib.import_module(import_name)
                except ImportError:
                    still_missing.append(pip_name)

            if still_missing:
                print(f"\n  ⚠  Still missing after install: {', '.join(still_missing)}")
                print("  Some features may not work. Try installing manually:")
                print(f"    {sys.executable} -m pip install {' '.join(still_missing)}")
            else:
                print(f"  All packages resolved after install. ✓")
    except Exception as e:
        print(f"  Error running pip: {e}")
        print(f"  Install manually:  {sys.executable} -m pip install -r requirements.txt")


# =========================================================================
# .env loader — reads .env file into os.environ at startup
# =========================================================================

def _load_env_file(filepath: str = _ENV_FILE):
    """Parse a .env file and load variables into os.environ.

    Only sets variables that are NOT already set in the environment,
    so real env vars always take priority.
    """
    if not os.path.exists(filepath):
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, _, value = line.partition('=')
            key = key.strip()
            value = value.strip()
            # Strip surrounding quotes if present
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            # Only set if not already present in environment
            if key not in os.environ:
                os.environ[key] = value


def _update_env_file(updates: dict, filepath: str = _ENV_FILE):
    """Update specific key=value pairs in the .env file.

    Preserves comments, ordering, and unrelated lines.
    If a key doesn't exist in the file, it is appended.
    """
    lines = []
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

    keys_written = set()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('#') and '=' in stripped:
            key = stripped.split('=', 1)[0].strip()
            if key in updates:
                new_lines.append(f"{key}={updates[key]}\n")
                keys_written.add(key)
                continue
        new_lines.append(line)

    # Append any keys not already in the file
    for key, value in updates.items():
        if key not in keys_written:
            new_lines.append(f"{key}={value}\n")

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)


# =========================================================================
# API Token Setup — interactive first-run configuration
# =========================================================================

# All supported API tokens with metadata
_API_TOKENS = [
    {
        'env_var': 'HF_TOKEN',
        'name': 'HuggingFace',
        'required_for': 'Downloading gated models (Llama, Gemma, Mistral, etc.) and pushing to Hub',
        'signup_url': 'https://huggingface.co/join',
        'token_url': 'https://huggingface.co/settings/tokens',
        'category': 'model_access',
    },
    {
        'env_var': 'OPENAI_API_KEY',
        'name': 'OpenAI',
        'required_for': 'GPT-4o, o3, o4-mini, GPT-4.1 via external chat/API',
        'signup_url': 'https://platform.openai.com/signup',
        'token_url': 'https://platform.openai.com/api-keys',
        'category': 'external_api',
    },
    {
        'env_var': 'ANTHROPIC_API_KEY',
        'name': 'Anthropic',
        'required_for': 'Claude 3.7/3.5 Sonnet, Opus via external chat/API',
        'signup_url': 'https://console.anthropic.com/',
        'token_url': 'https://console.anthropic.com/settings/keys',
        'category': 'external_api',
    },
    {
        'env_var': 'GOOGLE_API_KEY',
        'name': 'Google AI (Gemini)',
        'required_for': 'Gemini 2.5 Pro/Flash via external chat/API',
        'signup_url': 'https://ai.google.dev/',
        'token_url': 'https://aistudio.google.com/apikey',
        'category': 'external_api',
    },
    {
        'env_var': 'DEEPSEEK_API_KEY',
        'name': 'DeepSeek',
        'required_for': 'DeepSeek Chat/Reasoner via external chat/API',
        'signup_url': 'https://platform.deepseek.com/',
        'token_url': 'https://platform.deepseek.com/api_keys',
        'category': 'external_api',
    },
]


def _mask_key(key: str) -> str:
    """Show first 4 and last 4 chars of a key, mask the rest."""
    if len(key) <= 10:
        return key[:2] + '*' * (len(key) - 2)
    return key[:4] + '*' * (len(key) - 8) + key[-4:]


def check_and_setup_api_tokens():
    """Check API token status and offer interactive setup if tokens are missing.

    Runs at startup after .env is loaded. Shows status of all tokens
    with links to obtain them, and optionally lets the user enter keys.
    """
    print("\n  ─── API Token Status ─────────────────────────────")
    print()

    # Categorize tokens
    configured = []
    missing = []

    for token_info in _API_TOKENS:
        env_var = token_info['env_var']
        value = os.environ.get(env_var, '').strip()
        if value:
            configured.append(token_info)
        else:
            missing.append(token_info)

    # Show configured tokens
    if configured:
        for t in configured:
            key_val = os.environ.get(t['env_var'], '')
            print(f"  ✓ {t['name']:<20} {_mask_key(key_val)}")

    # Show missing tokens
    if missing:
        if configured:
            print()
        for t in missing:
            print(f"  ✗ {t['name']:<20} not set")
            print(f"    {'':<20} Used for: {t['required_for']}")
            print(f"    {'':<20} Get key:  {t['token_url']}")
    else:
        print(f"\n  All API tokens are configured. ✓")
        print()
        return

    print()

    # Only offer interactive setup if there are missing tokens
    # and we're in an interactive terminal
    if not sys.stdin.isatty():
        print("  Set missing tokens in .env or as environment variables.\n")
        return

    try:
        setup = input("  Would you like to set up API tokens now? [y/N/skip]: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print("\n  Skipping token setup.\n")
        return

    if setup not in ('y', 'yes'):
        print("  Skipping — you can set tokens later in .env or re-run setup.\n")
        return

    print()
    print("  ─── API Token Setup ──────────────────────────────")
    print("  Paste each token when prompted. Press Enter to skip.")
    print("  Tokens are saved to .env (git-ignored, never committed).")
    print()

    env_updates = {}

    for t in missing:
        print(f"  {t['name']}:")
        print(f"    Sign up:  {t['signup_url']}")
        print(f"    Get key:  {t['token_url']}")
        try:
            value = input(f"    {t['env_var']}: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Token setup cancelled.\n")
            break

        if value:
            os.environ[t['env_var']] = value
            env_updates[t['env_var']] = value
            print(f"    ✓ Set {t['env_var']}")
        else:
            print(f"    – Skipped")
        print()

    # Save entered tokens to .env
    if env_updates:
        try:
            _update_env_file(env_updates)
            print(f"  Saved {len(env_updates)} token(s) to .env ✓")
        except Exception as e:
            print(f"  Warning: Could not save to .env: {e}")
            print("  Tokens are set for this session but won't persist.")
    print()


def print_banner():
    print("\n" + "=" * 55)
    print("       AI Model Runtime")
    print("=" * 55)


def print_menu():
    print()
    print("  ─── Training & Data ─────────────────────────")
    print("   1  train            Train the model")
    print("   2  prompt-train     Create model from prompts")
    print("   3  auto-train       Auto-train from public data")
    print("   4  auto-image       Auto-collect image tags")
    print("   5  dataset-mgr      Dataset management hub")
    print("   6  curriculum       Curriculum learning")
    print()
    print("  ─── Chat & Generation ────────────────────────")
    print("   7  chat             Interactive chat (local)")
    print("   8  image-gen        Image generation & training")
    print("   9  external         Chat via external API")
    print("  10  rag              RAG document chat")
    print("  11  agent            Agent with tool use")
    print()
    print("  ─── Model Management ────────────────────────")
    print("  12  load             Load any model by number")
    print("  13  explore          Browse all models")
    print("  14  registry         List registered models")
    print("  15  load-model       Chat with registered model")
    print("  16  packager         Export/import model archives")
    print()
    print("  ─── Evaluation & Monitoring ─────────────────")
    print("  17  eval             Evaluation suite (BLEU, etc)")
    print("  18  dashboard        Training dashboard (web)")
    print()
    print("  ─── Image Tools ────────────────────────────")
    print("  19  tag-mgr          Tag frequency & ontology")
    print("  20  image-tools      Upscale, variations, color")
    print()
    print("  ─── API & Servers ──────────────────────────")
    print("  21  api              Start REST API server")
    print("  22  compat-api       OpenAI-compatible API")
    print("  23  web-ui           Browser-based interface")
    print("  24  providers        List API providers")
    print()
    print("  ─── Infrastructure ─────────────────────────")
    print("  25  plugins          Plugin manager")
    print("  26  config-mgr       Config profiles & validation")
    print()
    print("  ─── Info & System ───────────────────────────")
    print("  27  models           Available base models")
    print("  28  pipelines        Training pipelines")
    print("  29  model-families   Models grouped by family")
    print("  30  config           Show configuration")
    print("  31  status           Model/checkpoint status")
    print("  32  refresh-models   Update model lists")
    print("  33  tutorial         Interactive tutorial")
    print("  34  test             Run all tests")
    print()
    print("  ─── Local Model Tools ──────────────────────")
    print("  35  local-models     Manage locally stored models")
    print("  36  verify-tokens    Verify tokenizer integrity")
    print()
    print("   0  stop / quit      Exit")
    print()


# =========================================================================
# Command handlers
# =========================================================================

def cmd_train():
    """Run training."""
    from training.train import main as train_main
    try:
        train_main()
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    except Exception as e:
        print(f"\nTraining error: {e}")


def cmd_chat():
    """Run interactive chat — requires a model to be loaded first via 'load' (option 12)."""
    if _active_model is None or _active_tokenizer is None:
        print("\n  No model loaded.")
        print("  Use 'load' (option 12) to load a model first, then come back to chat.")
        return

    try:
        import yaml
        try:
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
        except Exception:
            config = {'generation': {'max_length': 100, 'temperature': 0.8,
                                     'top_k': 50, 'top_p': 0.9, 'repetition_penalty': 1.2},
                      'device': {'use_cuda': False, 'cuda_device': 0}}

        from services.chat import interactive_chat
        interactive_chat(_active_model, _active_tokenizer, config,
                         _active_device or 'cpu', model_name=_active_model_name)
    except KeyboardInterrupt:
        print("\nChat ended.")
    except Exception as e:
        print(f"\nChat error: {e}")


def cmd_api():
    """Start the API server."""
    from services.api import main as api_main
    print("\nStarting API server... Press Ctrl+C to stop and return to menu.")
    try:
        api_main()
    except KeyboardInterrupt:
        print("\nAPI server stopped.")
    except Exception as e:
        print(f"\nAPI error: {e}")


def cmd_models():
    """List available baseline models."""
    from models.model_factory import list_models
    list_models()


def cmd_pipelines():
    """List available training pipelines."""
    from models.model_factory import list_pipelines
    list_pipelines()


def cmd_test():
    """Run all tests."""
    import subprocess
    print("\n  Running tests...\n")
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', 'tests/', '-v', '--tb=short'],
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        if result.returncode == 0:
            print("\n  All tests passed!")
        else:
            print(f"\n  Some tests failed (exit code {result.returncode})")
    except Exception as e:
        print(f"\n  Error running tests: {e}")


def cmd_config():
    """Show current configuration."""
    import yaml
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("\n  Current Configuration:")
        print("  " + "-" * 40)
        print(f"  Model:      {config['model']['base_model']}")
        print(f"  Pipeline:   {config['training']['pipeline']}")
        print(f"  Epochs:     {config['training']['num_epochs']}")
        print(f"  Batch size: {config['training']['batch_size']}")
        print(f"  LR:         {config['training']['learning_rate']}")
        print(f"  Max length: {config['model']['max_seq_length']}")
        print(f"  Device:     {'CUDA' if config['device']['use_cuda'] else 'CPU'}")
        print(f"  API port:   {config.get('api', {}).get('port', 8000)}")
        ext = config.get('external_api', {})
        if ext:
            print(f"\n  External APIs:")
            for prov in ('openai', 'anthropic', 'google', 'deepseek', 'ollama'):
                prov_cfg = ext.get(prov, {})
                if prov_cfg:
                    key = prov_cfg.get('api_key', '')
                    has_key = bool(key and key != 'your-api-key-here')
                    status = 'configured' if has_key else 'no key'
                    if prov == 'ollama':
                        status = 'local'
                    print(f"    {prov:<12} [{status}]")
        print()
    except Exception as e:
        print(f"\nError reading config: {e}")


def cmd_status():
    """Check model and checkpoint status."""
    import yaml
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        ckpt_dir = config['checkpoint']['save_dir']
        tok_path = os.path.join(ckpt_dir, 'tokenizer.pkl')

        print("\n  Status:")
        print("  " + "-" * 40)

        # Tokenizer
        if os.path.exists(tok_path):
            size = os.path.getsize(tok_path)
            print(f"  Tokenizer:  Found ({size:,} bytes)")
        else:
            print(f"  Tokenizer:  Not found (run 'train' first)")

        # Checkpoints
        if os.path.exists(ckpt_dir):
            ckpts = [f for f in os.listdir(ckpt_dir)
                     if f.startswith('model_epoch_') and f.endswith('.pt')]
            best = os.path.exists(os.path.join(ckpt_dir, 'best_model.pt'))
            print(f"  Checkpoints: {len(ckpts)} found")
            print(f"  Best model:  {'Yes' if best else 'No'}")
            if ckpts:
                ckpts.sort()
                latest = ckpts[-1]
                size = os.path.getsize(os.path.join(ckpt_dir, latest))
                print(f"  Latest:      {latest} ({size:,} bytes)")
        else:
            print(f"  Checkpoints: Directory not found")

        # Registered models
        try:
            from models.registry import _load_registry
            registry = _load_registry()
            n_models = len(registry.get('models', {}))
            print(f"  Registered:  {n_models} model(s)")
        except Exception:
            pass

        # GPU
        try:
            import torch
            if torch.cuda.is_available():
                print(f"  GPU:         {torch.cuda.get_device_name(0)}")
                mem = torch.cuda.get_device_properties(0).total_mem
                print(f"  GPU Memory:  {mem / 1e9:.1f} GB")
            else:
                print(f"  GPU:         Not available (CPU only)")
        except Exception:
            print(f"  GPU:         Check failed")

        print()
    except Exception as e:
        print(f"\nError: {e}")


def cmd_prompt_train():
    """Create a custom model from interactive prompts."""
    from training.prompt_trainer import prompt_train_interactive
    try:
        prompt_train_interactive()
    except KeyboardInterrupt:
        print("\n  Prompt training cancelled.")
    except Exception as e:
        print(f"\n  Prompt training error: {e}")


def cmd_registry():
    """List all registered models."""
    from models.registry import list_registered_models
    list_registered_models()


def cmd_load_model():
    """Chat with a registered model."""
    from models.registry import list_registered_models, load_registered_model, get_model_info
    import yaml

    list_registered_models()

    try:
        name = input("  Enter model name to load: ").strip()
        if not name:
            return
    except (KeyboardInterrupt, EOFError):
        return

    info = get_model_info(name)
    if not info:
        print(f"  Model '{name}' not found.")
        return

    print(f"\n  Loading '{info['name']}'...")
    print(f"  Intent: {info['intent']}")
    print(f"  Base: {info['base_model']} / {info['pipeline']}")

    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except Exception:
        config = {'device': {'use_cuda': False, 'cuda_device': 0},
                  'generation': {'max_length': 100, 'temperature': 0.8,
                                 'top_k': 50, 'top_p': 0.9, 'repetition_penalty': 1.2}}

    import torch
    device = 'cpu'
    if config['device']['use_cuda'] and torch.cuda.is_available():
        device = f"cuda:{config['device']['cuda_device']}"

    model, tokenizer = load_registered_model(name, device=device)
    if model is None:
        return

    from services.chat import interactive_chat
    interactive_chat(model, tokenizer, config, device)


def cmd_external():
    """Chat via external API."""
    from services.external_api import interactive_external_chat
    try:
        interactive_external_chat()
    except KeyboardInterrupt:
        print("\n  External chat ended.")
    except Exception as e:
        print(f"\n  External API error: {e}")


def cmd_providers():
    """List external API providers."""
    from services.external_api import list_providers
    list_providers()


def cmd_refresh_models():
    """Refresh model lists from HuggingFace Hub and external API providers."""
    print("\n" + "=" * 55)
    print("       Model Registry Refresh")
    print("=" * 55)

    # Refresh local HuggingFace baseline models
    try:
        from models.model_factory import refresh_models
        refresh_models(force=True)
    except Exception as e:
        print(f"\n  Error refreshing HuggingFace models: {e}")

    # Refresh external API provider model lists
    try:
        from services.external_api import refresh_provider_models
        refresh_provider_models(force=True)
    except Exception as e:
        print(f"\n  Error refreshing API providers: {e}")

    print("  Refresh complete!")


def cmd_model_families():
    """List models grouped by family."""
    from models.model_factory import list_models
    list_models(by_family=True)


def cmd_load():
    """Load any model by number — chat or train.  Sets the active model for subsequent 'chat' commands."""
    global _active_model, _active_tokenizer, _active_device, _active_model_name

    from models.loader import interactive_load_and_act
    try:
        result = interactive_load_and_act()
        if result is not None:
            model, tokenizer, device, name = result
            _active_model = model
            _active_tokenizer = tokenizer
            _active_device = device
            _active_model_name = name
            print(f"\n  ✓ Active model set: {name}")
            print("  Use 'chat' (option 7) to continue chatting with this model.")
    except KeyboardInterrupt:
        print("\n  Model loader cancelled.")
    except Exception as e:
        print(f"\n  Model loader error: {e}")


def cmd_image_gen():
    """Image generation and tag-based training."""
    from training.image_gen import interactive_image_gen
    try:
        interactive_image_gen()
    except KeyboardInterrupt:
        print("\n  Image generation cancelled.")
    except Exception as e:
        print(f"\n  Image generation error: {e}")


def cmd_auto_train():
    """Auto-train a model from public domain data (web, Wikipedia, etc.)."""
    from training.auto_trainer import auto_train_interactive
    try:
        auto_train_interactive()
    except KeyboardInterrupt:
        print("\n  Auto training cancelled.")
    except Exception as e:
        print(f"\n  Auto training error: {e}")


def cmd_auto_image():
    """Auto-collect image tags and build datasets from public sources."""
    from training.image_auto_trainer import auto_image_train_interactive
    try:
        auto_image_train_interactive()
    except KeyboardInterrupt:
        print("\n  Image auto training cancelled.")
    except Exception as e:
        print(f"\n  Image auto training error: {e}")


def cmd_explore():
    """Unified model explorer — access local, saved, and API models from one place."""
    print("\n" + "=" * 55)
    print("       Model Explorer")
    print("=" * 55)

    while True:
        print("\n  Choose a model source:")
        print("  " + "-" * 50)
        print("  1   Local Model     Trained checkpoint (checkpoints/)")
        print("  2   Saved Models    Registered/named models")
        print("  3   API Models      External APIs (OpenAI, etc.)")
        print("  0   Back            Return to main menu")
        print()

        try:
            choice = input("  explore>> ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n  Returning to menu...")
            break

        if choice in ('0', 'back', 'quit', 'exit', 'q'):
            break

        if choice in ('1', 'local'):
            _explore_local()
        elif choice in ('2', 'saved', 'registry'):
            _explore_saved()
        elif choice in ('3', 'api', 'external'):
            _explore_api()
        else:
            print(f"  Unknown option: '{choice}'")


def _explore_local():
    """Explore: load and chat with the local trained model."""
    import yaml
    import torch

    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"\n  Error reading config: {e}")
        return

    ckpt_dir = config['checkpoint']['save_dir']
    best_path = os.path.join(ckpt_dir, 'best_model.pt')
    tok_path = os.path.join(ckpt_dir, 'tokenizer.pkl')

    # Check what's available
    has_best = os.path.exists(best_path)
    has_tok = os.path.exists(tok_path)
    ckpts = []
    if os.path.exists(ckpt_dir):
        ckpts = sorted([f for f in os.listdir(ckpt_dir)
                        if f.startswith('model_epoch_') and f.endswith('.pt')])

    print(f"\n  Local Model Status ({ckpt_dir}/):")
    print("  " + "-" * 50)
    print(f"  Tokenizer:   {'Found' if has_tok else 'Not found'}")
    print(f"  Best model:  {'Found' if has_best else 'Not found'}")
    print(f"  Checkpoints: {len(ckpts)} found")

    if not has_tok or (not has_best and not ckpts):
        print("\n  No trained model available.")
        print("  Run 'train' first to create a local model.")
        return

    # List available checkpoints
    available = []
    if has_best:
        available.append(('best', best_path))
        print(f"\n  Available:")
        print(f"    best  - best_model.pt")
    for i, ckpt in enumerate(ckpts):
        available.append((str(i + 1), os.path.join(ckpt_dir, ckpt)))
        print(f"    {i + 1:<5} - {ckpt}")

    try:
        sel = input("\n  Load which? [best]: ").strip().lower() or 'best'
    except (KeyboardInterrupt, EOFError):
        return

    # Find the selected checkpoint
    model_path = None
    for key, path in available:
        if sel == key:
            model_path = path
            break

    if model_path is None:
        # Try matching by name
        full = os.path.join(ckpt_dir, sel)
        if os.path.exists(full):
            model_path = full
        elif os.path.exists(sel):
            model_path = sel
        else:
            print(f"  Checkpoint '{sel}' not found.")
            return

    # Load model
    print(f"\n  Loading {os.path.basename(model_path)}...")
    device = 'cpu'
    if config['device']['use_cuda'] and torch.cuda.is_available():
        device = f"cuda:{config['device']['cuda_device']}"

    from models.model_factory import load_model
    from models.tokenizer import Tokenizer
    from services.chat import interactive_chat

    try:
        model = load_model(model_path, device=device)
        tokenizer = Tokenizer.load(tok_path)
        interactive_chat(model, tokenizer, config, device)
    except Exception as e:
        print(f"  Error loading model: {e}")


def _explore_saved():
    """Explore: browse and chat with registered/named models."""
    from models.registry import list_registered_models, load_registered_model, get_model_info, _load_registry
    import yaml
    import torch

    registry = _load_registry()
    models = registry.get("models", {})

    if not models:
        print("\n  No registered models yet.")
        print("  Use 'prompt-train' (option 2) to create and register a custom model.")
        return

    list_registered_models()

    print("  Actions:")
    print("    Type a model name to load and chat")
    print("    Type 'info <name>' for details")
    print("    Type 'delete <name>' to remove a model")
    print("    Type 'back' to return")
    print()

    try:
        cmd = input("  explore/saved>> ").strip()
    except (KeyboardInterrupt, EOFError):
        return

    if not cmd or cmd.lower() in ('back', 'quit', 'exit'):
        return

    # Handle 'info <name>'
    if cmd.lower().startswith('info '):
        name = cmd[5:].strip()
        info = get_model_info(name)
        if info:
            print(f"\n  Model Details:")
            print(f"  " + "-" * 50)
            print(f"  Name:        {info['name']}")
            print(f"  Intent:      {info['intent']}")
            print(f"  Base Model:  {info['base_model']}")
            print(f"  Pipeline:    {info['pipeline']}")
            print(f"  Created:     {info.get('created', 'unknown')}")
            print(f"  Path:        {info['path']}")
            print(f"  Notes:       {info.get('notes', '')}")
            sources = info.get('data_sources', [])
            if sources:
                print(f"  Data:        {', '.join(sources)}")
            print()
        else:
            print(f"  Model '{name}' not found.")
        return

    # Handle 'delete <name>'
    if cmd.lower().startswith('delete '):
        name = cmd[7:].strip()
        from models.registry import delete_model
        try:
            confirm = input(f"  Delete '{name}'? [y/N]: ").strip().lower()
            if confirm == 'y':
                delete_model(name)
            else:
                print("  Cancelled.")
        except (KeyboardInterrupt, EOFError):
            pass
        return

    # Otherwise, treat as model name to load
    name = cmd
    info = get_model_info(name)
    if not info:
        print(f"  Model '{name}' not found in registry.")
        return

    print(f"\n  Loading '{info['name']}'...")
    print(f"  Intent: {info['intent']}")
    print(f"  Base: {info['base_model']} / {info['pipeline']}")

    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except Exception:
        config = {'device': {'use_cuda': False, 'cuda_device': 0},
                  'generation': {'max_length': 100, 'temperature': 0.8,
                                 'top_k': 50, 'top_p': 0.9, 'repetition_penalty': 1.2}}

    device = 'cpu'
    if config['device']['use_cuda'] and torch.cuda.is_available():
        device = f"cuda:{config['device']['cuda_device']}"

    model, tokenizer = load_registered_model(name, device=device)
    if model is None:
        return

    from services.chat import interactive_chat
    interactive_chat(model, tokenizer, config, device)


def _explore_api():
    """Explore: connect to an external API model."""
    from services.external_api import (interactive_external_chat, list_providers,
                              is_provider_configured, find_available_provider, PROVIDERS)

    list_providers()

    # Quick check: are any configured?
    available = find_available_provider()
    if available:
        print(f"  Quickest option: '{available}' is ready to use.")
        print()

    try:
        interactive_external_chat()
    except KeyboardInterrupt:
        print("\n  External chat ended.")
    except Exception as e:
        print(f"\n  External API error: {e}")


# =========================================================================
# New module command handlers
# =========================================================================

def cmd_dataset_mgr():
    """Dataset management hub — versioning, quality, augmentation, dedup."""
    from utils.dataset_manager import interactive_dataset_manager
    try:
        interactive_dataset_manager()
    except KeyboardInterrupt:
        print("\n  Dataset manager cancelled.")
    except Exception as e:
        print(f"\n  Dataset manager error: {e}")


def cmd_curriculum():
    """Curriculum learning — progressive training and domain mixing."""
    from utils.curriculum import interactive_curriculum
    try:
        interactive_curriculum()
    except KeyboardInterrupt:
        print("\n  Curriculum manager cancelled.")
    except Exception as e:
        print(f"\n  Curriculum error: {e}")


def cmd_eval():
    """Evaluation suite — BLEU, ROUGE, perplexity, A/B comparison."""
    from utils.eval_suite import interactive_eval
    try:
        interactive_eval()
    except KeyboardInterrupt:
        print("\n  Evaluation cancelled.")
    except Exception as e:
        print(f"\n  Evaluation error: {e}")


def cmd_dashboard():
    """Training dashboard — live loss curves in a web UI."""
    from utils.training_dashboard import interactive_dashboard
    try:
        interactive_dashboard()
    except KeyboardInterrupt:
        print("\n  Dashboard stopped.")
    except Exception as e:
        print(f"\n  Dashboard error: {e}")


def cmd_tag_mgr():
    """Tag management — frequency analysis, hierarchy, negative prompts."""
    from utils.tag_manager import interactive_tag_manager
    try:
        interactive_tag_manager()
    except KeyboardInterrupt:
        print("\n  Tag manager cancelled.")
    except Exception as e:
        print(f"\n  Tag manager error: {e}")


def cmd_image_tools():
    """Image tools — upscale, variations, color transfer, LoRA merge."""
    from utils.image_tools import interactive_image_tools
    try:
        interactive_image_tools()
    except KeyboardInterrupt:
        print("\n  Image tools cancelled.")
    except Exception as e:
        print(f"\n  Image tools error: {e}")


def cmd_rag():
    """RAG — ingest documents, context-aware chat with citations."""
    from services.rag import interactive_rag
    try:
        interactive_rag()
    except KeyboardInterrupt:
        print("\n  RAG cancelled.")
    except Exception as e:
        print(f"\n  RAG error: {e}")


def cmd_agent():
    """Agent framework — tool use, memory, multi-step reasoning."""
    from services.agent import interactive_agent
    try:
        interactive_agent()
    except KeyboardInterrupt:
        print("\n  Agent stopped.")
    except Exception as e:
        print(f"\n  Agent error: {e}")


def cmd_packager():
    """Model packaging — export/import .tar.gz model archives."""
    from utils.model_packager import interactive_packager
    try:
        interactive_packager()
    except KeyboardInterrupt:
        print("\n  Packager cancelled.")
    except Exception as e:
        print(f"\n  Packager error: {e}")


def cmd_plugins():
    """Plugin manager — discover, load, and manage plugins."""
    from plugins.loader import get_plugin_manager
    try:
        pm = get_plugin_manager()
        print("\n" + "=" * 55)
        print("       Plugin Manager")
        print("=" * 55)
        pm.discover()
        plugins = pm.list_plugins()
        if plugins:
            print(f"\n  Found {len(plugins)} plugin(s):")
            for p in plugins:
                status = "loaded" if p.get('loaded') else "available"
                print(f"    {p['name']:<20} [{status}]  {p.get('description', '')}")
        else:
            print("\n  No plugins found.")
            print("  Create a plugins/<name>/manifest.json to add one.")
        print()
    except KeyboardInterrupt:
        print("\n  Plugin manager cancelled.")
    except Exception as e:
        print(f"\n  Plugin manager error: {e}")


def cmd_config_mgr():
    """Config manager — profiles, env overrides, validation."""
    from utils.config_manager import interactive_config_manager
    try:
        interactive_config_manager()
    except KeyboardInterrupt:
        print("\n  Config manager cancelled.")
    except Exception as e:
        print(f"\n  Config manager error: {e}")


def cmd_compat_api():
    """Start OpenAI-compatible API server."""
    from services.api_compat import create_app
    print("\n  Starting OpenAI-compatible API on port 8001...")
    print("  Endpoints: /v1/chat/completions, /v1/images/generations, /v1/models")
    print("  Use as base_url: http://localhost:8001/v1")
    print("  Press Ctrl+C to stop.\n")
    try:
        app = create_app()
        app.run(host='0.0.0.0', port=8001, debug=False)
    except KeyboardInterrupt:
        print("\n  Compat API stopped.")
    except Exception as e:
        print(f"\n  Compat API error: {e}")


def cmd_web_ui():
    """Launch browser-based web interface."""
    from services.web_ui import interactive_web_ui
    try:
        interactive_web_ui()
    except KeyboardInterrupt:
        print("\n  Web UI stopped.")
    except Exception as e:
        print(f"\n  Web UI error: {e}")


def cmd_tutorial():
    """Run the interactive tutorial."""
    from docs.tutorial import run_tutorial
    try:
        run_tutorial()
    except KeyboardInterrupt:
        print("\n  Tutorial ended.")
    except Exception as e:
        print(f"\n  Tutorial error: {e}")


def cmd_local_models():
    """Local model manager — list, load, verify, and uninstall all locally stored models."""
    from models.loader import interactive_local_models
    try:
        interactive_local_models()
    except KeyboardInterrupt:
        print("\n  Local model manager cancelled.")
    except Exception as e:
        print(f"\n  Local model manager error: {e}")


def cmd_verify_tokenizers():
    """Verify integrity of all local tokenizers."""
    from models.loader import verify_tokenizer_integrity
    try:
        verify_tokenizer_integrity(verbose=True)
    except Exception as e:
        print(f"\n  Tokenizer verification error: {e}")


# =========================================================================
# Main
# =========================================================================

def main():
    """Main interactive runtime loop."""
    global _running

    print_banner()

    # 1. Load .env file into os.environ (before anything else reads env vars)
    _load_env_file()

    # 2. Check and install missing dependencies
    check_and_install_dependencies()

    # 3. Check API tokens and offer interactive setup (only in interactive mode)
    if len(sys.argv) <= 1:
        check_and_setup_api_tokens()

    # 4. Quick tokenizer integrity check (non-verbose, just reports issues)
    try:
        from models.loader import verify_tokenizer_integrity
        verify_tokenizer_integrity(verbose=False)
    except Exception:
        pass  # Don't block startup if verification fails

    commands = {
        # Training & Data
        '1': cmd_train, 'train': cmd_train,
        '2': cmd_prompt_train, 'prompt-train': cmd_prompt_train,
        '3': cmd_auto_train, 'auto-train': cmd_auto_train, 'auto': cmd_auto_train,
        '4': cmd_auto_image, 'auto-image': cmd_auto_image,
        '5': cmd_dataset_mgr, 'dataset-mgr': cmd_dataset_mgr, 'dataset': cmd_dataset_mgr,
        '6': cmd_curriculum, 'curriculum': cmd_curriculum,
        # Chat & Generation
        '7': cmd_chat, 'chat': cmd_chat,
        '8': cmd_image_gen, 'image-gen': cmd_image_gen, 'image': cmd_image_gen,
        '9': cmd_external, 'external': cmd_external,
        '10': cmd_rag, 'rag': cmd_rag,
        '11': cmd_agent, 'agent': cmd_agent,
        # Model Management
        '12': cmd_load, 'load': cmd_load,
        '13': cmd_explore, 'explore': cmd_explore,
        '14': cmd_registry, 'registry': cmd_registry,
        '15': cmd_load_model, 'load-model': cmd_load_model,
        '16': cmd_packager, 'packager': cmd_packager,
        # Evaluation & Monitoring
        '17': cmd_eval, 'eval': cmd_eval,
        '18': cmd_dashboard, 'dashboard': cmd_dashboard,
        # Image Tools
        '19': cmd_tag_mgr, 'tag-mgr': cmd_tag_mgr, 'tags': cmd_tag_mgr,
        '20': cmd_image_tools, 'image-tools': cmd_image_tools,
        # API & Servers
        '21': cmd_api, 'api': cmd_api,
        '22': cmd_compat_api, 'compat-api': cmd_compat_api, 'openai-api': cmd_compat_api,
        '23': cmd_web_ui, 'web-ui': cmd_web_ui, 'web': cmd_web_ui,
        '24': cmd_providers, 'providers': cmd_providers,
        # Infrastructure
        '25': cmd_plugins, 'plugins': cmd_plugins,
        '26': cmd_config_mgr, 'config-mgr': cmd_config_mgr,
        # Info & System
        '27': cmd_models, 'models': cmd_models,
        '28': cmd_pipelines, 'pipelines': cmd_pipelines,
        '29': cmd_model_families, 'model-families': cmd_model_families, 'families': cmd_model_families,
        '30': cmd_config, 'config': cmd_config,
        '31': cmd_status, 'status': cmd_status,
        '32': cmd_refresh_models, 'refresh-models': cmd_refresh_models, 'refresh': cmd_refresh_models,
        '33': cmd_tutorial, 'tutorial': cmd_tutorial,
        '34': cmd_test, 'test': cmd_test,
        # Local Model Tools
        '35': cmd_local_models, 'local-models': cmd_local_models, 'local': cmd_local_models,
        '36': cmd_verify_tokenizers, 'verify-tokens': cmd_verify_tokenizers, 'verify': cmd_verify_tokenizers,
    }

    # If command-line args provided, run directly
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd in commands:
            commands[cmd]()
            return
        elif cmd in ('--help', '-h', 'help'):
            print_menu()
            return
        else:
            print(f"Unknown command: {cmd}")
            print_menu()
            return

    # Interactive mode
    print_menu()

    while _running:
        try:
            choice = input(">> ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n\nShutting down. Goodbye!")
            break

        if not choice:
            continue

        if choice in ('0', 'stop', 'quit', 'exit', 'q'):
            print("\nShutting down. Goodbye!")
            break

        if choice in ('help', 'menu', '?'):
            print_menu()
            continue

        if choice in commands:
            commands[choice]()
            print_menu()  # Show menu again after command
        else:
            print(f"  Unknown command: '{choice}'. Type 'help' for options.")

    sys.exit(0)


if __name__ == '__main__':
    main()
