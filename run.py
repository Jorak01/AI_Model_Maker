"""
AI Model Runtime - Interactive entry point.

Run with: python run.py
All functions are accessible from the interactive menu.
Type 'stop' or 'quit' at any time to exit.
"""

import sys
import os
import signal

# Global flag for graceful shutdown
_running = True


def signal_handler(sig, frame):
    global _running
    _running = False
    print("\n\nShutting down...")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


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
    print("   0  stop / quit      Exit")
    print()


# =========================================================================
# Command handlers
# =========================================================================

def cmd_train():
    """Run training."""
    from train import main as train_main
    try:
        train_main()
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    except Exception as e:
        print(f"\nTraining error: {e}")


def cmd_chat():
    """Run interactive chat."""
    from chat import main as chat_main
    try:
        chat_main()
    except KeyboardInterrupt:
        print("\nChat ended.")
    except Exception as e:
        print(f"\nChat error: {e}")


def cmd_api():
    """Start the API server."""
    from api import main as api_main
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
            from model_registry import _load_registry
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
    from prompt_trainer import prompt_train_interactive
    try:
        prompt_train_interactive()
    except KeyboardInterrupt:
        print("\n  Prompt training cancelled.")
    except Exception as e:
        print(f"\n  Prompt training error: {e}")


def cmd_registry():
    """List all registered models."""
    from model_registry import list_registered_models
    list_registered_models()


def cmd_load_model():
    """Chat with a registered model."""
    from model_registry import list_registered_models, load_registered_model, get_model_info
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

    from chat import interactive_chat
    interactive_chat(model, tokenizer, config, device)


def cmd_external():
    """Chat via external API."""
    from external_api import interactive_external_chat
    try:
        interactive_external_chat()
    except KeyboardInterrupt:
        print("\n  External chat ended.")
    except Exception as e:
        print(f"\n  External API error: {e}")


def cmd_providers():
    """List external API providers."""
    from external_api import list_providers
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
        from external_api import refresh_provider_models
        refresh_provider_models(force=True)
    except Exception as e:
        print(f"\n  Error refreshing API providers: {e}")

    print("  Refresh complete!")


def cmd_model_families():
    """List models grouped by family."""
    from models.model_factory import list_models
    list_models(by_family=True)


def cmd_load():
    """Load any model by number — chat or train."""
    from model_loader import interactive_load_and_act
    try:
        interactive_load_and_act()
    except KeyboardInterrupt:
        print("\n  Model loader cancelled.")
    except Exception as e:
        print(f"\n  Model loader error: {e}")


def cmd_image_gen():
    """Image generation and tag-based training."""
    from image_gen import interactive_image_gen
    try:
        interactive_image_gen()
    except KeyboardInterrupt:
        print("\n  Image generation cancelled.")
    except Exception as e:
        print(f"\n  Image generation error: {e}")


def cmd_auto_train():
    """Auto-train a model from public domain data (web, Wikipedia, etc.)."""
    from auto_trainer import auto_train_interactive
    try:
        auto_train_interactive()
    except KeyboardInterrupt:
        print("\n  Auto training cancelled.")
    except Exception as e:
        print(f"\n  Auto training error: {e}")


def cmd_auto_image():
    """Auto-collect image tags and build datasets from public sources."""
    from image_auto_trainer import auto_image_train_interactive
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
    from chat import interactive_chat

    try:
        model = load_model(model_path, device=device)
        tokenizer = Tokenizer.load(tok_path)
        interactive_chat(model, tokenizer, config, device)
    except Exception as e:
        print(f"  Error loading model: {e}")


def _explore_saved():
    """Explore: browse and chat with registered/named models."""
    from model_registry import list_registered_models, load_registered_model, get_model_info, _load_registry
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
        from model_registry import delete_model
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

    from chat import interactive_chat
    interactive_chat(model, tokenizer, config, device)


def _explore_api():
    """Explore: connect to an external API model."""
    from external_api import (interactive_external_chat, list_providers,
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
    from rag import interactive_rag
    try:
        interactive_rag()
    except KeyboardInterrupt:
        print("\n  RAG cancelled.")
    except Exception as e:
        print(f"\n  RAG error: {e}")


def cmd_agent():
    """Agent framework — tool use, memory, multi-step reasoning."""
    from agent import interactive_agent
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
    from api_compat import create_app
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
    from web_ui import interactive_web_ui
    try:
        interactive_web_ui()
    except KeyboardInterrupt:
        print("\n  Web UI stopped.")
    except Exception as e:
        print(f"\n  Web UI error: {e}")


def cmd_tutorial():
    """Run the interactive tutorial."""
    from tutorial import run_tutorial
    try:
        run_tutorial()
    except KeyboardInterrupt:
        print("\n  Tutorial ended.")
    except Exception as e:
        print(f"\n  Tutorial error: {e}")


# =========================================================================
# Main
# =========================================================================

def main():
    """Main interactive runtime loop."""
    global _running

    print_banner()

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
