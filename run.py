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
    print("  ─── Training & Chat ─────────────────────────")
    print("   1  train           Train the model")
    print("   2  chat            Interactive chat (local)")
    print("   3  prompt-train    Create model from prompts")
    print("   4  image-gen       Image generation & training")
    print()
    print("  ─── Model Management ────────────────────────")
    print("   5  load            Load any model by number")
    print("   6  explore         Browse all models")
    print("   7  registry        List registered models")
    print("   8  load-model      Chat with registered model")
    print()
    print("  ─── External & API ──────────────────────────")
    print("   9  external        Chat via external API")
    print("  10  providers       List API providers")
    print("  11  api             Start REST API server")
    print()
    print("  ─── Info & System ───────────────────────────")
    print("  12  models          Available base models")
    print("  13  pipelines       Training pipelines")
    print("  14  model-families  Models grouped by family")
    print("  15  config          Show configuration")
    print("  16  status          Model/checkpoint status")
    print("  17  refresh-models  Update model lists")
    print("  18  test            Run all tests")
    print()
    print("   0  stop / quit     Exit")
    print()


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
        print("  Use 'prompt-train' (option 3) to create and register a custom model.")
        return

    list_registered_models()

    # Show detailed info for each model
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


def main():
    """Main interactive runtime loop."""
    global _running

    print_banner()

    commands = {
        # Training & Chat
        '1': cmd_train, 'train': cmd_train,
        '2': cmd_chat, 'chat': cmd_chat,
        '3': cmd_prompt_train, 'prompt-train': cmd_prompt_train,
        '4': cmd_image_gen, 'image-gen': cmd_image_gen, 'image': cmd_image_gen,
        # Model Management
        '5': cmd_load, 'load': cmd_load,
        '6': cmd_explore, 'explore': cmd_explore,
        '7': cmd_registry, 'registry': cmd_registry,
        '8': cmd_load_model, 'load-model': cmd_load_model,
        # External & API
        '9': cmd_external, 'external': cmd_external,
        '10': cmd_providers, 'providers': cmd_providers,
        '11': cmd_api, 'api': cmd_api,
        # Info & System
        '12': cmd_models, 'models': cmd_models,
        '13': cmd_pipelines, 'pipelines': cmd_pipelines,
        '14': cmd_model_families, 'model-families': cmd_model_families, 'families': cmd_model_families,
        '15': cmd_config, 'config': cmd_config,
        '16': cmd_status, 'status': cmd_status,
        '17': cmd_refresh_models, 'refresh-models': cmd_refresh_models, 'refresh': cmd_refresh_models,
        '18': cmd_test, 'test': cmd_test,
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
