"""Prompt Trainer - Interactively enter prompts to create custom training data and models."""

import os
import json
import yaml
import torch
from typing import Optional, List, Dict
from models.tokenizer import Tokenizer
from models.model_factory import create_model, list_models, list_pipelines
from utils.data_loader import create_data_loaders, load_and_prepare_data
from utils.trainer import Trainer
from model_registry import register_model


def load_config(path: str = 'config.yaml') -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_device(config: dict) -> str:
    if config['device']['use_cuda'] and torch.cuda.is_available():
        return f"cuda:{config['device']['cuda_device']}"
    return 'cpu'


def collect_prompts() -> List[Dict[str, str]]:
    """Interactively collect prompt/response pairs from the user.

    Returns:
        List of {"prompt": ..., "response": ...} dicts
    """
    pairs: List[Dict[str, str]] = []
    print("\n  === Prompt Entry Mode ===")
    print("  Enter prompt/response pairs for training.")
    print("  Commands: 'done' = finish, 'show' = view entries, 'undo' = remove last")
    print("  " + "-" * 50)

    while True:
        try:
            print(f"\n  [{len(pairs) + 1}] Enter a PROMPT (or 'done'):")
            prompt = input("  > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Entry cancelled.")
            break

        if prompt.lower() == 'done':
            break
        if prompt.lower() == 'show':
            _show_pairs(pairs)
            continue
        if prompt.lower() == 'undo':
            if pairs:
                removed = pairs.pop()
                print(f"  Removed: '{removed['prompt'][:50]}...'")
            else:
                print("  Nothing to undo.")
            continue
        if not prompt:
            print("  Prompt cannot be empty.")
            continue

        try:
            print(f"  Enter the RESPONSE:")
            response = input("  > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Entry cancelled.")
            break

        if not response:
            print("  Response cannot be empty. Skipping.")
            continue

        pairs.append({"prompt": prompt, "response": response})
        print(f"  Added pair #{len(pairs)}")

    print(f"\n  Total pairs collected: {len(pairs)}")
    return pairs


def _show_pairs(pairs: List[Dict[str, str]]):
    """Display collected pairs."""
    if not pairs:
        print("  No pairs entered yet.")
        return
    print(f"\n  Collected Pairs ({len(pairs)}):")
    print("  " + "-" * 60)
    for i, p in enumerate(pairs, 1):
        prompt_preview = p['prompt'][:50] + ('...' if len(p['prompt']) > 50 else '')
        response_preview = p['response'][:50] + ('...' if len(p['response']) > 50 else '')
        print(f"  {i}. P: {prompt_preview}")
        print(f"     R: {response_preview}")
    print()


def save_training_data(pairs: List[Dict[str, str]], output_path: str,
                       merge_existing: bool = True) -> str:
    """Save prompt/response pairs to a JSON training file.

    Args:
        pairs: List of {"prompt": ..., "response": ...} dicts
        output_path: Path to save the JSON file
        merge_existing: If True, append to existing data file

    Returns:
        Path to the saved file
    """
    existing: List[Dict[str, str]] = []
    if merge_existing and os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        print(f"  Merging with {len(existing)} existing pairs")

    combined = existing + pairs
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=4, ensure_ascii=False)

    print(f"  Saved {len(combined)} pairs to {output_path}")
    return output_path


def prompt_train_interactive(config_path: str = 'config.yaml'):
    """Full interactive flow: collect prompts -> train -> register model.

    Steps:
    1. Ask user for model name and intent
    2. Collect prompt/response pairs
    3. Optionally include existing training data
    4. Choose base model and pipeline
    5. Train the model
    6. Register it in the model registry
    """
    config = load_config(config_path)
    device = get_device(config)

    print("\n" + "=" * 55)
    print("       Prompt-Based Model Trainer")
    print("=" * 55)

    # 1. Model name and intent
    try:
        print("\n  What should this model be called?")
        print("  (e.g., 'customer-support', 'code-helper', 'trivia-bot')")
        model_name = input("  Name: ").strip()
        if not model_name:
            print("  Cancelled.")
            return

        print("\n  What is the main intent/purpose of this model?")
        intent = input("  Intent: ").strip()
        if not intent:
            intent = model_name
    except (KeyboardInterrupt, EOFError):
        print("\n  Cancelled.")
        return

    # 2. Collect prompts
    pairs = collect_prompts()
    if not pairs:
        print("  No data entered. Cancelled.")
        return

    # 3. Merge with existing data?
    try:
        print(f"\n  Merge with existing training data ({config['data']['train_path']})?")
        merge = input("  [y/N]: ").strip().lower() == 'y'
    except (KeyboardInterrupt, EOFError):
        merge = False

    # Save custom data
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in model_name.lower().strip())
    data_dir = os.path.join("data", "custom")
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, f"{safe_name}_train.json")

    if merge:
        # Load existing data and merge
        existing = []
        if os.path.exists(config['data']['train_path']):
            with open(config['data']['train_path'], 'r', encoding='utf-8') as f:
                existing = json.load(f)
        all_pairs = existing + pairs
        save_training_data(all_pairs, data_path, merge_existing=False)
    else:
        save_training_data(pairs, data_path, merge_existing=False)

    # Use same file as both train and test (small custom dataset)
    test_path = data_path

    # 4. Choose base model and pipeline
    try:
        print("\n  Choose base model:")
        print("  1. custom (train from scratch)")
        print("  2. gpt2 (fine-tune GPT-2)")
        print("  3. distilgpt2 (fine-tune DistilGPT-2)")
        choice = input("  [1/2/3, default=1]: ").strip()

        base_model = {"2": "gpt2", "3": "distilgpt2"}.get(choice, "custom")
        pipeline = "scratch" if base_model == "custom" else "finetune"

        print(f"\n  Using: {base_model} / {pipeline}")
    except (KeyboardInterrupt, EOFError):
        base_model = "custom"
        pipeline = "scratch"

    # 5. Configure epochs
    try:
        print(f"\n  Training epochs? (default={config['training']['num_epochs']})")
        ep = input("  Epochs: ").strip()
        epochs = int(ep) if ep else config['training']['num_epochs']
    except (KeyboardInterrupt, EOFError, ValueError):
        epochs = config['training']['num_epochs']

    # 6. Train
    ckpt_dir = os.path.join("checkpoints", safe_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    tok_path = os.path.join(ckpt_dir, "tokenizer.pkl")

    print("\n  Building tokenizer...")
    tokenizer = Tokenizer(config['model']['vocab_size'], method='word')
    texts = load_and_prepare_data(data_path)
    tokenizer.build_vocab(texts)
    tokenizer.save(tok_path)

    print("  Creating data loaders...")
    train_loader, test_loader = create_data_loaders(
        data_path, test_path, tokenizer,
        config['training']['batch_size'], config['model']['max_seq_length']
    )

    print("  Initializing model...")
    train_config = dict(config)
    train_config['model'] = dict(config['model'])
    train_config['training'] = dict(config['training'])
    train_config['model']['base_model'] = base_model
    train_config['training']['pipeline'] = pipeline
    if base_model == "custom":
        train_config['model']['vocab_size'] = len(tokenizer)
    model = create_model(train_config)

    print("  Starting training...")
    trainer = Trainer(
        model=model, train_loader=train_loader, test_loader=test_loader,
        learning_rate=config['training']['learning_rate'], device=device,
        checkpoint_dir=ckpt_dir, gradient_clip=config['training']['gradient_clip'],
        warmup_steps=min(config['training']['warmup_steps'], len(train_loader) * epochs // 2),
        weight_decay=config['training']['weight_decay'],
        pad_token_id=tokenizer.pad_token_id
    )
    trainer.train(num_epochs=epochs, save_every=1, keep_last=3)

    # 7. Register model
    print("\n  Registering model...")
    register_model(
        name=model_name,
        intent=intent,
        base_model=base_model,
        pipeline=pipeline,
        checkpoint_dir=ckpt_dir,
        data_sources=[data_path],
        notes=f"Trained on {len(pairs)} custom prompt/response pairs"
    )

    print("\n" + "=" * 55)
    print(f"  Model '{model_name}' trained and registered!")
    print(f"  Use 'registry' command to see all models.")
    print(f"  Use 'load-model' to chat with this model.")
    print("=" * 55 + "\n")


def add_prompts_to_existing(model_name: str, config_path: str = 'config.yaml'):
    """Add more prompts to an existing model's training data and retrain."""
    from model_registry import get_model_info

    info = get_model_info(model_name)
    if not info:
        print(f"  Model '{model_name}' not found in registry.")
        return

    print(f"\n  Adding prompts to '{info['name']}' ({info['intent']})")

    # Collect new pairs
    pairs = collect_prompts()
    if not pairs:
        print("  No data entered.")
        return

    # Find existing data
    data_sources = info.get('data_sources', [])
    if data_sources and os.path.exists(data_sources[0]):
        data_path = data_sources[0]
        save_training_data(pairs, data_path, merge_existing=True)
    else:
        safe_name = info['safe_name']
        data_path = os.path.join("data", "custom", f"{safe_name}_train.json")
        save_training_data(pairs, data_path, merge_existing=False)

    print(f"\n  Data updated. Run 'prompt-train' with model '{model_name}' to retrain.")


if __name__ == '__main__':
    prompt_train_interactive()
