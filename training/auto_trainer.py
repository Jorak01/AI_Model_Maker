"""Auto Trainer — Automated training from public domain data.

Searches the internet for public information on given topics, builds a
training dataset automatically, and trains (or fine-tunes) a model on it.

Usage:
    # Interactive
    python auto_trainer.py

    # Programmatic
    from training.auto_trainer import auto_train
    auto_train(
        topics=["quantum computing", "neural networks"],
        model_name="science-bot",
    )

    # CLI
    python auto_trainer.py --topics "quantum computing" "neural networks" --name science-bot
"""

import os
import json
import yaml
import torch
from typing import Optional, List, Dict

from utils.web_collector import WebCollector, save_collected_data
from models.tokenizer import Tokenizer
from models.model_factory import create_model
from utils.data_loader import create_data_loaders, load_and_prepare_data
from utils.trainer import Trainer
from models.registry import register_model


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device(config: dict) -> str:
    if config["device"]["use_cuda"] and torch.cuda.is_available():
        return f"cuda:{config['device']['cuda_device']}"
    return "cpu"


# ---------------------------------------------------------------------------
# Core: automated collection + training
# ---------------------------------------------------------------------------

def auto_train(
    topics: List[str],
    model_name: str = "auto-trained",
    intent: str = "",
    sources: Optional[List[str]] = None,
    urls: Optional[List[str]] = None,
    max_pairs_per_topic: int = 150,
    base_model: str = "custom",
    pipeline: str = "scratch",
    epochs: Optional[int] = None,
    merge_existing: bool = False,
    config_path: str = "config.yaml",
    interactive: bool = False,
) -> str:
    """Collect public data, build a dataset, and train a model — fully automated.

    Args:
        topics: List of search queries / topics to collect data about
        model_name: Name for the trained model in the registry
        intent: Description of the model's purpose
        sources: Data sources to use (wikipedia, web, stackexchange). Default: all.
        urls: Additional specific URLs to fetch data from
        max_pairs_per_topic: Max training pairs to collect per topic
        base_model: Which model architecture (custom, gpt2, distilgpt2, etc.)
        pipeline: Training pipeline (scratch, finetune, freeze)
        epochs: Number of training epochs (None = use config default)
        merge_existing: Merge with any existing data for this model
        config_path: Path to config.yaml
        interactive: If True, pause for user confirmations

    Returns:
        Path to the registered model directory
    """
    config = load_config(config_path)
    device = get_device(config)
    auto_cfg = config.get("auto_training", {})

    # Defaults from config
    if sources is None:
        sources = auto_cfg.get("sources", ["wikipedia", "web"])
    if max_pairs_per_topic == 150:
        max_pairs_per_topic = auto_cfg.get("max_pairs_per_topic", 150)
    if epochs is None:
        epochs = int(auto_cfg.get("epochs", config["training"]["num_epochs"]))
    if not intent:
        intent = f"Auto-trained on: {', '.join(topics[:5])}"

    safe_name = "".join(
        c if c.isalnum() or c in "-_" else "_" for c in model_name.lower().strip()
    )

    print("\n" + "=" * 60)
    print("       Auto Trainer — Public Domain Data Collection")
    print("=" * 60)
    print(f"\n  Model name : {model_name}")
    print(f"  Topics     : {topics}")
    print(f"  Sources    : {sources}")
    print(f"  Base model : {base_model} / {pipeline}")
    print(f"  Max pairs  : {max_pairs_per_topic} per topic")
    print(f"  Epochs     : {epochs}")
    if urls:
        print(f"  Extra URLs : {len(urls)}")
    print()

    if interactive:
        try:
            confirm = input("  Proceed? [Y/n]: ").strip().lower()
            if confirm == "n":
                print("  Cancelled.")
                return ""
        except (KeyboardInterrupt, EOFError):
            print("\n  Cancelled.")
            return ""

    # ── Step 1: Collect data from the internet ──────────────────────
    print("\n" + "-" * 60)
    print("  Step 1/4: Collecting data from public sources...")
    print("-" * 60)

    collector = WebCollector(verbose=True)

    if len(topics) == 1:
        pairs = collector.collect(
            topics[0],
            max_pairs=max_pairs_per_topic,
            sources=sources,
            urls=urls,
        )
    else:
        pairs = collector.collect_multi_topic(
            topics,
            max_pairs_per_topic=max_pairs_per_topic,
            sources=sources,
        )
        # Also fetch any extra URLs
        if urls:
            for url in urls:
                url_pairs = collector.collect_from_url(url, topic=topics[0])
                pairs.extend(url_pairs)

    if not pairs:
        print("\n  ❌ No data collected. Check your internet connection or try different topics.")
        return ""

    print(f"\n  ✓ Collected {len(pairs)} training pairs")

    # ── Step 2: Save dataset ────────────────────────────────────────
    print("\n" + "-" * 60)
    print("  Step 2/4: Building training dataset...")
    print("-" * 60)

    data_dir = os.path.join("data", "auto")
    os.makedirs(data_dir, exist_ok=True)
    data_path = os.path.join(data_dir, f"{safe_name}_train.json")

    save_collected_data(pairs, data_path, merge_existing=merge_existing)

    # Show a sample
    print(f"\n  Sample pairs:")
    for p in pairs[:3]:
        print(f"    Q: {p['prompt'][:70]}...")
        print(f"    A: {p['response'][:70]}...")
        print()

    if interactive:
        try:
            confirm = input("  Continue to training? [Y/n]: ").strip().lower()
            if confirm == "n":
                print(f"  Data saved to {data_path}. Training skipped.")
                return data_path
        except (KeyboardInterrupt, EOFError):
            print(f"\n  Data saved to {data_path}. Training skipped.")
            return data_path

    # ── Step 3: Build tokenizer & data loaders ──────────────────────
    print("\n" + "-" * 60)
    print("  Step 3/4: Preparing model and tokenizer...")
    print("-" * 60)

    ckpt_dir = os.path.join("checkpoints", safe_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    tok_path = os.path.join(ckpt_dir, "tokenizer.pkl")

    # Build tokenizer
    print("  Building tokenizer...")
    tokenizer = Tokenizer(config["model"]["vocab_size"], method="word")
    texts = load_and_prepare_data(data_path)
    tokenizer.build_vocab(texts)
    tokenizer.save(tok_path)

    # Data loaders
    perf = config.get("performance", {})
    num_workers = perf.get("num_workers", 0)
    print(f"  Creating data loaders (workers={num_workers})...")
    train_loader, test_loader = create_data_loaders(
        data_path,
        data_path,  # Use same data for train/test (auto-collected)
        tokenizer,
        config["training"]["batch_size"],
        config["model"]["max_seq_length"],
        num_workers=num_workers,
    )

    # Model
    print("  Initializing model...")
    train_config = dict(config)
    train_config["model"] = dict(config["model"])
    train_config["training"] = dict(config["training"])
    train_config["model"]["base_model"] = base_model
    train_config["training"]["pipeline"] = pipeline
    if base_model == "custom":
        train_config["model"]["vocab_size"] = len(tokenizer)
    model = create_model(train_config)

    # ── Step 4: Train ───────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("  Step 4/4: Training model...")
    print("-" * 60)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        learning_rate=config["training"]["learning_rate"],
        device=device,
        checkpoint_dir=ckpt_dir,
        gradient_clip=config["training"]["gradient_clip"],
        warmup_steps=min(
            config["training"]["warmup_steps"],
            len(train_loader) * int(epochs) // 2,
        ),
        weight_decay=config["training"]["weight_decay"],
        pad_token_id=tokenizer.pad_token_id,
        use_amp=perf.get("mixed_precision", True),
        compile_model=perf.get("compile_model", True),
    )
    trainer.train(num_epochs=int(epochs), save_every=1, keep_last=3)

    # ── Register model ──────────────────────────────────────────────
    print("\n  Registering model...")
    source_list: List[str] = sources if sources is not None else []
    model_dir = register_model(
        name=model_name,
        intent=intent,
        base_model=base_model,
        pipeline=pipeline,
        checkpoint_dir=ckpt_dir,
        data_sources=[data_path],
        notes=(
            f"Auto-trained on {len(pairs)} pairs from public sources. "
            f"Topics: {', '.join(topics[:5])}. "
            f"Sources: {', '.join(source_list)}."
        ),
    )

    print("\n" + "=" * 60)
    print(f"  ✓ Model '{model_name}' trained and registered!")
    print(f"    Pairs collected : {len(pairs)}")
    print(f"    Topics          : {', '.join(topics)}")
    print(f"    Data saved to   : {data_path}")
    print(f"    Model saved to  : {model_dir}")
    print(f"    Use 'registry' command to see all models.")
    print(f"    Use 'load-model' to chat with this model.")
    print("=" * 60 + "\n")

    return model_dir


# ---------------------------------------------------------------------------
# Collect-only mode (no training)
# ---------------------------------------------------------------------------

def auto_collect(
    topics: List[str],
    output_name: str = "collected",
    sources: Optional[List[str]] = None,
    urls: Optional[List[str]] = None,
    max_pairs_per_topic: int = 150,
    merge_existing: bool = True,
) -> str:
    """Collect data from public sources and save as a training dataset, without training.

    Returns:
        Path to saved dataset file
    """
    print("\n" + "=" * 60)
    print("       Auto Collector — Public Domain Data")
    print("=" * 60)
    print(f"\n  Topics: {topics}")
    print(f"  Sources: {sources or ['wikipedia', 'web']}")

    collector = WebCollector(verbose=True)

    if len(topics) == 1:
        pairs = collector.collect(
            topics[0],
            max_pairs=max_pairs_per_topic,
            sources=sources,
            urls=urls,
        )
    else:
        pairs = collector.collect_multi_topic(
            topics,
            max_pairs_per_topic=max_pairs_per_topic,
            sources=sources,
        )
        if urls:
            for url in urls:
                url_pairs = collector.collect_from_url(url, topic=topics[0])
                pairs.extend(url_pairs)

    if not pairs:
        print("\n  ❌ No data collected.")
        return ""

    safe_name = "".join(
        c if c.isalnum() or c in "-_" else "_" for c in output_name.lower().strip()
    )
    data_dir = os.path.join("data", "auto")
    data_path = os.path.join(data_dir, f"{safe_name}_train.json")
    save_collected_data(pairs, data_path, merge_existing=merge_existing)

    print(f"\n  ✓ Collected {len(pairs)} pairs → {data_path}")
    return data_path


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------

def auto_train_interactive(config_path: str = "config.yaml"):
    """Full interactive auto-training flow.

    Steps:
    1. Ask for model name and topics
    2. Choose data sources
    3. Choose base model and pipeline
    4. Collect data from the internet
    5. Train the model
    6. Register it
    """
    config = load_config(config_path)

    print("\n" + "=" * 60)
    print("       Auto Trainer — Build Models from Public Data")
    print("=" * 60)
    print("\n  This will search the internet for public domain information,")
    print("  build a training dataset, and train a model automatically.")

    # 1. Model name
    try:
        print("\n  What should this model be called?")
        print("  (e.g., 'history-bot', 'science-helper', 'python-tutor')")
        model_name = input("  Name: ").strip()
        if not model_name:
            print("  Cancelled.")
            return
    except (KeyboardInterrupt, EOFError):
        print("\n  Cancelled.")
        return

    # 2. Topics
    try:
        print("\n  Enter topics to search for (comma-separated):")
        print("  (e.g., 'quantum physics, relativity, particle physics')")
        topics_raw = input("  Topics: ").strip()
        if not topics_raw:
            print("  No topics entered. Cancelled.")
            return
        topics = [t.strip() for t in topics_raw.split(",") if t.strip()]
    except (KeyboardInterrupt, EOFError):
        print("\n  Cancelled.")
        return

    # 3. Intent
    try:
        print(f"\n  What is the model's purpose? (default: knowledge about {topics[0]})")
        intent = input("  Intent: ").strip()
        if not intent:
            intent = f"Knowledge about: {', '.join(topics[:3])}"
    except (KeyboardInterrupt, EOFError):
        intent = f"Knowledge about: {', '.join(topics[:3])}"

    # 4. Data sources
    try:
        print("\n  Which data sources to use?")
        print("  1. Wikipedia + Web search (default, recommended)")
        print("  2. Wikipedia only (highest quality)")
        print("  3. Web search only")
        print("  4. All sources (Wikipedia + Web + StackExchange)")
        src_choice = input("  [1/2/3/4, default=1]: ").strip()

        source_map = {
            "1": ["wikipedia", "web"],
            "2": ["wikipedia"],
            "3": ["web"],
            "4": ["wikipedia", "web", "stackexchange"],
        }
        sources = source_map.get(src_choice, ["wikipedia", "web"])
    except (KeyboardInterrupt, EOFError):
        sources = ["wikipedia", "web"]

    # 5. Additional URLs
    urls: List[str] = []
    try:
        print("\n  Any specific URLs to include? (comma-separated, or press Enter to skip)")
        urls_raw = input("  URLs: ").strip()
        if urls_raw:
            urls = [u.strip() for u in urls_raw.split(",") if u.strip()]
    except (KeyboardInterrupt, EOFError):
        pass

    # 6. Max pairs
    auto_cfg = config.get("auto_training", {})
    default_max = auto_cfg.get("max_pairs_per_topic", 150)
    try:
        print(f"\n  Max training pairs per topic? (default={default_max})")
        mp = input("  Max pairs: ").strip()
        max_pairs = int(mp) if mp else default_max
    except (KeyboardInterrupt, EOFError, ValueError):
        max_pairs = default_max

    # 7. Base model & pipeline
    try:
        print("\n  Choose base model:")
        print("  1. custom (train from scratch, ~2M params)")
        print("  2. gpt2 (fine-tune GPT-2)")
        print("  3. distilgpt2 (fine-tune DistilGPT-2, lighter)")
        choice = input("  [1/2/3, default=1]: ").strip()

        base_model = {"2": "gpt2", "3": "distilgpt2"}.get(choice, "custom")
        pipeline = "scratch" if base_model == "custom" else "finetune"
    except (KeyboardInterrupt, EOFError):
        base_model = "custom"
        pipeline = "scratch"

    # 8. Epochs
    default_epochs = auto_cfg.get("epochs", config["training"]["num_epochs"])
    try:
        print(f"\n  Training epochs? (default={default_epochs})")
        ep = input("  Epochs: ").strip()
        epochs = int(ep) if ep else default_epochs
    except (KeyboardInterrupt, EOFError, ValueError):
        epochs = default_epochs

    # 9. Collect-only option
    try:
        print("\n  Mode:")
        print("  1. Collect data AND train (default)")
        print("  2. Collect data only (save dataset, no training)")
        mode = input("  [1/2, default=1]: ").strip()
    except (KeyboardInterrupt, EOFError):
        mode = "1"

    if mode == "2":
        auto_collect(
            topics=topics,
            output_name=model_name,
            sources=sources,
            urls=urls,
            max_pairs_per_topic=max_pairs,
        )
        return

    # 10. Run auto training
    auto_train(
        topics=topics,
        model_name=model_name,
        intent=intent,
        sources=sources,
        urls=urls,
        max_pairs_per_topic=max_pairs,
        base_model=base_model,
        pipeline=pipeline,
        epochs=epochs,
        config_path=config_path,
        interactive=True,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Auto Trainer — Build models from public domain data"
    )
    parser.add_argument(
        "--topics", nargs="+",
        help="Topics to search for (e.g., 'machine learning' 'neural networks')",
    )
    parser.add_argument("--name", default="auto-trained", help="Model name")
    parser.add_argument("--intent", default="", help="Model intent/purpose")
    parser.add_argument(
        "--sources", nargs="+", default=None,
        help="Data sources: wikipedia, web, stackexchange",
    )
    parser.add_argument(
        "--urls", nargs="+", default=None,
        help="Additional URLs to fetch data from",
    )
    parser.add_argument("--max-pairs", type=int, default=150, help="Max pairs per topic")
    parser.add_argument("--base-model", default="custom", help="Base model to use")
    parser.add_argument("--pipeline", default="scratch", help="Training pipeline")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--collect-only", action="store_true", help="Only collect data, don't train")
    parser.add_argument("--config", default="config.yaml", help="Config file path")

    args = parser.parse_args()

    if args.topics:
        if args.collect_only:
            auto_collect(
                topics=args.topics,
                output_name=args.name,
                sources=args.sources,
                urls=args.urls,
                max_pairs_per_topic=args.max_pairs,
            )
        else:
            auto_train(
                topics=args.topics,
                model_name=args.name,
                intent=args.intent,
                sources=args.sources,
                urls=args.urls,
                max_pairs_per_topic=args.max_pairs,
                base_model=args.base_model,
                pipeline=args.pipeline,
                epochs=args.epochs,
                config_path=args.config,
            )
    else:
        # Interactive mode
        auto_train_interactive(args.config)
