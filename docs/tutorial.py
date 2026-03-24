"""
Interactive Tutorial — Walk new users through the AI Model Suite step by step.

Run: python run.py tutorial
     python tutorial.py
"""

import os
import sys
import time


def _pause():
    """Wait for user to press Enter."""
    try:
        input("\n  Press Enter to continue...")
    except (KeyboardInterrupt, EOFError):
        raise KeyboardInterrupt


def _clear():
    """Clear screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


STEPS = [
    # (title, content)
    (
        "Welcome to the AI Model Suite",
        """  This tutorial will walk you through the key features of the suite.

  The AI Model Suite is a comprehensive toolkit for:
    • Training custom language models from scratch
    • Training from public domain data (Wikipedia, StackExchange, etc.)
    • Interactive chat with your trained models
    • Image generation and tag-based training
    • Connecting to external APIs (OpenAI, Anthropic, Google, etc.)
    • RAG (Retrieval-Augmented Generation)
    • Agent framework with tool use
    • And much more!

  You can exit this tutorial at any time by pressing Ctrl+C."""
    ),
    (
        "Step 1: Training a Model",
        """  The simplest way to get started is to train a model.

  From the main menu, choose option [1] 'train'

  This will:
    1. Load the base model architecture (configured in config.yaml)
    2. Load your training data from the data/ directory
    3. Train for the configured number of epochs
    4. Save checkpoints to checkpoints/

  Training data format (JSONL):
    {"prompt": "What is Python?", "completion": "Python is a programming language."}

  Tip: Start with a small dataset (100-1000 pairs) for quick experiments."""
    ),
    (
        "Step 2: Prompt Training (No Data Needed)",
        """  Don't have training data? Use Prompt Training!

  From the menu, choose option [2] 'prompt-train'

  This interactive mode lets you:
    1. Define your model's intent (e.g., "helpful coding assistant")
    2. Type example conversations directly
    3. The system creates a trained model from your examples
    4. Models are automatically registered in the model registry

  This is the fastest way to create a specialized model."""
    ),
    (
        "Step 3: Auto-Training from the Web",
        """  The Auto-Trainer collects public domain data automatically!

  From the menu, choose option [3] 'auto-train'

  It can pull from:
    • Wikipedia — articles on any topic
    • DuckDuckGo — web search results
    • StackExchange — Q&A pairs

  You choose the topics and how many samples to collect.
  The data is cleaned, deduplicated, and formatted for training.

  For image models, use option [4] 'auto-image' to collect
  tag-based training data from Danbooru and other sources."""
    ),
    (
        "Step 4: Chatting with Your Model",
        """  Once trained, chat with your model interactively!

  From the menu, choose option [5] 'chat'

  Chat features:
    • Real-time text generation
    • Adjustable temperature, top-k, top-p
    • Conversation history
    • Type 'quit' to exit

  Or use option [9] 'explore' to browse all available models
  (local, saved, and API) from one unified interface."""
    ),
    (
        "Step 5: External API Providers",
        """  Connect to cloud AI providers for powerful models!

  From the menu, choose option [7] 'external'

  Supported providers:
    • OpenAI (GPT-4, GPT-3.5)
    • Anthropic (Claude)
    • Google (Gemini)
    • DeepSeek
    • Ollama (local, no API key needed!)

  Configure API keys in config.yaml under 'external_api'.
  Ollama requires no key — just install and run Ollama locally."""
    ),
    (
        "Step 6: Dataset Management",
        """  The Dataset Manager helps you maintain high-quality training data.

  From the menu, choose 'dataset-mgr'

  Features:
    • Version control — snapshot and restore datasets
    • Quality scoring — automatic coherence/diversity/noise detection
    • Augmentation — synonym swap, word dropout, case variation
    • Deduplication — remove duplicates across all datasets

  Higher quality data = better model performance!"""
    ),
    (
        "Step 7: RAG (Retrieval-Augmented Generation)",
        """  RAG supercharges your model with external knowledge!

  From the menu, choose 'rag'

  How it works:
    1. Ingest documents (PDF, Markdown, text files)
    2. Documents are chunked and indexed
    3. When chatting, relevant chunks are retrieved
    4. The model generates answers grounded in your documents

  Use cases:
    • Chat with your documentation
    • Q&A over research papers
    • Knowledge base assistant"""
    ),
    (
        "Step 8: Agent Framework",
        """  The Agent can use tools to accomplish complex tasks!

  From the menu, choose 'agent'

  Built-in tools:
    • Calculator — math expressions
    • Web Search — find information online
    • File Operations — read/list files
    • Wikipedia — lookup articles
    • Current Time — get the date/time

  The agent detects when to use tools from your natural language
  and chains multiple tool calls to solve problems.

  It also has memory — short-term (conversation) and long-term
  (persistent across sessions)."""
    ),
    (
        "Step 9: Image Generation",
        """  Generate and manage images with diffusion models!

  From the menu, choose option [6] 'image-gen'

  Additional image tools (choose 'image-tools' from menu):
    • Upscaling — enhance resolution
    • Variations — generate N variations of an image
    • Color transfer — apply color palette from reference
    • LoRA merge — combine style weights

  Tag management (choose 'tag-mgr' from menu):
    • Analyze tag frequency in your dataset
    • Tag hierarchy/ontology
    • Negative prompt suggestions"""
    ),
    (
        "Step 10: Plugins & Configuration",
        """  Extend the suite with plugins!

  Plugin System (choose 'plugins' from menu):
    • Drop plugin folders into plugins/
    • Each plugin has a manifest.json
    • Plugins can add menu entries and commands
    • Lifecycle hooks: on_train_start, on_generate_complete, etc.

  Configuration (choose 'config-mgr' from menu):
    • Named profiles: fast, quality, gpu
    • Environment variable overrides (AI_MODEL_*)
    • Config validation against schema

  Model Packaging (choose 'packager' from menu):
    • Export models as .tar.gz archives
    • Share trained models between machines
    • Import with automatic registration"""
    ),
    (
        "Step 11: Evaluation & Monitoring",
        """  Track your model's performance over time!

  Evaluation Suite (choose 'eval' from menu):
    • Perplexity, BLEU, ROUGE-L, WER metrics
    • A/B model comparison
    • Evaluation history logging

  Training Dashboard (choose 'dashboard' from menu):
    • Live loss curves in your browser
    • Learning rate schedule visualization
    • Epoch-by-epoch metrics

  Curriculum Learning (choose 'curriculum' from menu):
    • Progressive difficulty scheduling
    • Domain mixing ratios"""
    ),
    (
        "Step 12: Local Model Tools & Tokenizer Verification",
        """  Manage all your locally stored models from one place!

  Local Model Manager (choose 'local-models' from menu):
    • Lists all HuggingFace cached, registered, and checkpoint models
    • Prefixed IDs: H1 (HuggingFace), R1 (Registered), C1 (Checkpoint)
    • Commands: load, uninstall, verify, verify-all, uninstall-all-hf
    • Reclaims disk space by removing unused cached models

  Tokenizer Verification (choose 'verify-tokens' from menu):
    • Checks special token IDs (PAD, UNK, BOS, EOS, SEP)
    • Encode/decode roundtrip validation
    • Conversation encoding length checks
    • Token2ID ↔ ID2Token consistency
    • Runs automatically at startup for early error detection"""
    ),
    (
        "What's Next?",
        """  You've completed the tutorial! Here are some next steps:

  Beginner:
    1. Run 'prompt-train' to create your first model
    2. Chat with it using 'chat'
    3. Try the 'explore' command to browse models

  Intermediate:
    4. Use 'auto-train' to build a dataset from Wikipedia
    5. Set up RAG with your own documents
    6. Try the Agent framework

  Advanced:
    7. Create a plugin to extend functionality
    8. Set up the training dashboard
    9. Use the OpenAI-compatible API for tool integration
   10. Deploy with Docker (see Dockerfile)

  Run 'python run.py' to return to the main menu.
  Happy building! 🚀"""
    ),
]


def run_tutorial():
    """Run the interactive tutorial."""
    _clear()
    print("\n" + "=" * 55)
    print("       AI Model Suite — Interactive Tutorial")
    print("=" * 55)

    for i, (title, content) in enumerate(STEPS):
        print(f"\n  ─── {title} {'─' * max(1, 45 - len(title))}")
        print(f"  ({i + 1}/{len(STEPS)})")
        print()
        print(content)

        if i < len(STEPS) - 1:
            try:
                _pause()
            except KeyboardInterrupt:
                print("\n\n  Tutorial ended early. Run 'tutorial' anytime to resume!")
                return
        else:
            print()

    print("  " + "=" * 55)
    print("  Tutorial complete!")
    print("  " + "=" * 55)


if __name__ == '__main__':
    try:
        run_tutorial()
    except KeyboardInterrupt:
        print("\n\n  Tutorial ended.")
