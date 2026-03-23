# Conversational AI Model

A transformer-based conversational AI with multiple baseline models, training pipelines, model registry, external API integration, and REST API.

---

## Table of Contents

- [Conversational AI Model](#conversational-ai-model)
  - [Table of Contents](#table-of-contents)
  - [Quick Start](#quick-start)
  - [How to Use](#how-to-use)
    - [Interactive Runtime](#interactive-runtime)
    - [Training a Model](#training-a-model)
      - [Step 1: Prepare Your Data](#step-1-prepare-your-data)
      - [Step 2: Configure Training](#step-2-configure-training)
      - [Step 3: Train](#step-3-train)
      - [Step 4: Chat with Your Model](#step-4-chat-with-your-model)
    - [Training Pipelines Explained](#training-pipelines-explained)
    - [Creating Custom Models with Prompt Training](#creating-custom-models-with-prompt-training)
    - [Chatting with Your Model](#chatting-with-your-model)
    - [Using Saved \& Registered Models](#using-saved--registered-models)
      - [List all saved models:](#list-all-saved-models)
      - [Load and chat with a saved model:](#load-and-chat-with-a-saved-model)
      - [Where models are stored:](#where-models-are-stored)
    - [Exploring All Models (Local, Saved, API)](#exploring-all-models-local-saved-api)
    - [External API Calls](#external-api-calls)
      - [Setup](#setup)
      - [Interactive Chat](#interactive-chat)
      - [Programmatic API Calls (Python)](#programmatic-api-calls-python)
      - [Check Provider Status](#check-provider-status)
    - [REST API Server](#rest-api-server)
      - [Endpoints](#endpoints)
      - [Examples](#examples)
  - [Baseline Models](#baseline-models)
  - [Configuration Reference](#configuration-reference)
  - [Project Structure](#project-structure)
  - [Data Format](#data-format)
  - [Running Tests](#running-tests)

---

## Quick Start

```bash
pip install -r requirements.txt
python run.py                # Launch interactive runtime menu
```

---

## How to Use

### Interactive Runtime

Run `python run.py` to launch the interactive menu. All features are accessible from here:

```
  Commands:
  --------------------------------------------------
  1   train          Train the model
  2   chat           Interactive chat (local model)
  3   api            Start REST API server
  4   models         List available base models
  5   pipelines      List training pipelines
  6   config         Show current configuration
  7   status         Check model/checkpoint status
  8   test           Run all tests
  --------------------------------------------------
  9   prompt-train   Create model from prompts
  10  registry       List registered models
  11  load-model     Chat with a registered model
  12  external       Chat via external API
  13  providers      List external API providers
  14  explore        Browse all models (local, saved, API)
  --------------------------------------------------
  0   stop/quit      Exit the runtime
```

You can also run any command directly from the command line:

```bash
python run.py train          # Start training immediately
python run.py chat           # Open chat with trained model
python run.py api            # Start the REST API server
python run.py explore        # Open the model explorer
python run.py prompt-train   # Create a custom model interactively
python run.py external       # Chat with an external API model
python run.py registry       # List all saved/registered models
python run.py status         # Check what's trained and available
```

---

### Training a Model

Training teaches the model to generate conversational responses from your data.

#### Step 1: Prepare Your Data

Training data is a JSON file of prompt/response pairs in `data/train.json`:

```json
[
  {"prompt": "Hello!", "response": "Hi! How can I help you today?"},
  {"prompt": "What is AI?", "response": "AI is the simulation of human intelligence by machines."},
  {"prompt": "Tell me a joke", "response": "Why do programmers prefer dark mode? Because light attracts bugs!"}
]
```

#### Step 2: Configure Training

Edit `config.yaml` to set your model and training options:

```yaml
model:
  base_model: "custom"       # Which model to use (see Baseline Models)
  vocab_size: 10000           # Vocabulary size (custom model only)
  embedding_dim: 256          # Embedding dimensions (custom model only)
  num_layers: 4               # Transformer layers (custom model only)
  max_seq_length: 128         # Max input length

training:
  pipeline: "scratch"         # How to train (see Pipelines)
  batch_size: 32              # Samples per batch
  learning_rate: 0.0001       # Learning rate
  num_epochs: 10              # Training iterations over full dataset
```

#### Step 3: Train

```bash
python run.py train
```

This will:
1. Build a tokenizer from your training data
2. Create the model based on your config
3. Train for the configured number of epochs
4. Save checkpoints to `checkpoints/` after each epoch
5. Save the best model as `checkpoints/best_model.pt`

You'll see training loss and validation loss printed each epoch. Lower loss = better model.

#### Step 4: Chat with Your Model

```bash
python run.py chat
```

The chat command automatically loads the best checkpoint and tokenizer.

---

### Training Pipelines Explained

Pipelines control **how** the model learns. Choose the right one based on your needs:

| Pipeline | What It Does | When to Use | Config |
|----------|-------------|-------------|--------|
| **`scratch`** | Trains a brand-new model from random weights | When you want full control with your own architecture | `base_model: "custom"` |
| **`finetune`** | Takes a pretrained model and trains ALL layers on your data | When you want a smart starting point and have enough data (100+ pairs) | `base_model: "gpt2"` (or any pretrained) |
| **`freeze`** | Takes a pretrained model but only trains the top layers; base layers stay frozen | When you have limited data (10-50 pairs) and want to avoid overfitting | `base_model: "gpt2"` + `freeze_layers: 4` |

**How `scratch` works:**
- Creates a custom transformer from scratch with random weights
- All parameters are trainable — the model learns everything from your data
- Needs more data and epochs to produce good results
- Best for specialized domains where pretrained knowledge would be irrelevant

**How `finetune` works:**
- Downloads a pretrained model (e.g., GPT-2 with 124M parameters already trained on internet text)
- Unlocks all layers so every weight gets updated on your data
- The model starts with general language knowledge and adapts it to your style
- Faster convergence than scratch; 3-5 epochs is often enough

**How `freeze` works:**
- Downloads a pretrained model just like `finetune`
- Freezes the bottom N layers (set by `freeze_layers` in config)
- Only the top layers and output head are updated during training
- Preserves the pretrained knowledge while adapting the output
- Ideal when you have very few training examples

**Example configurations:**

```yaml
# Train from scratch (custom architecture)
model:
  base_model: "custom"
training:
  pipeline: "scratch"
  num_epochs: 20

# Fine-tune GPT-2 (all layers)
model:
  base_model: "gpt2"
training:
  pipeline: "finetune"
  num_epochs: 5
  learning_rate: 0.00005

# Freeze-tune DistilGPT-2 (fast, small data)
model:
  base_model: "distilgpt2"
training:
  pipeline: "freeze"
  freeze_layers: 4
  num_epochs: 3
```

---

### Creating Custom Models with Prompt Training

The **Prompt Trainer** lets you create a named model interactively without editing files:

```bash
python run.py prompt-train
```

**What happens:**

1. **Name your model** — e.g., "customer-support-bot"
2. **Describe its intent** — e.g., "Answer customer support questions"
3. **Enter prompt/response pairs** — type them in one by one
4. **Choose a base model** — custom (scratch), GPT-2, or DistilGPT-2
5. **Train** — the system trains the model automatically
6. **Register** — the model is saved to the registry with its name

**Interactive example:**

```
  What should this model be called?
  Name: trivia-bot

  What is the main intent/purpose of this model?
  Intent: Answer trivia questions about science

  [1] Enter a PROMPT (or 'done'):
  > What is the speed of light?
  Enter the RESPONSE:
  > The speed of light in a vacuum is approximately 299,792,458 meters per second.

  [2] Enter a PROMPT (or 'done'):
  > What is photosynthesis?
  Enter the RESPONSE:
  > Photosynthesis is the process by which plants convert sunlight into energy.

  [3] Enter a PROMPT (or 'done'):
  > done

  Choose base model:
  1. custom (train from scratch)
  2. gpt2 (fine-tune GPT-2)
  3. distilgpt2 (fine-tune DistilGPT-2)
  [1/2/3, default=1]: 1

  Training epochs? (default=10)
  Epochs: 5

  Building tokenizer...
  Creating data loaders...
  Starting training...
  Epoch 1/5  Loss: 2.4531  Val Loss: 2.3890
  ...
  Model 'trivia-bot' trained and registered!
```

Your model is now saved and accessible anytime via `load-model` or `explore`.

---

### Chatting with Your Model

There are three ways to chat:

**1. Chat with the default trained model** (from `checkpoints/`):
```bash
python run.py chat
```

**2. Chat with a registered/saved model** (from the model registry):
```bash
python run.py load-model
# Then type the model name when prompted, e.g. "trivia-bot"
```

**3. Chat with an external API model** (OpenAI, Anthropic, Ollama):
```bash
python run.py external
```

---

### Using Saved & Registered Models

Every model you create with `prompt-train` is automatically registered. You can also manually register models.

#### List all saved models:
```bash
python run.py registry
```
Output:
```
  Registered Models (2):
  ----------------------------------------------------------------------
  Name                      Base           Pipeline   Intent
  ----------------------------------------------------------------------
  trivia-bot                custom         scratch    Answer trivia questions
  support-agent             gpt2           finetune   Customer support responses
```

#### Load and chat with a saved model:
```bash
python run.py load-model
```
You'll be shown the registry and asked to type a model name. The system loads the model and tokenizer automatically and opens an interactive chat session.

#### Where models are stored:

| Location | What's There | How to Access |
|----------|-------------|---------------|
| `checkpoints/` | Default trained model (from `python run.py train`) | `python run.py chat` |
| `trained_models/<name>/` | Named registered models (from prompt-train) | `python run.py load-model` |
| `trained_models/registry.json` | Registry index of all named models | `python run.py registry` |

Each registered model directory contains:
- `model.pt` — The trained model weights
- `tokenizer.pkl` — The tokenizer used during training
- `data/` — Copy of the training data used (if available)

---

### Exploring All Models (Local, Saved, API)

The **explore** command (option 14) is a unified hub for accessing any model — whether it's a local checkpoint, a registered custom model, or an external API model:

```bash
python run.py explore
```

```
  ╔══════════════════════════════════════════════╗
  ║           Model Explorer                     ║
  ╠══════════════════════════════════════════════╣
  ║  1. Local Model   (trained checkpoint)       ║
  ║  2. Saved Models  (registered/named models)  ║
  ║  3. API Models    (OpenAI, Anthropic, etc.)  ║
  ║  0. Back to menu                             ║
  ╚══════════════════════════════════════════════╝
```

- **Option 1** — Loads your default trained model from `checkpoints/` and opens chat
- **Option 2** — Shows all registered models, lets you pick one by name, loads it, and opens chat
- **Option 3** — Connects to an external API (OpenAI, Anthropic, Ollama, or custom) for chat. If the chosen provider has no API key configured, it automatically tries other providers. If none are configured, it returns to the menu.

This is the easiest way to access everything in one place.

---

### External API Calls

You can chat with external AI models (OpenAI GPT-4, Anthropic Claude, local Ollama models, or any OpenAI-compatible API).

#### Setup

Set API keys in `config.yaml` or as environment variables:

```yaml
# In config.yaml
external_api:
  openai:
    api_key: "sk-your-openai-key-here"
    default_model: "gpt-4o-mini"
  anthropic:
    api_key: "sk-ant-your-anthropic-key-here"
    default_model: "claude-3-haiku-20240307"
  ollama:
    base_url: "http://localhost:11434"
    default_model: "llama3"
```

Or via environment variables:
```bash
set OPENAI_API_KEY=sk-your-key-here          # Windows
export OPENAI_API_KEY=sk-your-key-here       # Linux/Mac
export ANTHROPIC_API_KEY=sk-ant-your-key
```

#### Interactive Chat

```bash
python run.py external
```

Choose a provider (or type `auto` to automatically find the first configured one). If a provider isn't configured, the system skips it and checks the next one. If no providers are configured at all, it returns to the menu.

#### Programmatic API Calls (Python)

```python
from external_api import api_call, ExternalAPIClient

# Single call
response = api_call("openai", "gpt-4o-mini", "What is machine learning?")
print(response)

# Multi-turn conversation
client = ExternalAPIClient("openai", model="gpt-4o-mini")
r1 = client.chat("What is Python?")
r2 = client.chat("Can you give me an example?")  # Remembers context
client.clear_history()  # Reset conversation

# With system prompt
response = api_call("anthropic", "claude-3-haiku-20240307",
                    "Explain quantum computing",
                    system_prompt="You are a physics professor")
```

#### Check Provider Status

```bash
python run.py providers
```

Output:
```
  External API Providers:
  ------------------------------------------------------------
  openai       OpenAI (GPT-4, GPT-3.5)                [configured]
  anthropic    Anthropic (Claude)                      [no key set]
  ollama       Ollama (Local)                          [not running]
  custom       Custom API Endpoint                     [no key set]
```

---

### REST API Server

Start the REST API to integrate with other applications:

```bash
python run.py api
```

The server runs at `http://localhost:8000` by default.

#### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check — returns status and whether a model is loaded |
| `/chat` | POST | Send a message, get a response |
| `/generate` | POST | Generate with custom parameters (temperature, top_k, etc.) |
| `/config` | GET | View the current model configuration |

#### Examples

**Health check:**
```bash
curl http://localhost:8000/health
```
```json
{"status": "ok", "model_loaded": true}
```

**Chat:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'
```
```json
{"response": "Hi! How can I help you today?"}
```

**Generate with custom parameters:**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me about AI",
    "temperature": 0.5,
    "top_k": 30,
    "top_p": 0.85,
    "max_length": 150
  }'
```

---

## Baseline Models

| Model | Params | Type | Description |
|-------|--------|------|-------------|
| `custom` | ~2M | From scratch | Custom transformer architecture |
| `gpt2` | 124M | Pretrained | OpenAI GPT-2 Small |
| `distilgpt2` | 82M | Pretrained | Distilled GPT-2 (faster, lighter) |
| `gpt2-medium` | 355M | Pretrained | OpenAI GPT-2 Medium |
| `pythia-160m` | 160M | Pretrained | EleutherAI Pythia 160M |
| `pythia-410m` | 410M | Pretrained | EleutherAI Pythia 410M |

---

## Configuration Reference

All settings are in `config.yaml`. Key sections:

| Section | Key Settings |
|---------|-------------|
| `model` | `base_model`, `vocab_size`, `embedding_dim`, `hidden_dim`, `num_layers`, `num_heads`, `max_seq_length`, `dropout` |
| `training` | `pipeline`, `batch_size`, `learning_rate`, `num_epochs`, `gradient_clip`, `warmup_steps`, `weight_decay`, `freeze_layers` |
| `data` | `train_path`, `test_path` |
| `checkpoint` | `save_dir`, `save_every`, `keep_last` |
| `generation` | `max_length`, `temperature`, `top_k`, `top_p`, `repetition_penalty` |
| `api` | `host`, `port`, `debug` |
| `external_api` | `openai`, `anthropic`, `ollama`, `custom` (each with `api_key`, `base_url`, `default_model`) |
| `device` | `use_cuda`, `cuda_device` |

---

## Project Structure

```
AI_Model/
├── run.py                  # Interactive runtime (main entry point)
├── train.py                # Training script
├── chat.py                 # Interactive chat with local models
├── api.py                  # REST API server (Flask)
├── model_registry.py       # Named model registry (save/load/delete)
├── prompt_trainer.py       # Interactive prompt-based model creator
├── external_api.py         # External API client (OpenAI, Anthropic, Ollama)
├── config.yaml             # All configuration settings
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── models/
│   ├── __init__.py         # Package exports
│   ├── model.py            # Custom transformer architecture
│   ├── model_factory.py    # Model factory (creates/loads any baseline)
│   └── tokenizer.py        # Text tokenizer (word/char)
├── utils/
│   ├── __init__.py         # Package exports
│   ├── data_loader.py      # Dataset and data loader creation
│   └── trainer.py          # Training loop with checkpointing
├── data/
│   ├── train.json          # Training data
│   ├── test.json           # Test/validation data
│   └── custom/             # Custom prompt-trained data (auto-created)
├── checkpoints/            # Model checkpoints (auto-created)
├── trained_models/         # Registered named models (auto-created)
│   ├── registry.json       # Model registry index
│   └── <model-name>/      # Individual model directories
└── tests/
    ├── __init__.py
    ├── test_model.py       # Model unit tests
    ├── test_tokenizer.py   # Tokenizer unit tests
    ├── test_data_loader.py # Data loader tests
    ├── test_factory.py     # Model factory tests
    ├── test_trainer.py     # Trainer tests
    ├── test_api.py         # API endpoint tests
    ├── test_registry.py    # Model registry tests
    ├── test_prompt_trainer.py # Prompt trainer tests
    ├── test_external_api.py   # External API tests
    └── test_runtime.py     # Comprehensive runtime tests (all modules)
```

---

## Data Format

Training and test data files are JSON arrays of prompt/response objects:

```json
[
  {"prompt": "Hello!", "response": "Hi! How can I help?"},
  {"prompt": "What is AI?", "response": "AI is the simulation of human intelligence by machines."}
]
```

---

## Running Tests

```bash
python run.py test              # From the interactive menu
python -m pytest tests/ -v      # Directly with pytest
```

The test suite includes 184+ tests covering every module, runtime operation, and integration point.
