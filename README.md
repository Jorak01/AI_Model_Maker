# AI Model Suite

A full-featured AI toolkit: custom transformer training, external API integration (OpenAI, Anthropic, Google, DeepSeek, Ollama), automated data collection, RAG, agent framework, image generation, evaluation, and more — all from a single interactive runtime.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Setup & Configuration](#setup--configuration)
- [Interactive Runtime](#interactive-runtime)
- [Training](#training)
  - [Manual Training](#manual-training)
  - [Prompt Training](#prompt-training)
  - [Auto Training](#auto-training)
  - [Pipelines Explained](#pipelines-explained)
  - [Curriculum Learning](#curriculum-learning)
- [Chat & Generation](#chat--generation)
  - [Local Chat](#local-chat)
  - [External API Chat](#external-api-chat)
  - [Agent (Tool Use)](#agent-tool-use)
  - [RAG (Document Chat)](#rag-document-chat)
  - [Image Generation](#image-generation)
- [Model Management](#model-management)
  - [Model Registry](#model-registry)
  - [Model Explorer](#model-explorer)
  - [Model Packager](#model-packager)
- [APIs & Servers](#apis--servers)
  - [REST API](#rest-api)
  - [OpenAI-Compatible API](#openai-compatible-api)
  - [Web UI](#web-ui)
- [Data & Evaluation](#data--evaluation)
  - [Dataset Manager](#dataset-manager)
  - [Evaluation Suite](#evaluation-suite)
  - [Training Dashboard](#training-dashboard)
- [Image Tools](#image-tools)
- [Plugins](#plugins)
- [Configuration Reference](#configuration-reference)
- [Docker](#docker)
- [Project Structure](#project-structure)
- [Running Tests](#running-tests)

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch (API tokens are configured on first run)
python run.py
```

---

## Setup & Configuration

The main configuration files live in the project root. On first launch, `run.py` will automatically prompt you to set up API tokens interactively.

| File | Purpose |
|------|---------|
| `config.yaml` | Model, training, API, and generation settings |
| `.env` | API keys, tokens, and environment variables |
| `docker-compose.yml` | Docker Compose orchestration |
| `pytest.ini` | Test configuration |

> **Backup templates** are available in `config/` if you ever need to recreate a file from scratch.

**API keys** can be set in `.env`, `config.yaml`, or as environment variables:

```bash
# Environment variables (any OS)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...
DEEPSEEK_API_KEY=sk-...
```

The `.gitignore` ensures `config.yaml`, `.env`, `docker-compose.yml`, and `pytest.ini` are never committed (they contain secrets). Backup templates in `config/` are tracked by git.

---

## Interactive Runtime

Run `python run.py` to launch the interactive menu with **34 commands** across 8 categories:

```
  ─── Training & Data ─────────────────────────
   1  train            Train the model
   2  prompt-train     Create model from prompts
   3  auto-train       Auto-train from public data
   4  auto-image       Auto-collect image tags
   5  dataset-mgr      Dataset management hub
   6  curriculum       Curriculum learning

  ─── Chat & Generation ────────────────────────
   7  chat             Interactive chat (local)
   8  image-gen        Image generation & training
   9  external         Chat via external API
  10  rag              RAG document chat
  11  agent            Agent with tool use

  ─── Model Management ────────────────────────
  12  load             Load any model by number
  13  explore          Browse all models
  14  registry         List registered models
  15  load-model       Chat with registered model
  16  packager         Export/import model archives

  ─── Evaluation & Monitoring ─────────────────
  17  eval             Evaluation suite (BLEU, etc)
  18  dashboard        Training dashboard (web)

  ─── Image Tools ────────────────────────────
  19  tag-mgr          Tag frequency & ontology
  20  image-tools      Upscale, variations, color

  ─── API & Servers ──────────────────────────
  21  api              Start REST API server
  22  compat-api       OpenAI-compatible API
  23  web-ui           Browser-based interface
  24  providers        List API providers

  ─── Infrastructure ─────────────────────────
  25  plugins          Plugin manager
  26  config-mgr       Config profiles & validation

  ─── Info & System ───────────────────────────
  27  models           Available base models
  28  pipelines        Training pipelines
  29  model-families   Models grouped by family
  30  config           Show configuration
  31  status           Model/checkpoint status
  32  refresh-models   Update model lists
  33  tutorial         Interactive tutorial
  34  test             Run all tests

   0  stop / quit      Exit
```

Any command can be run directly from the command line:

```bash
python run.py train          # Start training
python run.py chat           # Chat with trained model
python run.py auto-train     # Auto-train from public data
python run.py external       # Chat via external API
python run.py agent          # Launch tool-use agent
python run.py rag            # RAG document chat
python run.py api            # Start REST API server
python run.py explore        # Browse all models
python run.py eval           # Run evaluation suite
python run.py dashboard      # Launch training dashboard
```

---

## Training

### Manual Training

Prepare a JSON dataset of prompt/response pairs in `data/train.json`:

```json
[
  {"prompt": "Hello!", "response": "Hi! How can I help you today?"},
  {"prompt": "What is AI?", "response": "AI is the simulation of human intelligence by machines."}
]
```

Configure `config.yaml`, then train:

```bash
python run.py train
```

This builds a tokenizer, creates the model, trains for the configured epochs, and saves checkpoints to `checkpoints/`. The best model is saved as `checkpoints/best_model.pt`.

### Prompt Training

Create a named model interactively — no file editing needed:

```bash
python run.py prompt-train
```

1. Name your model (e.g., `trivia-bot`)
2. Describe its purpose
3. Enter prompt/response pairs one by one
4. Choose a base model (custom, GPT-2, DistilGPT-2)
5. Train — the model is automatically registered

### Auto Training

**The auto-trainer searches the internet for public domain data, builds a dataset, and trains a model — fully automated:**

```bash
python run.py auto-train
```

Interactive flow:
1. Name the model and enter topics (e.g., `quantum physics, relativity`)
2. Choose data sources: Wikipedia, web search, StackExchange
3. Optionally add specific URLs
4. Choose base model and training epochs
5. The system collects data, builds a dataset, trains, and registers the model

Programmatic usage:

```python
from training.auto_trainer import auto_train

auto_train(
    topics=["quantum computing", "neural networks"],
    model_name="science-bot",
    sources=["wikipedia", "web"],
    max_pairs_per_topic=150,
    base_model="custom",
    epochs=10,
)
```

Or collect data without training:

```python
from training.auto_trainer import auto_collect

path = auto_collect(
    topics=["machine learning", "deep learning"],
    output_name="ml-dataset",
)
```

### Pipelines Explained

| Pipeline | What It Does | When to Use |
|----------|-------------|-------------|
| **`scratch`** | Train from random weights | Full control, specialized domains |
| **`finetune`** | Train ALL layers of a pretrained model | Good data (100+ pairs), best quality |
| **`freeze`** | Only train top layers of a pretrained model | Limited data (10-50 pairs), avoid overfitting |

```yaml
# config.yaml examples
model:
  base_model: "custom"      # or "gpt2", "distilgpt2", "gpt2-medium", "pythia-160m"
training:
  pipeline: "scratch"        # or "finetune", "freeze"
  freeze_layers: 4           # for freeze pipeline only
```

### Curriculum Learning

Progressive training that starts with easy examples and increases difficulty:

```bash
python run.py curriculum
```

Features:
- Automatic difficulty estimation based on text complexity
- Progressive data introduction (easy → hard)
- Domain detection and mixing ratios
- Schedule visualization

---

## Chat & Generation

### Local Chat

Chat with your trained model:

```bash
python run.py chat           # Default trained model from checkpoints/
python run.py load-model     # Choose a registered model by name
```

### External API Chat

Chat with OpenAI, Anthropic, Google, DeepSeek, Ollama, or any OpenAI-compatible endpoint:

```bash
python run.py external       # Interactive provider/model selection
python run.py providers      # Check which providers are configured
python run.py refresh-models # Update model lists from provider APIs
```

Supported providers and their latest models:

| Provider | Models | Auth |
|----------|--------|------|
| **OpenAI** | gpt-4o, gpt-4.1, o3, o4-mini, gpt-4.5-preview | `OPENAI_API_KEY` |
| **Anthropic** | claude-sonnet-4, claude-3.7-sonnet, claude-3.5-sonnet | `ANTHROPIC_API_KEY` |
| **Google** | gemini-2.5-pro, gemini-2.5-flash, gemini-2.0-flash | `GOOGLE_API_KEY` |
| **DeepSeek** | deepseek-chat, deepseek-reasoner | `DEEPSEEK_API_KEY` |
| **Ollama** | llama3.3, mistral, phi4, qwen3, deepseek-r1 (local) | No key needed |
| **Custom** | Any OpenAI-compatible endpoint | `CUSTOM_API_KEY` |

Programmatic usage:

```python
from services.external_api import api_call, ExternalAPIClient

# Single call
response = api_call("openai", "gpt-4o", "What is machine learning?")

# Multi-turn conversation
client = ExternalAPIClient("google", model="gemini-2.5-flash")
r1 = client.chat("What is Python?")
r2 = client.chat("Can you give me an example?")  # Remembers context
client.clear_history()
```

### Agent (Tool Use)

An AI agent that can use tools, search the web, read files, do math, and remember facts:

```bash
python run.py agent
```

Built-in tools:
- **Calculator** — evaluate math expressions
- **Web search** — search DuckDuckGo
- **Wikipedia** — look up any topic
- **File reader** — read file contents
- **File listing** — list directory contents
- **Current time** — get date/time

The agent automatically detects when a tool is needed. It also has short-term (conversation) and long-term (persistent) memory:

```
You: What is 245 * 18?
Agent: [Used calculator] 4410

You: Search for latest news on quantum computing
Agent: [Used web_search] ...

You: Remember that my project deadline is March 30
Agent: Remembered: my project deadline is March 30
```

### RAG (Document Chat)

Retrieval-Augmented Generation — ingest documents, then chat with context and citations:

```bash
python run.py rag
```

Features:
- Ingest text, markdown, Python, JSON, CSV, and PDF files
- TF-IDF based vector search (no external dependencies)
- Automatic context retrieval and prompt augmentation
- Source citations in responses

```python
from services.rag import RAGPipeline

rag = RAGPipeline()
rag.ingest_file("docs/manual.md")
rag.ingest_directory("src/", extensions=[".py", ".md"])

response, sources = rag.chat_with_context("How does the tokenizer work?")
```

### Image Generation

Tag-based image generation and training with Stable Diffusion models:

```bash
python run.py image-gen
```

Features:
- Generate images from text prompts
- Tag normalization (Danbooru, e621, Derpibooru formats)
- Tag-based dataset creation
- Fine-tune image models on custom tag datasets

---

## Model Management

### Model Registry

Every model created with `prompt-train` or `auto-train` is automatically registered:

```bash
python run.py registry       # List all registered models
python run.py load-model     # Load and chat with a registered model
```

Models are stored in `trained_models/<name>/` with weights, tokenizer, and training data.

### Model Explorer

A unified hub to access any model — local checkpoints, registered models, or external APIs:

```bash
python run.py explore
```

### Model Packager

Export models as portable `.tar.gz` archives, or import them:

```bash
python run.py packager
```

Archives include the model weights, tokenizer, config, training data, and a manifest.

---

## APIs & Servers

### REST API

```bash
python run.py api            # Starts on http://localhost:8000
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/chat` | POST | Send message, get response |
| `/generate` | POST | Generate with custom params (temperature, top_k, etc.) |
| `/config` | GET | View model configuration |

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'
```

### OpenAI-Compatible API

Drop-in replacement for OpenAI's API — use with any OpenAI SDK client:

```bash
python run.py compat-api     # Starts on http://localhost:8001
```

Endpoints: `/v1/chat/completions`, `/v1/images/generations`, `/v1/models`

Routes requests to local models or configured external APIs transparently.

### Web UI

Browser-based chat interface:

```bash
python run.py web-ui         # Starts on http://localhost:7860
```

---

## Data & Evaluation

### Dataset Manager

Comprehensive dataset tools — versioning, quality scoring, augmentation, and deduplication:

```bash
python run.py dataset-mgr
```

Features:
- **Versioning** — snapshot datasets, list versions, restore any version
- **Quality scoring** — score prompt/response pairs on length, diversity, coherence
- **Augmentation** — synonym swap, word dropout, word swap, case changes
- **Deduplication** — remove duplicates with a global index

### Evaluation Suite

Measure model quality with standard metrics:

```bash
python run.py eval
```

Metrics: BLEU, ROUGE-1/ROUGE-L, perplexity, word error rate. Supports A/B model comparison and evaluation history logging.

### Training Dashboard

Live web dashboard showing training loss curves:

```bash
python run.py dashboard      # Starts on http://localhost:8501
```

Tracks training and validation loss across runs with a browser-based UI.

---

## Image Tools

Advanced image processing utilities:

```bash
python run.py image-tools    # Upscale, variations, color transfer
python run.py tag-mgr        # Tag frequency analysis & ontology
python run.py auto-image     # Auto-collect image tags from Danbooru/web
```

Features:
- Image upscaling (1x–8x)
- Style variations generation
- Color transfer between images
- LoRA weight merging
- ONNX export
- Tag frequency analysis, rare/overrepresented tag detection
- Tag categorization hierarchy (subject, style, quality, etc.)
- Negative prompt suggestions

---

## Plugins

Extend functionality with plugins:

```bash
python run.py plugins
```

Create a plugin by adding a `plugins/<name>/manifest.json` with name, description, and entry point.

---

## Configuration Reference

All settings in `config.yaml`:

| Section | Key Settings |
|---------|-------------|
| `model` | `base_model`, `vocab_size`, `embedding_dim`, `hidden_dim`, `num_layers`, `num_heads`, `max_seq_length`, `dropout` |
| `training` | `pipeline`, `batch_size`, `learning_rate`, `num_epochs`, `gradient_clip`, `warmup_steps`, `weight_decay`, `freeze_layers` |
| `data` | `train_path`, `test_path` |
| `checkpoint` | `save_dir`, `save_every`, `keep_last` |
| `generation` | `max_length`, `temperature`, `top_k`, `top_p`, `repetition_penalty` |
| `performance` | `mixed_precision`, `compile_model`, `num_workers` |
| `api` | `host`, `port`, `debug` |
| `external_api` | `openai`, `anthropic`, `google`, `deepseek`, `ollama`, `custom` — each with `api_key`, `base_url`, `default_model` |
| `auto_training` | `sources`, `max_pairs_per_topic`, `epochs` |
| `device` | `use_cuda`, `cuda_device` |

Baseline models:

| Model | Params | Type |
|-------|--------|------|
| `custom` | ~2M | From scratch |
| `gpt2` | 124M | Pretrained |
| `distilgpt2` | 82M | Pretrained |
| `gpt2-medium` | 355M | Pretrained |
| `pythia-160m` | 160M | Pretrained |
| `pythia-410m` | 410M | Pretrained |

---

## Docker

```bash
# Build
docker build -t ai-model-suite .

# Run (API server)
docker run -p 8000:8000 ai-model-suite

# Run with GPU
docker run --gpus all -p 8000:8000 ai-model-suite

# Exposed ports: 8000 (API), 5555 (Dashboard), 7860 (Web UI)
```

---

## Project Structure

```
AI_Model/
├── run.py                      # Interactive runtime (main entry point)
├── Dockerfile                  # Docker container definition
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── config/                     # Backup config templates (for recreating files)
│   ├── .env.template
│   ├── config.yaml.template
│   ├── docker-compose.yml.template
│   └── pytest.ini.template
│
├── services/                   # APIs, web interfaces, chat, integrations
│   ├── api.py                  #   REST API server (Flask)
│   ├── api_compat.py           #   OpenAI-compatible API endpoint
│   ├── chat.py                 #   Interactive chat with local models
│   ├── external_api.py         #   External API client (5+ providers)
│   ├── web_ui.py               #   Browser-based web interface
│   ├── agent.py                #   Agent with tool use & memory
│   └── rag.py                  #   RAG document chat with citations
│
├── training/                   # Training pipelines & data collection
│   ├── train.py                #   Main training script
│   ├── prompt_trainer.py       #   Interactive prompt-based model creator
│   ├── auto_trainer.py         #   Auto-train from public domain data
│   ├── image_auto_trainer.py   #   Auto-collect image tags & datasets
│   └── image_gen.py            #   Image generation & tag-based training
│
├── models/                     # Model architectures, registry, loading
│   ├── model.py                #   Custom transformer architecture
│   ├── model_factory.py        #   Model factory (creates/loads any baseline)
│   ├── tokenizer.py            #   Text tokenizer (word/char)
│   ├── registry.py             #   Named model registry (save/load/delete)
│   └── loader.py               #   Universal model loader
│
├── utils/                      # Utilities and tools
│   ├── data_loader.py          #   Dataset and data loader creation
│   ├── trainer.py              #   Training loop with checkpointing
│   ├── web_collector.py        #   Public domain web data collector
│   ├── config_manager.py       #   Config profiles, env overrides, validation
│   ├── dataset_manager.py      #   Dataset versioning, quality, augmentation
│   ├── curriculum.py           #   Curriculum learning & domain mixing
│   ├── eval_suite.py           #   Evaluation (BLEU, ROUGE, perplexity)
│   ├── training_dashboard.py   #   Live training dashboard (web UI)
│   ├── tag_manager.py          #   Tag frequency & ontology tools
│   ├── image_tools.py          #   Upscale, variations, color transfer
│   └── model_packager.py       #   Export/import model archives
│
├── plugins/                    # Plugin system
│   └── loader.py               #   Plugin discovery & management
│
├── docs/                       # Documentation & tutorials
│   └── tutorial.py             #   Interactive walkthrough
│
├── tests/                      # Test suite (700+ tests)
│   └── test_*.py               #   One test file per module
│
├── data/                       # Training data (auto-created)
├── checkpoints/                # Model checkpoints (auto-created)
└── trained_models/             # Registered named models (auto-created)
```

---

## Running Tests

```bash
python run.py test                    # From the interactive menu
python -m pytest tests/ -v            # Directly with pytest
python -m pytest tests/ -q --tb=short # Quick summary
```

The test suite includes **700+ tests** covering every module, all runtime commands, API endpoints, model operations, data pipelines, and integration points.
