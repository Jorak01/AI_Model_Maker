"""Model factory - create and load different baseline models."""

import os
import torch
import torch.nn as nn
from typing import Optional, Tuple

# Available baseline models and their HuggingFace identifiers
PRETRAINED_MODELS = {
    "gpt2": "gpt2",
    "distilgpt2": "distilgpt2",
    "gpt2-medium": "gpt2-medium",
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-410m": "EleutherAI/pythia-410m",
}

MODEL_INFO = {
    "custom": {"params": "~2M", "desc": "Custom transformer (train from scratch)", "pretrained": False},
    "gpt2": {"params": "124M", "desc": "OpenAI GPT-2 Small", "pretrained": True},
    "distilgpt2": {"params": "82M", "desc": "Distilled GPT-2 (smaller, faster)", "pretrained": True},
    "gpt2-medium": {"params": "355M", "desc": "OpenAI GPT-2 Medium", "pretrained": True},
    "pythia-160m": {"params": "160M", "desc": "EleutherAI Pythia 160M", "pretrained": True},
    "pythia-410m": {"params": "410M", "desc": "EleutherAI Pythia 410M", "pretrained": True},
}

PIPELINE_INFO = {
    "scratch": {"desc": "Train custom model from scratch (no pretrained weights)"},
    "finetune": {"desc": "Fine-tune ALL layers of a pretrained model"},
    "freeze": {"desc": "Freeze base layers, only train top N layers"},
}


def list_models():
    """Print available baseline models."""
    print("\n  Available Baseline Models:")
    print("  " + "-" * 60)
    for name, info in MODEL_INFO.items():
        tag = "[pretrained]" if info["pretrained"] else "[from scratch]"
        print(f"  {name:<16} {info['params']:>6} params  {tag:<15} {info['desc']}")
    print()


def list_pipelines():
    """Print available training pipelines."""
    print("\n  Available Training Pipelines:")
    print("  " + "-" * 60)
    for name, info in PIPELINE_INFO.items():
        print(f"  {name:<12} {info['desc']}")
    print()
    print("  Pipeline Compatibility:")
    print("    scratch   → custom model only")
    print("    finetune  → any pretrained model (gpt2, distilgpt2, etc.)")
    print("    freeze    → any pretrained model (faster training)")
    print()


class PretrainedWrapper(nn.Module):
    """Wrapper around HuggingFace pretrained models for unified interface."""

    def __init__(self, model_name: str, max_seq_length: int = 128, pad_token_id: int = 0):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoConfig

        self.model_name = model_name
        self.hf_name = PRETRAINED_MODELS[model_name]
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id

        print(f"Loading pretrained model: {self.hf_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(self.hf_name)
        cfg = self.model.config
        self.vocab_size = cfg.vocab_size
        self.embedding_dim = cfg.n_embd if hasattr(cfg, 'n_embd') else cfg.hidden_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids)
        return outputs.logits

    def generate(self, input_ids: torch.Tensor, max_length: int = 100,
                 temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9,
                 repetition_penalty: float = 1.2, eos_token_id: Optional[int] = None) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            output = self.model.generate(
                input_ids, max_new_tokens=max_length, temperature=temperature,
                top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                eos_token_id=eos_token_id, do_sample=True, pad_token_id=self.pad_token_id
            )
        return output

    def freeze_layers(self, num_layers: int = 4):
        """Freeze the bottom N transformer layers."""
        # Freeze embeddings
        for param in self.model.get_input_embeddings().parameters():
            param.requires_grad = False

        # Freeze transformer layers
        layers = None
        if hasattr(self.model, 'transformer'):  # GPT-2 style
            layers = self.model.transformer.h
        elif hasattr(self.model, 'gpt_neox'):  # Pythia style
            layers = self.model.gpt_neox.layers

        if layers is not None:
            for i, layer in enumerate(layers):
                if i < num_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

        frozen = sum(1 for p in self.parameters() if not p.requires_grad)
        total = sum(1 for p in self.parameters())
        print(f"Frozen {frozen}/{total} parameter groups ({num_layers} layers)")

    def save(self, path: str):
        torch.save({
            'state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'max_seq_length': self.max_seq_length,
            'pad_token_id': self.pad_token_id,
            'type': 'pretrained'
        }, path)

    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'PretrainedWrapper':
        ckpt = torch.load(path, map_location=device)
        wrapper = cls(ckpt['model_name'], ckpt.get('max_seq_length', 128),
                      ckpt.get('pad_token_id', 0))
        wrapper.model.load_state_dict(ckpt['state_dict'])
        return wrapper.to(device)


def create_model(config: dict) -> nn.Module:
    """Create a model based on config settings."""
    base = config['model']['base_model']
    pipeline = config['training']['pipeline']
    max_len = config['model']['max_seq_length']

    if base == "custom":
        if pipeline != "scratch":
            print(f"Warning: Custom model only supports 'scratch' pipeline. Switching from '{pipeline}'.")
        from models.model import ConversationalModel
        return ConversationalModel(
            vocab_size=config['model']['vocab_size'],
            embedding_dim=config['model']['embedding_dim'],
            hidden_dim=config['model']['hidden_dim'],
            num_layers=config['model']['num_layers'],
            num_heads=config['model']['num_heads'],
            max_seq_length=max_len,
            dropout=config['model']['dropout'],
            pad_token_id=0
        )

    if base not in PRETRAINED_MODELS:
        raise ValueError(f"Unknown model: {base}. Options: {list(MODEL_INFO.keys())}")

    if pipeline == "scratch":
        print(f"Warning: Pretrained model '{base}' should use 'finetune' or 'freeze'. Switching to 'finetune'.")

    wrapper = PretrainedWrapper(base, max_len)

    if pipeline == "freeze":
        freeze_n = config['training'].get('freeze_layers', 4)
        wrapper.freeze_layers(freeze_n)

    return wrapper


def load_model(path: str, device: str = 'cpu') -> nn.Module:
    """Load any model from checkpoint (auto-detects type)."""
    ckpt = torch.load(path, map_location=device)

    if ckpt.get('type') == 'pretrained':
        return PretrainedWrapper.load(path, device)
    else:
        from models.model import ConversationalModel
        return ConversationalModel.load(path, device)
