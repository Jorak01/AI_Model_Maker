"""Tests for the model factory."""

import os
import sys
import tempfile
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_factory import (
    create_model, load_model, list_models, list_pipelines,
    MODEL_INFO, PIPELINE_INFO, PRETRAINED_MODELS
)
from models.model import ConversationalModel


class TestModelInfo:
    def test_model_info_has_entries(self):
        assert len(MODEL_INFO) >= 6
        assert "custom" in MODEL_INFO
        assert "gpt2" in MODEL_INFO

    def test_pipeline_info_has_entries(self):
        assert len(PIPELINE_INFO) == 3
        assert "scratch" in PIPELINE_INFO
        assert "finetune" in PIPELINE_INFO
        assert "freeze" in PIPELINE_INFO

    def test_pretrained_models_mapping(self):
        assert "gpt2" in PRETRAINED_MODELS
        assert "distilgpt2" in PRETRAINED_MODELS
        assert PRETRAINED_MODELS["gpt2"] == "gpt2"


class TestCreateModel:
    def test_create_custom_model(self):
        config = {
            'model': {
                'base_model': 'custom', 'vocab_size': 100, 'embedding_dim': 32,
                'hidden_dim': 64, 'num_layers': 2, 'num_heads': 4,
                'max_seq_length': 32, 'dropout': 0.1
            },
            'training': {'pipeline': 'scratch'}
        }
        model = create_model(config)
        assert isinstance(model, ConversationalModel)
        assert model.vocab_size == 100

    def test_create_custom_wrong_pipeline_warns(self, capsys):
        config = {
            'model': {
                'base_model': 'custom', 'vocab_size': 100, 'embedding_dim': 32,
                'hidden_dim': 64, 'num_layers': 2, 'num_heads': 4,
                'max_seq_length': 32, 'dropout': 0.1
            },
            'training': {'pipeline': 'finetune'}
        }
        model = create_model(config)
        captured = capsys.readouterr()
        assert "Warning" in captured.out

    def test_invalid_model_raises(self):
        config = {
            'model': {'base_model': 'nonexistent', 'max_seq_length': 32},
            'training': {'pipeline': 'finetune'}
        }
        with pytest.raises(ValueError):
            create_model(config)


class TestLoadModel:
    def test_load_custom_model(self):
        model = ConversationalModel(vocab_size=50, embedding_dim=16, hidden_dim=32,
                                     num_layers=1, num_heads=2, max_seq_length=16)
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name
        try:
            model.save(path)
            loaded = load_model(path)
            assert isinstance(loaded, ConversationalModel)
            assert loaded.vocab_size == 50
        finally:
            os.unlink(path)


class TestListFunctions:
    def test_list_models(self, capsys):
        list_models()
        captured = capsys.readouterr()
        assert "custom" in captured.out
        assert "gpt2" in captured.out

    def test_list_pipelines(self, capsys):
        list_pipelines()
        captured = capsys.readouterr()
        assert "scratch" in captured.out
        assert "finetune" in captured.out
        assert "freeze" in captured.out
