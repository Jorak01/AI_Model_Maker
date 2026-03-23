"""Tests for data loading utilities."""

import os
import sys
import json
import tempfile
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.tokenizer import Tokenizer
from utils.data_loader import ConversationDataset, create_data_loaders, load_and_prepare_data


@pytest.fixture
def sample_data_file():
    """Create a temporary data file."""
    data = [
        {"prompt": "Hello", "response": "Hi there"},
        {"prompt": "How are you", "response": "I am fine"},
        {"prompt": "What is AI", "response": "Artificial intelligence"},
    ]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
        json.dump(data, f)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def tokenizer():
    """Create a tokenizer with vocab."""
    tok = Tokenizer(vocab_size=500, method="word")
    tok.build_vocab(["hello hi there how are you what is ai artificial intelligence fine"])
    return tok


class TestConversationDataset:
    def test_dataset_length(self, sample_data_file, tokenizer):
        ds = ConversationDataset(sample_data_file, tokenizer, max_length=32)
        assert len(ds) == 3

    def test_dataset_getitem(self, sample_data_file, tokenizer):
        ds = ConversationDataset(sample_data_file, tokenizer, max_length=32)
        input_ids, target_ids = ds[0]
        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(target_ids, torch.Tensor)
        assert input_ids.shape == (32,)
        assert target_ids.shape == (32,)

    def test_dataset_all_items(self, sample_data_file, tokenizer):
        ds = ConversationDataset(sample_data_file, tokenizer, max_length=32)
        for i in range(len(ds)):
            inp, tgt = ds[i]
            assert inp.shape == (32,)
            assert tgt.shape == (32,)


class TestCreateDataLoaders:
    def test_loaders_created(self, sample_data_file, tokenizer):
        train_loader, test_loader = create_data_loaders(
            sample_data_file, sample_data_file, tokenizer,
            batch_size=2, max_length=32
        )
        assert len(train_loader) > 0
        assert len(test_loader) > 0

    def test_loader_batch_shape(self, sample_data_file, tokenizer):
        train_loader, _ = create_data_loaders(
            sample_data_file, sample_data_file, tokenizer,
            batch_size=2, max_length=32
        )
        for input_ids, target_ids in train_loader:
            assert input_ids.shape[1] == 32
            assert target_ids.shape[1] == 32
            break  # Just check first batch


class TestLoadAndPrepareData:
    def test_extracts_texts(self, sample_data_file):
        texts = load_and_prepare_data(sample_data_file)
        assert len(texts) == 6  # 3 prompts + 3 responses
        assert "Hello" in texts
        assert "Hi there" in texts

    def test_returns_strings(self, sample_data_file):
        texts = load_and_prepare_data(sample_data_file)
        assert all(isinstance(t, str) for t in texts)
