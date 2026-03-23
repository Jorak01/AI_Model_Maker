"""Tests for the model registry."""

import os
import sys
import json
import shutil
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model_registry


@pytest.fixture(autouse=True)
def temp_registry(monkeypatch, tmp_path):
    """Use a temporary directory for the registry during tests."""
    reg_dir = str(tmp_path / "trained_models")
    reg_file = os.path.join(reg_dir, "registry.json")
    monkeypatch.setattr(model_registry, "REGISTRY_DIR", reg_dir)
    monkeypatch.setattr(model_registry, "REGISTRY_FILE", reg_file)
    return reg_dir


@pytest.fixture
def fake_checkpoint(tmp_path):
    """Create a fake checkpoint directory with model files."""
    ckpt_dir = str(tmp_path / "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Create fake model file
    best_path = os.path.join(ckpt_dir, "best_model.pt")
    with open(best_path, 'wb') as f:
        f.write(b"fake model data")

    # Create fake tokenizer
    tok_path = os.path.join(ckpt_dir, "tokenizer.pkl")
    with open(tok_path, 'wb') as f:
        f.write(b"fake tokenizer data")

    return ckpt_dir


class TestSanitizeName:
    def test_basic_name(self):
        assert model_registry._sanitize_name("my-model") == "my-model"

    def test_spaces_converted(self):
        assert model_registry._sanitize_name("my model") == "my_model"

    def test_special_chars(self):
        result = model_registry._sanitize_name("My Model! v2.0")
        assert result == "my_model__v2_0"

    def test_empty_after_sanitize(self):
        assert model_registry._sanitize_name("!!!") == ""

    def test_truncation(self):
        long_name = "a" * 100
        assert len(model_registry._sanitize_name(long_name)) <= 64


class TestRegistryOperations:
    def test_empty_registry(self):
        reg = model_registry._load_registry()
        assert reg == {"models": {}}

    def test_register_model(self, fake_checkpoint):
        path = model_registry.register_model(
            name="test-bot",
            intent="Testing",
            base_model="custom",
            pipeline="scratch",
            checkpoint_dir=fake_checkpoint
        )
        assert os.path.exists(path)
        assert os.path.exists(os.path.join(path, "model.pt"))

    def test_register_and_list(self, fake_checkpoint, capsys):
        model_registry.register_model(
            name="test-bot",
            intent="Testing",
            base_model="custom",
            pipeline="scratch",
            checkpoint_dir=fake_checkpoint
        )
        model_registry.list_registered_models()
        captured = capsys.readouterr()
        assert "test-bot" in captured.out
        assert "Testing" in captured.out

    def test_get_model_info(self, fake_checkpoint):
        model_registry.register_model(
            name="my-bot",
            intent="Chat helper",
            base_model="gpt2",
            pipeline="finetune",
            checkpoint_dir=fake_checkpoint
        )
        info = model_registry.get_model_info("my-bot")
        assert info is not None
        assert info["name"] == "my-bot"
        assert info["intent"] == "Chat helper"
        assert info["base_model"] == "gpt2"

    def test_get_model_path(self, fake_checkpoint):
        model_registry.register_model(
            name="path-test",
            intent="Testing paths",
            base_model="custom",
            pipeline="scratch",
            checkpoint_dir=fake_checkpoint
        )
        path = model_registry.get_model_path("path-test")
        assert path is not None
        assert path.endswith("model.pt")
        assert os.path.exists(path)

    def test_get_tokenizer_path(self, fake_checkpoint):
        model_registry.register_model(
            name="tok-test",
            intent="Testing tokenizer",
            base_model="custom",
            pipeline="scratch",
            checkpoint_dir=fake_checkpoint
        )
        path = model_registry.get_tokenizer_path("tok-test")
        assert path is not None
        assert path.endswith("tokenizer.pkl")

    def test_delete_model(self, fake_checkpoint):
        model_registry.register_model(
            name="delete-me",
            intent="To be deleted",
            base_model="custom",
            pipeline="scratch",
            checkpoint_dir=fake_checkpoint
        )
        assert model_registry.get_model_info("delete-me") is not None
        result = model_registry.delete_model("delete-me")
        assert result is True
        assert model_registry.get_model_info("delete-me") is None

    def test_delete_nonexistent(self):
        result = model_registry.delete_model("nonexistent")
        assert result is False

    def test_get_info_nonexistent(self):
        info = model_registry.get_model_info("nonexistent")
        assert info is None

    def test_list_empty(self, capsys):
        model_registry.list_registered_models()
        captured = capsys.readouterr()
        assert "No registered models" in captured.out

    def test_register_with_data_sources(self, fake_checkpoint, tmp_path):
        # Create a fake data file
        data_file = str(tmp_path / "train.json")
        with open(data_file, 'w') as f:
            json.dump([{"prompt": "hi", "response": "hello"}], f)

        path = model_registry.register_model(
            name="data-bot",
            intent="Test with data",
            base_model="custom",
            pipeline="scratch",
            checkpoint_dir=fake_checkpoint,
            data_sources=[data_file]
        )
        # Check data was copied
        assert os.path.exists(os.path.join(path, "data", "train.json"))

    def test_invalid_name(self, fake_checkpoint):
        with pytest.raises(ValueError):
            model_registry.register_model(
                name="!!!",
                intent="Bad name",
                base_model="custom",
                pipeline="scratch",
                checkpoint_dir=fake_checkpoint
            )
