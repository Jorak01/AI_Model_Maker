"""Tests for the model factory."""

import os
import sys
import json
import tempfile
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_factory import (
    create_model, load_model, list_models, list_pipelines,
    MODEL_INFO, PIPELINE_INFO, PRETRAINED_MODELS,
    list_model_families, refresh_models, get_model_latest_info,
    _load_model_cache, _save_model_cache, _cache_is_stale,
    MODEL_CACHE_FILE, MODEL_CACHE_DIR,
)
from models.model import ConversationalModel


class TestModelInfo:
    def test_model_info_has_many_entries(self):
        """Expanded registry should have 30+ models."""
        assert len(MODEL_INFO) >= 30
        assert "custom" in MODEL_INFO
        assert "gpt2" in MODEL_INFO

    def test_all_pretrained_have_info(self):
        """Every model in PRETRAINED_MODELS should have a MODEL_INFO entry."""
        for name in PRETRAINED_MODELS:
            assert name in MODEL_INFO, f"Missing MODEL_INFO for '{name}'"

    def test_model_info_fields(self):
        """Each MODEL_INFO entry should have required fields."""
        for name, info in MODEL_INFO.items():
            assert "params" in info, f"'{name}' missing 'params'"
            assert "desc" in info, f"'{name}' missing 'desc'"
            assert "pretrained" in info, f"'{name}' missing 'pretrained'"
            assert "family" in info, f"'{name}' missing 'family'"

    def test_custom_is_not_pretrained(self):
        assert MODEL_INFO["custom"]["pretrained"] is False

    def test_all_pretrained_are_flagged(self):
        for name in PRETRAINED_MODELS:
            assert MODEL_INFO[name]["pretrained"] is True

    def test_pipeline_info_has_entries(self):
        assert len(PIPELINE_INFO) == 3
        assert "scratch" in PIPELINE_INFO
        assert "finetune" in PIPELINE_INFO
        assert "freeze" in PIPELINE_INFO

    def test_pretrained_models_mapping(self):
        assert "gpt2" in PRETRAINED_MODELS
        assert "distilgpt2" in PRETRAINED_MODELS
        assert PRETRAINED_MODELS["gpt2"] == "gpt2"

    def test_expanded_model_families(self):
        """Check that key model families are present."""
        families = set(info["family"] for info in MODEL_INFO.values())
        expected = {"custom", "gpt2", "pythia", "phi", "llama", "qwen", "smollm"}
        assert expected.issubset(families), f"Missing families: {expected - families}"

    def test_new_models_present(self):
        """Verify newly added models are in the registry."""
        new_models = [
            "gpt2-large", "gpt2-xl",
            "pythia-1b", "pythia-2.8b",
            "phi-2", "phi-3-mini", "phi-4-mini",
            "llama-3.2-1b",
            "qwen2.5-0.5b", "qwen2.5-1.5b",
            "smollm2-135m", "smollm2-1.7b",
            "bloom-560m", "falcon-rw-1b",
            "tinyllama-1.1b", "stablelm-2-1.6b",
            "olmo-1b",
            "deepseek-r1-distill-qwen-1.5b",
        ]
        for model in new_models:
            assert model in MODEL_INFO, f"Expected model '{model}' not found"
            assert model in PRETRAINED_MODELS, f"Expected pretrained '{model}' not found"


class TestModelFamilies:
    def test_list_model_families(self):
        families = list_model_families()
        assert isinstance(families, dict)
        assert "gpt2" in families
        assert "custom" in families
        # GPT-2 family should have multiple members
        assert len(families["gpt2"]) >= 3

    def test_all_models_in_some_family(self):
        families = list_model_families()
        all_names = set()
        for names in families.values():
            all_names.update(names)
        for model_name in MODEL_INFO:
            assert model_name in all_names, f"'{model_name}' not in any family"


class TestModelCache:
    def test_load_empty_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr("models.model_factory.MODEL_CACHE_FILE",
                            str(tmp_path / "nonexistent.json"))
        cache = _load_model_cache()
        assert cache == {}

    def test_save_and_load_cache(self, tmp_path, monkeypatch):
        cache_file = str(tmp_path / "cache.json")
        cache_dir = str(tmp_path)
        monkeypatch.setattr("models.model_factory.MODEL_CACHE_FILE", cache_file)
        monkeypatch.setattr("models.model_factory.MODEL_CACHE_DIR", cache_dir)

        data = {"last_updated": "2026-01-01T00:00:00", "models": {"gpt2": {"available": True}}}
        _save_model_cache(data)
        loaded = _load_model_cache()
        assert loaded["models"]["gpt2"]["available"] is True

    def test_stale_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr("models.model_factory.MODEL_CACHE_FILE",
                            str(tmp_path / "nonexistent.json"))
        assert _cache_is_stale() is True

    def test_fresh_cache(self, tmp_path, monkeypatch):
        from datetime import datetime
        cache_file = str(tmp_path / "cache.json")
        cache_dir = str(tmp_path)
        monkeypatch.setattr("models.model_factory.MODEL_CACHE_FILE", cache_file)
        monkeypatch.setattr("models.model_factory.MODEL_CACHE_DIR", cache_dir)

        data = {"last_updated": datetime.now().isoformat()}
        _save_model_cache(data)
        assert _cache_is_stale() is False


class TestRefreshModels:
    def test_refresh_returns_dict(self, tmp_path, monkeypatch):
        """refresh_models should return a dict even if network is unavailable."""
        cache_file = str(tmp_path / "cache.json")
        cache_dir = str(tmp_path)
        monkeypatch.setattr("models.model_factory.MODEL_CACHE_FILE", cache_file)
        monkeypatch.setattr("models.model_factory.MODEL_CACHE_DIR", cache_dir)

        # Mock _fetch_hf_model_info to avoid network calls
        monkeypatch.setattr("models.model_factory._fetch_hf_model_info", lambda *a, **kw: None)

        results = refresh_models(force=True, verbose=False)
        assert isinstance(results, dict)
        # All models should be marked unavailable since we mocked the fetch
        for name in PRETRAINED_MODELS:
            assert name in results
            assert results[name]["available"] is False

    def test_refresh_with_mock_data(self, tmp_path, monkeypatch):
        cache_file = str(tmp_path / "cache.json")
        cache_dir = str(tmp_path)
        monkeypatch.setattr("models.model_factory.MODEL_CACHE_FILE", cache_file)
        monkeypatch.setattr("models.model_factory.MODEL_CACHE_DIR", cache_dir)

        fake_info = {"lastModified": "2026-03-01", "downloads": 1000, "tags": ["text-generation"]}
        monkeypatch.setattr("models.model_factory._fetch_hf_model_info",
                            lambda *a, **kw: fake_info)

        results = refresh_models(force=True, verbose=False)
        for name in PRETRAINED_MODELS:
            assert results[name]["available"] is True
            assert results[name]["downloads"] == 1000

    def test_refresh_uses_cache(self, tmp_path, monkeypatch):
        from datetime import datetime
        cache_file = str(tmp_path / "cache.json")
        cache_dir = str(tmp_path)
        monkeypatch.setattr("models.model_factory.MODEL_CACHE_FILE", cache_file)
        monkeypatch.setattr("models.model_factory.MODEL_CACHE_DIR", cache_dir)

        cached = {
            "last_updated": datetime.now().isoformat(),
            "models": {"gpt2": {"available": True, "hf_id": "gpt2"}}
        }
        _save_model_cache(cached)

        # Should return cached data without calling fetch
        fetch_called = []
        monkeypatch.setattr("models.model_factory._fetch_hf_model_info",
                            lambda *a, **kw: fetch_called.append(1) or None)

        results = refresh_models(force=False, verbose=False)
        assert len(fetch_called) == 0  # Fetch was not called
        assert results.get("gpt2", {}).get("available") is True


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
        assert "Total:" in captured.out
        assert "pretrained" in captured.out.lower()

    def test_list_models_by_family(self, capsys):
        list_models(by_family=True)
        captured = capsys.readouterr()
        assert "GPT2" in captured.out
        assert "PYTHIA" in captured.out
        assert "PHI" in captured.out

    def test_list_pipelines(self, capsys):
        list_pipelines()
        captured = capsys.readouterr()
        assert "scratch" in captured.out
        assert "finetune" in captured.out
        assert "freeze" in captured.out
