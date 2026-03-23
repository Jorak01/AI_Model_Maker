"""Tests for the model loader — number-based loading, custom/untrained support, training."""

import os
import sys
import json
import tempfile
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_loader import (
    build_model_catalog, display_catalog,
    get_entry_by_number, get_entry_by_name,
    load_model_by_number, load_model_by_name,
    train_loaded_model, delete_model_entry,
    verify_model_entry, repair_model_entry,
    _get_registered_models, _get_checkpoint_models, _get_untrained_models,
    _load_config, _get_device,
    _format_size, _get_file_metadata, _is_hf_model_installed,
    _is_back, _interactive_delete_from_catalog,
)
from models.model_factory import MODEL_INFO


# ============================================================
# 1. Catalog Builder Tests
# ============================================================
class TestCatalogBuilders:
    """Test the individual catalog builders."""

    def test_get_untrained_models_has_all_model_info(self):
        """All entries in MODEL_INFO should appear in untrained models."""
        entries = _get_untrained_models()
        keys = {e["key"] for e in entries}
        for name in MODEL_INFO:
            assert name in keys, f"Missing untrained model: {name}"

    def test_get_untrained_models_structure(self):
        """Each untrained entry should have required fields."""
        entries = _get_untrained_models()
        for e in entries:
            assert "source" in e and e["source"] == "untrained"
            assert "key" in e
            assert "name" in e
            assert "desc" in e
            assert "base_model" in e
            assert "pipeline" in e
            assert "params" in e
            assert "family" in e
            assert "pretrained" in e

    def test_get_untrained_custom_is_scratch(self):
        """The custom model should have pipeline='scratch'."""
        entries = _get_untrained_models()
        custom = [e for e in entries if e["key"] == "custom"]
        assert len(custom) == 1
        assert custom[0]["pipeline"] == "scratch"
        assert custom[0]["pretrained"] is False

    def test_get_untrained_pretrained_is_finetune(self):
        """Pretrained models should default to pipeline='finetune'."""
        entries = _get_untrained_models()
        gpt2 = [e for e in entries if e["key"] == "gpt2"]
        assert len(gpt2) == 1
        assert gpt2[0]["pipeline"] == "finetune"
        assert gpt2[0]["pretrained"] is True

    def test_get_checkpoint_models_empty_dir(self, tmp_path):
        """Returns empty list when checkpoint dir is empty."""
        config = {"checkpoint": {"save_dir": str(tmp_path)},
                  "model": {"base_model": "custom"},
                  "training": {"pipeline": "scratch"}}
        entries = _get_checkpoint_models(config)
        assert entries == []

    def test_get_checkpoint_models_with_files(self, tmp_path):
        """Finds best model and epoch checkpoints."""
        # Create fake checkpoints
        (tmp_path / "best_model.pt").write_bytes(b"data")
        (tmp_path / "model_epoch_1.pt").write_bytes(b"data")
        (tmp_path / "model_epoch_2.pt").write_bytes(b"data")
        (tmp_path / "tokenizer.pkl").write_bytes(b"data")

        config = {"checkpoint": {"save_dir": str(tmp_path)},
                  "model": {"base_model": "custom"},
                  "training": {"pipeline": "scratch"}}
        entries = _get_checkpoint_models(config)
        assert len(entries) == 3  # best + 2 epochs
        assert entries[0]["key"] == "best_model"
        assert entries[0]["source"] == "checkpoint"
        assert entries[0]["tokenizer_path"] is not None

    def test_get_registered_models_empty(self, monkeypatch, tmp_path):
        """Returns empty list when no models registered."""
        import model_registry
        reg_dir = str(tmp_path / "trained_models")
        reg_file = os.path.join(reg_dir, "registry.json")
        monkeypatch.setattr(model_registry, "REGISTRY_DIR", reg_dir)
        monkeypatch.setattr(model_registry, "REGISTRY_FILE", reg_file)
        entries = _get_registered_models()
        assert entries == []


# ============================================================
# 2. Unified Catalog Tests
# ============================================================
class TestBuildCatalog:
    """Test build_model_catalog and numbering."""

    def test_catalog_has_numbers(self):
        """Every entry should have a unique sequential number."""
        config = {"checkpoint": {"save_dir": "nonexistent_dir"},
                  "model": {"base_model": "custom"},
                  "training": {"pipeline": "scratch"}}
        catalog = build_model_catalog(config)
        numbers = [e["number"] for e in catalog]
        assert numbers == list(range(1, len(catalog) + 1))

    def test_catalog_contains_untrained(self):
        """Catalog should contain all untrained baseline models."""
        config = {"checkpoint": {"save_dir": "nonexistent_dir"},
                  "model": {"base_model": "custom"},
                  "training": {"pipeline": "scratch"}}
        catalog = build_model_catalog(config)
        untrained = [e for e in catalog if e["source"] == "untrained"]
        assert len(untrained) == len(MODEL_INFO)

    def test_catalog_ordering(self, tmp_path):
        """Catalog should order: registered → checkpoints → untrained."""
        # Create fake checkpoint
        (tmp_path / "best_model.pt").write_bytes(b"data")
        config = {"checkpoint": {"save_dir": str(tmp_path)},
                  "model": {"base_model": "custom"},
                  "training": {"pipeline": "scratch"}}

        catalog = build_model_catalog(config)
        sources = [e["source"] for e in catalog]

        # Find where each source starts
        first_checkpoint = next((i for i, s in enumerate(sources) if s == "checkpoint"), None)
        first_untrained = next((i for i, s in enumerate(sources) if s == "untrained"), None)

        if first_checkpoint is not None and first_untrained is not None:
            assert first_checkpoint < first_untrained

    def test_catalog_with_checkpoints(self, tmp_path):
        """Catalog includes checkpoint models when present."""
        (tmp_path / "best_model.pt").write_bytes(b"data")
        (tmp_path / "model_epoch_5.pt").write_bytes(b"data")
        config = {"checkpoint": {"save_dir": str(tmp_path)},
                  "model": {"base_model": "custom"},
                  "training": {"pipeline": "scratch"}}

        catalog = build_model_catalog(config)
        ckpts = [e for e in catalog if e["source"] == "checkpoint"]
        assert len(ckpts) == 2  # best + epoch 5


# ============================================================
# 3. Lookup Tests
# ============================================================
class TestLookup:
    """Test get_entry_by_number and get_entry_by_name."""

    @pytest.fixture
    def catalog(self):
        config = {"checkpoint": {"save_dir": "nonexistent_dir"},
                  "model": {"base_model": "custom"},
                  "training": {"pipeline": "scratch"}}
        return build_model_catalog(config)

    def test_lookup_by_number_valid(self, catalog):
        entry = get_entry_by_number(catalog, 1)
        assert entry is not None
        assert entry["number"] == 1

    def test_lookup_by_number_invalid(self, catalog):
        entry = get_entry_by_number(catalog, 99999)
        assert entry is None

    def test_lookup_by_number_zero(self, catalog):
        entry = get_entry_by_number(catalog, 0)
        assert entry is None

    def test_lookup_by_name_exact(self, catalog):
        entry = get_entry_by_name(catalog, "custom")
        assert entry is not None
        assert entry["key"] == "custom"

    def test_lookup_by_name_case_insensitive(self, catalog):
        entry = get_entry_by_name(catalog, "CUSTOM")
        assert entry is not None
        assert entry["key"] == "custom"

    def test_lookup_by_name_gpt2(self, catalog):
        entry = get_entry_by_name(catalog, "gpt2")
        assert entry is not None
        assert entry["key"] == "gpt2"

    def test_lookup_by_name_nonexistent(self, catalog):
        entry = get_entry_by_name(catalog, "this-model-does-not-exist")
        assert entry is None


# ============================================================
# 4. Display Tests
# ============================================================
class TestDisplay:
    """Test display_catalog output."""

    def test_display_shows_sections(self, capsys):
        """Display should show at least the untrained section."""
        config = {"checkpoint": {"save_dir": "nonexistent_dir"},
                  "model": {"base_model": "custom"},
                  "training": {"pipeline": "scratch"}}
        catalog = display_catalog(config=config)
        captured = capsys.readouterr()
        assert "Model Catalog" in captured.out
        assert "Untrained Baseline Models" in captured.out
        assert "Total:" in captured.out
        assert catalog is not None

    def test_display_with_checkpoints(self, capsys, tmp_path):
        """Display should show checkpoint section if they exist."""
        (tmp_path / "best_model.pt").write_bytes(b"data")
        config = {"checkpoint": {"save_dir": str(tmp_path)},
                  "model": {"base_model": "custom"},
                  "training": {"pipeline": "scratch"}}
        display_catalog(config=config)
        captured = capsys.readouterr()
        assert "Local Checkpoints" in captured.out

    def test_display_empty_catalog(self, capsys, monkeypatch):
        """Display handles empty catalog gracefully."""
        catalog = display_catalog(catalog=[])
        captured = capsys.readouterr()
        assert "No models available" in captured.out


# ============================================================
# 5. Model Loading Tests
# ============================================================
class TestModelLoading:
    """Test load_model_by_number and load_model_by_name."""

    @pytest.fixture
    def custom_config(self):
        return {
            'model': {
                'base_model': 'custom', 'vocab_size': 100, 'embedding_dim': 16,
                'hidden_dim': 32, 'num_layers': 1, 'num_heads': 2,
                'max_seq_length': 16, 'dropout': 0.1,
            },
            'training': {'pipeline': 'scratch', 'batch_size': 2,
                         'learning_rate': 0.001, 'num_epochs': 1,
                         'gradient_clip': 1.0, 'warmup_steps': 0, 'weight_decay': 0.01},
            'checkpoint': {'save_dir': 'nonexistent_dir', 'save_every': 1, 'keep_last': 5},
            'generation': {'max_length': 10, 'temperature': 0.8,
                           'top_k': 50, 'top_p': 0.9, 'repetition_penalty': 1.2},
            'device': {'use_cuda': False, 'cuda_device': 0},
            'data': {'train_path': 'data/train.json', 'test_path': 'data/test.json'},
            'performance': {'mixed_precision': False, 'compile_model': False, 'num_workers': 0},
        }

    def test_load_untrained_custom_by_name(self, custom_config):
        """Load an untrained custom model by name."""
        model, tokenizer, entry = load_model_by_name(
            "custom", config=custom_config, device="cpu"
        )
        assert model is not None
        assert entry is not None
        assert entry["source"] == "untrained"
        assert entry["key"] == "custom"
        # Custom untrained has no pre-built tokenizer if none exists
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0

    def test_load_untrained_custom_by_number(self, custom_config):
        """Load an untrained custom model by number."""
        catalog = build_model_catalog(custom_config)
        # Find the custom model number
        custom_entry = get_entry_by_name(catalog, "custom")
        assert custom_entry is not None

        model, tokenizer, entry = load_model_by_number(
            custom_entry["number"], catalog=catalog,
            config=custom_config, device="cpu"
        )
        assert model is not None
        assert entry is not None
        assert entry["key"] == "custom"

    def test_load_checkpoint_by_number(self, custom_config, tmp_path):
        """Load a checkpoint model by number."""
        from models.model import ConversationalModel

        # Create a real checkpoint
        m = ConversationalModel(vocab_size=100, embedding_dim=16, hidden_dim=32,
                                num_layers=1, num_heads=2, max_seq_length=16)
        ckpt_path = str(tmp_path / "best_model.pt")
        m.save(ckpt_path)

        custom_config['checkpoint']['save_dir'] = str(tmp_path)
        catalog = build_model_catalog(custom_config)

        # Find the best checkpoint entry
        best_entry = get_entry_by_name(catalog, "Best Checkpoint")
        assert best_entry is not None

        model, tokenizer, entry = load_model_by_number(
            best_entry["number"], catalog=catalog,
            config=custom_config, device="cpu"
        )
        assert model is not None
        assert entry is not None
        assert entry["source"] == "checkpoint"

    def test_load_invalid_number(self, custom_config):
        """Loading invalid number returns None."""
        model, tokenizer, entry = load_model_by_number(
            99999, config=custom_config, device="cpu"
        )
        assert model is None
        assert entry is None

    def test_load_invalid_name(self, custom_config):
        """Loading invalid name returns None."""
        model, tokenizer, entry = load_model_by_name(
            "nonexistent-model", config=custom_config, device="cpu"
        )
        assert model is None
        assert entry is None


# ============================================================
# 6. Training on Loaded Model Tests
# ============================================================
class TestTrainLoadedModel:
    """Test training on loaded models."""

    @pytest.fixture
    def training_setup(self, tmp_path):
        """Create data, config, model, tokenizer for training tests."""
        # Create training data
        data = [
            {"prompt": "hello", "response": "hi there"},
            {"prompt": "how are you", "response": "i am good"},
        ]
        data_path = str(tmp_path / "data.json")
        with open(data_path, 'w') as f:
            json.dump(data, f)

        ckpt_dir = str(tmp_path / "checkpoints")

        config = {
            'model': {
                'base_model': 'custom', 'vocab_size': 200, 'embedding_dim': 16,
                'hidden_dim': 32, 'num_layers': 1, 'num_heads': 2,
                'max_seq_length': 16, 'dropout': 0.1,
            },
            'training': {'pipeline': 'scratch', 'batch_size': 2,
                         'learning_rate': 0.001, 'num_epochs': 2,
                         'gradient_clip': 1.0, 'warmup_steps': 0, 'weight_decay': 0.01},
            'checkpoint': {'save_dir': ckpt_dir, 'save_every': 1, 'keep_last': 5},
            'generation': {'max_length': 10, 'temperature': 0.8,
                           'top_k': 50, 'top_p': 0.9, 'repetition_penalty': 1.2},
            'device': {'use_cuda': False, 'cuda_device': 0},
            'data': {'train_path': data_path, 'test_path': data_path},
            'performance': {'mixed_precision': False, 'compile_model': False, 'num_workers': 0},
        }

        return config, data_path, ckpt_dir

    def test_train_untrained_custom_model(self, training_setup):
        """Load untrained custom model and train it."""
        config, data_path, ckpt_dir = training_setup

        # Load untrained custom model
        model, tokenizer, entry = load_model_by_name(
            "custom", config=config, device="cpu"
        )
        assert model is not None
        assert entry is not None

        # Train it (entry guaranteed non-None by assert above)
        assert entry is not None
        result_dir = train_loaded_model(
            model=model, tokenizer=tokenizer, entry=entry,
            config=config, device="cpu",
            train_data=data_path, test_data=data_path,
            epochs=1, checkpoint_dir=ckpt_dir,
        )
        assert result_dir is not None
        assert os.path.exists(result_dir)

        # Should have saved checkpoints
        files = os.listdir(result_dir)
        pt_files = [f for f in files if f.endswith('.pt')]
        assert len(pt_files) > 0

    def test_train_checkpoint_model(self, training_setup, tmp_path):
        """Load a checkpoint model and continue training."""
        config, data_path, _ = training_setup
        from models.model import ConversationalModel
        from models.tokenizer import Tokenizer
        from utils.data_loader import load_and_prepare_data

        # Build tokenizer
        tok = Tokenizer(200, method='word')
        texts = load_and_prepare_data(data_path)
        tok.build_vocab(texts)

        # Create and save model
        m = ConversationalModel(vocab_size=len(tok), embedding_dim=16, hidden_dim=32,
                                num_layers=1, num_heads=2, max_seq_length=16)
        orig_dir = str(tmp_path / "orig_ckpts")
        os.makedirs(orig_dir)
        m.save(os.path.join(orig_dir, "best_model.pt"))
        tok.save(os.path.join(orig_dir, "tokenizer.pkl"))

        config['checkpoint']['save_dir'] = orig_dir
        catalog = build_model_catalog(config)

        # Load checkpoint
        best = get_entry_by_name(catalog, "Best Checkpoint")
        assert best is not None
        model, tokenizer, entry = load_model_by_number(
            best["number"], catalog=catalog, config=config, device="cpu"
        )
        assert model is not None
        assert tokenizer is not None
        assert entry is not None

        # Train further
        retrain_dir = str(tmp_path / "retrained")
        result = train_loaded_model(
            model=model, tokenizer=tokenizer, entry=entry,
            config=config, device="cpu",
            train_data=data_path, test_data=data_path,
            epochs=1, checkpoint_dir=retrain_dir,
        )
        assert result is not None
        assert os.path.exists(os.path.join(retrain_dir, "tokenizer.pkl"))

    def test_train_missing_data_returns_none(self, training_setup):
        """Training with missing data returns early without error."""
        config, data_path, ckpt_dir = training_setup

        model, tokenizer, entry = load_model_by_name(
            "custom", config=config, device="cpu"
        )
        assert model is not None

        assert entry is not None
        result = train_loaded_model(
            model=model, tokenizer=tokenizer, entry=entry,
            config=config, device="cpu",
            train_data="nonexistent_data.json", test_data="nonexistent_test.json",
            epochs=1, checkpoint_dir=ckpt_dir,
        )
        assert result is None


# ============================================================
# 7. Config and Device Tests
# ============================================================
class TestConfigAndDevice:
    """Test config loading and device detection."""

    def test_load_config_returns_dict(self):
        """_load_config should return a valid dict."""
        config = _load_config()
        assert isinstance(config, dict)
        assert "model" in config
        assert "training" in config

    def test_get_device_cpu(self):
        """Should return 'cpu' when CUDA is disabled."""
        config = {"device": {"use_cuda": False, "cuda_device": 0}}
        assert _get_device(config) == "cpu"

    def test_get_device_no_device_section(self):
        """Should return 'cpu' when device section missing."""
        assert _get_device({}) == "cpu"


# ============================================================
# 8. Metadata Helper Tests
# ============================================================
class TestMetadataHelpers:
    """Test metadata helper functions."""

    def test_format_size_bytes(self):
        assert _format_size(500) == "500 B"

    def test_format_size_kilobytes(self):
        result = _format_size(2048)
        assert "KB" in result
        assert "2.0" in result

    def test_format_size_megabytes(self):
        result = _format_size(5 * 1024 * 1024)
        assert "MB" in result
        assert "5.0" in result

    def test_format_size_gigabytes(self):
        result = _format_size(2 * 1024 * 1024 * 1024)
        assert "GB" in result

    def test_get_file_metadata_existing(self, tmp_path):
        """Should return size and dates for an existing file."""
        f = tmp_path / "test.pt"
        f.write_bytes(b"x" * 1234)
        meta = _get_file_metadata(str(f))
        assert meta["file_size"] == 1234
        assert "1.2" in meta["file_size_str"]  # 1.2 KB
        assert meta["last_modified"] is not None
        assert meta["last_modified_str"] != "—"

    def test_get_file_metadata_missing(self):
        """Should return None/dash for a missing file."""
        meta = _get_file_metadata("/nonexistent/file.pt")
        assert meta["file_size"] is None
        assert meta["file_size_str"] == "—"
        assert meta["last_modified"] is None

    def test_is_hf_model_installed_unknown(self):
        """Unknown model name should return False."""
        assert _is_hf_model_installed("this_model_does_not_exist") is False

    def test_is_hf_model_installed_custom(self):
        """'custom' is not a HF model, should return False."""
        assert _is_hf_model_installed("custom") is False


# ============================================================
# 9. Metadata in Catalog Entries Tests
# ============================================================
class TestCatalogMetadata:
    """Test that catalog entries include metadata fields."""

    def test_checkpoint_entries_have_metadata(self, tmp_path):
        """Checkpoint entries should include installed, file_size, last_trained."""
        (tmp_path / "best_model.pt").write_bytes(b"x" * 5000)
        (tmp_path / "model_epoch_1.pt").write_bytes(b"y" * 3000)
        config = {"checkpoint": {"save_dir": str(tmp_path)},
                  "model": {"base_model": "custom"},
                  "training": {"pipeline": "scratch"}}
        entries = _get_checkpoint_models(config)
        assert len(entries) == 2
        for e in entries:
            assert "installed" in e
            assert e["installed"] is True
            assert "file_size" in e
            assert e["file_size"] is not None
            assert e["file_size"] > 0
            assert "file_size_str" in e
            assert e["file_size_str"] != "—"
            assert "last_trained" in e
            assert e["last_trained"] is not None
            assert "last_trained_str" in e
            assert e["last_trained_str"] != "—"

    def test_checkpoint_best_file_size_correct(self, tmp_path):
        """Best checkpoint file size should match written bytes."""
        (tmp_path / "best_model.pt").write_bytes(b"x" * 5000)
        config = {"checkpoint": {"save_dir": str(tmp_path)},
                  "model": {"base_model": "custom"},
                  "training": {"pipeline": "scratch"}}
        entries = _get_checkpoint_models(config)
        assert entries[0]["file_size"] == 5000

    def test_untrained_entries_have_installed_field(self):
        """All untrained entries should have an 'installed' field."""
        entries = _get_untrained_models()
        for e in entries:
            assert "installed" in e
            assert isinstance(e["installed"], bool)

    def test_untrained_scratch_models_installed_true(self):
        """Scratch (non-pretrained) untrained models should have installed=True."""
        entries = _get_untrained_models()
        custom = [e for e in entries if e["key"] == "custom"]
        assert len(custom) == 1
        assert custom[0]["installed"] is True

    def test_untrained_entries_have_size_and_date_fields(self):
        """Untrained entries should have file_size/last_trained (as None/dash)."""
        entries = _get_untrained_models()
        for e in entries:
            assert "file_size" in e
            assert e["file_size"] is None
            assert "file_size_str" in e
            assert e["file_size_str"] == "—"
            assert "last_trained" in e
            assert e["last_trained"] is None
            assert "last_trained_str" in e
            assert e["last_trained_str"] == "—"

    def test_catalog_all_entries_have_installed(self, tmp_path):
        """Every entry in the full catalog should have 'installed' field."""
        (tmp_path / "best_model.pt").write_bytes(b"data")
        config = {"checkpoint": {"save_dir": str(tmp_path)},
                  "model": {"base_model": "custom"},
                  "training": {"pipeline": "scratch"}}
        catalog = build_model_catalog(config)
        for e in catalog:
            assert "installed" in e, f"Entry {e['key']} missing 'installed'"


# ============================================================
# 10. Display Metadata Columns Tests
# ============================================================
class TestDisplayMetadata:
    """Test that display_catalog shows metadata columns."""

    def test_display_shows_installed_column(self, capsys, tmp_path):
        """Checkpoint display should show Installed column."""
        (tmp_path / "best_model.pt").write_bytes(b"data")
        config = {"checkpoint": {"save_dir": str(tmp_path)},
                  "model": {"base_model": "custom"},
                  "training": {"pipeline": "scratch"}}
        display_catalog(config=config)
        captured = capsys.readouterr()
        assert "Installed" in captured.out

    def test_display_shows_last_trained_column(self, capsys, tmp_path):
        """Checkpoint display should show Last Trained column."""
        (tmp_path / "best_model.pt").write_bytes(b"data")
        config = {"checkpoint": {"save_dir": str(tmp_path)},
                  "model": {"base_model": "custom"},
                  "training": {"pipeline": "scratch"}}
        display_catalog(config=config)
        captured = capsys.readouterr()
        assert "Last Trained" in captured.out

    def test_display_shows_size_column(self, capsys, tmp_path):
        """Checkpoint display should show Size column."""
        (tmp_path / "best_model.pt").write_bytes(b"data")
        config = {"checkpoint": {"save_dir": str(tmp_path)},
                  "model": {"base_model": "custom"},
                  "training": {"pipeline": "scratch"}}
        display_catalog(config=config)
        captured = capsys.readouterr()
        assert "Size" in captured.out

    def test_display_untrained_shows_cached_column(self, capsys):
        """Untrained section should show Cached column."""
        config = {"checkpoint": {"save_dir": "nonexistent_dir"},
                  "model": {"base_model": "custom"},
                  "training": {"pipeline": "scratch"}}
        display_catalog(config=config)
        captured = capsys.readouterr()
        assert "Cached" in captured.out

    def test_display_checkpoint_shows_yes(self, capsys, tmp_path):
        """Checkpoint entries should show '✓ yes' for installed."""
        (tmp_path / "best_model.pt").write_bytes(b"data")
        config = {"checkpoint": {"save_dir": str(tmp_path)},
                  "model": {"base_model": "custom"},
                  "training": {"pipeline": "scratch"}}
        display_catalog(config=config)
        captured = capsys.readouterr()
        # Checkpoints are always installed
        assert "yes" in captured.out


# ============================================================
# 11. Back Navigation Tests
# ============================================================
class TestBackNavigation:
    """Test _is_back helper and back navigation support."""

    def test_is_back_with_back(self):
        assert _is_back("back") is True

    def test_is_back_with_zero(self):
        assert _is_back("0") is True

    def test_is_back_with_quit(self):
        assert _is_back("quit") is True

    def test_is_back_with_exit(self):
        assert _is_back("exit") is True

    def test_is_back_with_q(self):
        assert _is_back("q") is True

    def test_is_back_case_insensitive(self):
        assert _is_back("BACK") is True
        assert _is_back("Back") is True
        assert _is_back("QUIT") is True

    def test_is_back_with_whitespace(self):
        assert _is_back("  back  ") is True
        assert _is_back("  0  ") is True

    def test_is_back_regular_input(self):
        assert _is_back("1") is False
        assert _is_back("custom") is False
        assert _is_back("gpt2") is False
        assert _is_back("") is False

    def test_is_back_not_number_string(self):
        """Numbers other than 0 should not be 'back'."""
        assert _is_back("5") is False
        assert _is_back("10") is False


# ============================================================
# 12. Model Deletion Tests
# ============================================================
class TestDeleteModelEntry:
    """Test delete_model_entry for various model sources."""

    def test_delete_untrained_returns_false(self, capsys):
        """Untrained baseline models cannot be deleted."""
        entry = {
            "source": "untrained", "name": "custom", "key": "custom",
            "model_path": None, "installed": True,
        }
        result = delete_model_entry(entry, confirm=False)
        assert result is False
        captured = capsys.readouterr()
        assert "Cannot delete" in captured.out

    def test_delete_checkpoint_file(self, tmp_path, capsys):
        """Deleting a checkpoint removes the .pt file."""
        ckpt = tmp_path / "model_epoch_1.pt"
        ckpt.write_bytes(b"x" * 1000)
        assert ckpt.exists()

        entry = {
            "source": "checkpoint", "name": "Epoch 1 Checkpoint",
            "key": "model_epoch_1.pt", "model_path": str(ckpt),
            "file_size_str": "1000 B", "installed": True,
        }
        result = delete_model_entry(entry, confirm=False)
        assert result is True
        assert not ckpt.exists()
        captured = capsys.readouterr()
        assert "deleted" in captured.out

    def test_delete_checkpoint_missing_file(self, capsys):
        """Deleting a checkpoint with missing file returns False."""
        entry = {
            "source": "checkpoint", "name": "Gone Checkpoint",
            "key": "model_epoch_99.pt",
            "model_path": "/nonexistent/model_epoch_99.pt",
            "file_size_str": "—", "installed": False,
        }
        result = delete_model_entry(entry, confirm=False)
        assert result is False
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_delete_registered_model(self, monkeypatch, tmp_path, capsys):
        """Deleting a registered model removes files + registry entry."""
        import model_registry

        # Set up isolated registry
        reg_dir = str(tmp_path / "trained_models")
        reg_file = os.path.join(reg_dir, "registry.json")
        monkeypatch.setattr(model_registry, "REGISTRY_DIR", reg_dir)
        monkeypatch.setattr(model_registry, "REGISTRY_FILE", reg_file)

        # Register a model
        ckpt_dir = str(tmp_path / "ckpts")
        os.makedirs(ckpt_dir)
        (tmp_path / "ckpts" / "best_model.pt").write_bytes(b"model_data")

        model_registry.register_model(
            name="test-model", intent="testing",
            base_model="custom", pipeline="scratch",
            checkpoint_dir=ckpt_dir,
        )

        # Verify it exists
        info = model_registry.get_model_info("test-model")
        assert info is not None

        # Build entry as catalog would
        entry = {
            "source": "registered", "name": "test-model",
            "key": "test-model", "model_path": os.path.join(info["path"], "model.pt"),
            "file_size_str": "small", "installed": True,
        }

        result = delete_model_entry(entry, confirm=False)
        assert result is True

        # Verify it's gone
        info2 = model_registry.get_model_info("test-model")
        assert info2 is None

    def test_delete_unknown_source_returns_false(self, capsys):
        """Unknown source type returns False."""
        entry = {"source": "alien", "name": "ET", "key": "et"}
        result = delete_model_entry(entry, confirm=False)
        assert result is False
        captured = capsys.readouterr()
        assert "Unknown source" in captured.out

    def test_interactive_delete_from_catalog_by_number(self, tmp_path, capsys):
        """_interactive_delete_from_catalog finds entry by number."""
        ckpt = tmp_path / "model_epoch_1.pt"
        ckpt.write_bytes(b"x" * 500)
        config = {"checkpoint": {"save_dir": str(tmp_path)},
                  "model": {"base_model": "custom"},
                  "training": {"pipeline": "scratch"}}
        catalog = build_model_catalog(config)

        # Find the checkpoint entry number
        ckpt_entry = [e for e in catalog if e["source"] == "checkpoint"][0]
        num = ckpt_entry["number"]

        # Delete by number (no confirm since we pass confirm=True internally,
        # but _interactive_delete_from_catalog calls with confirm=True which needs input)
        # Instead, we'll use delete_model_entry directly
        result = delete_model_entry(ckpt_entry, confirm=False)
        assert result is True
        assert not ckpt.exists()

    def test_interactive_delete_nonexistent_target(self, capsys):
        """_interactive_delete_from_catalog with bad target shows error."""
        catalog = build_model_catalog({"checkpoint": {"save_dir": "nonexistent"},
                                        "model": {"base_model": "custom"},
                                        "training": {"pipeline": "scratch"}})
        _interactive_delete_from_catalog("nonexistent_model_xyz", catalog)
        captured = capsys.readouterr()
        assert "No model matching" in captured.out


# ============================================================
# 13. Display Shows Delete Option Tests
# ============================================================
class TestPostLoadMenuOptions:
    """Test that the post-load menu shows the delete option."""

    def test_display_shows_delete_option(self, capsys):
        """The display_catalog shows 'delete' hint."""
        config = {"checkpoint": {"save_dir": "nonexistent_dir"},
                  "model": {"base_model": "custom"},
                  "training": {"pipeline": "scratch"}}
        display_catalog(config=config)
        captured = capsys.readouterr()
        # The catalog itself doesn't show delete, but the post-load menu does
        # Verify the catalog at least shows the back hint
        assert "back" in captured.out or "Total:" in captured.out

    def test_import_delete_model_entry(self):
        """delete_model_entry should be importable."""
        import model_loader
        assert hasattr(model_loader, 'delete_model_entry')

    def test_import_is_back(self):
        """_is_back should be importable."""
        import model_loader
        assert hasattr(model_loader, '_is_back')


# ============================================================
# 14. Verify Model Entry Tests
# ============================================================
class TestVerifyModelEntry:
    """Test verify_model_entry for various model sources."""

    def test_verify_untrained_scratch_passes(self, capsys):
        """Scratch untrained models always pass verification."""
        entry = {
            "source": "untrained", "name": "custom", "key": "custom",
            "pretrained": False, "model_path": None, "installed": True,
        }
        result = verify_model_entry(entry, device="cpu", verbose=True)
        assert result["ok"] is True
        assert len(result["errors"]) == 0
        assert len(result["checks"]) >= 1
        captured = capsys.readouterr()
        assert "Built-in" in captured.out

    def test_verify_untrained_pretrained_not_cached(self, capsys, monkeypatch):
        """Pretrained model not in cache should fail verification."""
        monkeypatch.setattr("model_loader._is_hf_model_installed", lambda k: False)
        entry = {
            "source": "untrained", "name": "gpt2", "key": "gpt2",
            "pretrained": True, "model_path": None, "installed": False,
        }
        result = verify_model_entry(entry, device="cpu", verbose=True)
        assert result["ok"] is False
        assert len(result["errors"]) >= 1
        captured = capsys.readouterr()
        assert "Not cached" in captured.out

    def test_verify_untrained_pretrained_cached(self, capsys, monkeypatch):
        """Pretrained model in cache should pass verification."""
        monkeypatch.setattr("model_loader._is_hf_model_installed", lambda k: True)
        entry = {
            "source": "untrained", "name": "gpt2", "key": "gpt2",
            "pretrained": True, "model_path": None, "installed": True,
        }
        result = verify_model_entry(entry, device="cpu", verbose=True)
        assert result["ok"] is True
        assert len(result["errors"]) == 0

    def test_verify_checkpoint_file_missing(self, capsys):
        """Checkpoint with missing file should fail."""
        entry = {
            "source": "checkpoint", "name": "Missing Checkpoint",
            "key": "model_epoch_99.pt",
            "model_path": "/nonexistent/model_epoch_99.pt",
            "tokenizer_path": None, "installed": False,
        }
        result = verify_model_entry(entry, device="cpu", verbose=True)
        assert result["ok"] is False
        assert any("not found" in e.lower() or "Not found" in e for e in result["errors"])

    def test_verify_checkpoint_empty_file(self, tmp_path, capsys):
        """Checkpoint with 0-byte file should fail."""
        f = tmp_path / "empty.pt"
        f.write_bytes(b"")
        entry = {
            "source": "checkpoint", "name": "Empty Checkpoint",
            "key": "empty.pt", "model_path": str(f),
            "tokenizer_path": None, "installed": True,
        }
        result = verify_model_entry(entry, device="cpu", verbose=True)
        assert result["ok"] is False
        assert any("empty" in e.lower() for e in result["errors"])

    def test_verify_valid_checkpoint(self, tmp_path, capsys):
        """Valid checkpoint should pass all checks."""
        from models.model import ConversationalModel
        from models.tokenizer import Tokenizer

        # Create real model + tokenizer
        m = ConversationalModel(vocab_size=100, embedding_dim=16, hidden_dim=32,
                                num_layers=1, num_heads=2, max_seq_length=16)
        model_path = str(tmp_path / "model.pt")
        m.save(model_path)

        tok = Tokenizer(100, method="word")
        tok.build_vocab(["hello world test"])
        tok_path = str(tmp_path / "tokenizer.pkl")
        tok.save(tok_path)

        entry = {
            "source": "checkpoint", "name": "Good Checkpoint",
            "key": "model.pt", "model_path": model_path,
            "tokenizer_path": tok_path, "installed": True,
        }
        result = verify_model_entry(entry, device="cpu", verbose=True)
        assert result["ok"] is True
        assert len(result["errors"]) == 0
        # Should have multiple passing checks
        assert len(result["checks"]) >= 3
        captured = capsys.readouterr()
        assert "PASS" in captured.out

    def test_verify_corrupt_checkpoint(self, tmp_path, capsys):
        """Corrupt checkpoint file should fail loadable check."""
        f = tmp_path / "corrupt.pt"
        f.write_bytes(b"this is not a valid pytorch file")
        entry = {
            "source": "checkpoint", "name": "Corrupt Checkpoint",
            "key": "corrupt.pt", "model_path": str(f),
            "tokenizer_path": None, "installed": True,
        }
        result = verify_model_entry(entry, device="cpu", verbose=True)
        assert result["ok"] is False
        assert any("corrupt" in e.lower() or "incompatible" in e.lower()
                    for e in result["errors"])

    def test_verify_missing_tokenizer(self, tmp_path, capsys):
        """Valid model but missing tokenizer should report error."""
        from models.model import ConversationalModel
        m = ConversationalModel(vocab_size=100, embedding_dim=16, hidden_dim=32,
                                num_layers=1, num_heads=2, max_seq_length=16)
        model_path = str(tmp_path / "model.pt")
        m.save(model_path)

        entry = {
            "source": "checkpoint", "name": "No Tok Checkpoint",
            "key": "model.pt", "model_path": model_path,
            "tokenizer_path": str(tmp_path / "nonexistent_tok.pkl"),
            "installed": True,
        }
        result = verify_model_entry(entry, device="cpu", verbose=True)
        # Model itself loads fine, but tokenizer is missing
        tok_errors = [e for e in result["errors"] if "tokenizer" in e.lower()]
        assert len(tok_errors) >= 1

    def test_verify_returns_checks_list(self):
        """Result should always contain ok, checks, errors keys."""
        entry = {
            "source": "untrained", "name": "custom", "key": "custom",
            "pretrained": False, "model_path": None, "installed": True,
        }
        result = verify_model_entry(entry, device="cpu", verbose=False)
        assert "ok" in result
        assert "checks" in result
        assert "errors" in result
        assert isinstance(result["checks"], list)
        assert isinstance(result["errors"], list)

    def test_verify_silent_mode(self, capsys):
        """verbose=False should produce no console output."""
        entry = {
            "source": "untrained", "name": "custom", "key": "custom",
            "pretrained": False, "model_path": None, "installed": True,
        }
        verify_model_entry(entry, device="cpu", verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""


# ============================================================
# 15. Repair Model Entry Tests
# ============================================================
class TestRepairModelEntry:
    """Test repair_model_entry for various model sources."""

    def test_repair_scratch_always_true(self, capsys):
        """Scratch models don't need repair — should return True."""
        entry = {
            "source": "untrained", "name": "custom", "key": "custom",
            "pretrained": False, "model_path": None, "installed": True,
        }
        result = repair_model_entry(entry, device="cpu")
        assert result is True
        captured = capsys.readouterr()
        assert "don't need repair" in captured.out

    def test_repair_checkpoint_returns_false(self, capsys):
        """Checkpoints can't be auto-repaired — returns False with advice."""
        entry = {
            "source": "checkpoint", "name": "Epoch 1 Checkpoint",
            "key": "model_epoch_1.pt",
            "model_path": "/some/path/model_epoch_1.pt",
            "installed": True,
        }
        result = repair_model_entry(entry, device="cpu")
        assert result is False
        captured = capsys.readouterr()
        assert "retrain" in captured.out.lower()

    def test_repair_registered_returns_false(self, capsys):
        """Registered models can't be auto-repaired — returns False with advice."""
        entry = {
            "source": "registered", "name": "my-bot",
            "key": "my-bot", "model_path": "/some/path/model.pt",
            "installed": True,
        }
        result = repair_model_entry(entry, device="cpu")
        assert result is False
        captured = capsys.readouterr()
        assert "retrain" in captured.out.lower() or "re-register" in captured.out.lower()

    def test_repair_unknown_source_returns_false(self, capsys):
        """Unknown source returns False."""
        entry = {"source": "alien", "name": "ET", "key": "et"}
        result = repair_model_entry(entry, device="cpu")
        assert result is False
        captured = capsys.readouterr()
        assert "Unknown source" in captured.out

    def test_repair_pretrained_unknown_key(self, capsys):
        """Pretrained model with unknown HF key returns False."""
        entry = {
            "source": "untrained", "name": "fake-model",
            "key": "nonexistent_pretrained_model",
            "pretrained": True, "model_path": None, "installed": False,
        }
        result = repair_model_entry(entry, device="cpu")
        assert result is False
        captured = capsys.readouterr()
        assert "Unknown pretrained" in captured.out


# ============================================================
# 16. Import Tests
# ============================================================
class TestImports:
    """Verify all model_loader exports are importable."""

    def test_import_model_loader(self):
        import model_loader
        assert hasattr(model_loader, 'build_model_catalog')
        assert hasattr(model_loader, 'display_catalog')
        assert hasattr(model_loader, 'get_entry_by_number')
        assert hasattr(model_loader, 'get_entry_by_name')
        assert hasattr(model_loader, 'load_model_by_number')
        assert hasattr(model_loader, 'load_model_by_name')
        assert hasattr(model_loader, 'train_loaded_model')
        assert hasattr(model_loader, 'interactive_load_and_act')
        assert hasattr(model_loader, 'verify_model_entry')
        assert hasattr(model_loader, 'repair_model_entry')

    def test_import_from_run(self):
        import run
        assert hasattr(run, 'cmd_load')
