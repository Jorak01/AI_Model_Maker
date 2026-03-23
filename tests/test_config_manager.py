"""Tests for the configuration manager — profiles, validation, env overrides."""

import os
import sys
import json
import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_manager import (
    load_config, save_config, validate_config, apply_env_overrides,
    diff_configs, list_profiles, load_profile, save_profile,
    create_default_profiles, CONFIG_SCHEMA, ENV_OVERRIDES,
)


class TestConfigSchema:
    def test_schema_has_required_sections(self):
        assert "model" in CONFIG_SCHEMA
        assert "training" in CONFIG_SCHEMA
        assert "device" in CONFIG_SCHEMA
        assert "generation" in CONFIG_SCHEMA

    def test_model_section_required_keys(self):
        assert "base_model" in CONFIG_SCHEMA["model"]["required"]
        assert "vocab_size" in CONFIG_SCHEMA["model"]["required"]

    def test_env_overrides_mapping(self):
        assert "AI_MODEL_DEVICE" in ENV_OVERRIDES
        assert "AI_MODEL_EPOCHS" in ENV_OVERRIDES
        assert len(ENV_OVERRIDES) >= 10


class TestLoadSaveConfig:
    def test_load_default_config(self):
        config = load_config()
        assert isinstance(config, dict)
        assert "model" in config
        assert "training" in config

    def test_save_and_load_roundtrip(self, tmp_path):
        path = str(tmp_path / "test_config.yaml")
        original = {"model": {"base_model": "custom"}, "training": {"epochs": 5}}
        save_config(original, path)
        loaded = load_config(path)
        assert loaded["model"]["base_model"] == "custom"
        assert loaded["training"]["epochs"] == 5


class TestValidation:
    def test_valid_config_passes(self):
        config = load_config()
        errors = validate_config(config, verbose=False)
        assert len(errors) == 0

    def test_missing_section_detected(self):
        config = {"model": {"base_model": "custom"}}
        errors = validate_config(config, verbose=False)
        assert any("Missing section" in e for e in errors)

    def test_type_error_detected(self):
        config = load_config()
        config["model"]["vocab_size"] = "not_a_number"
        errors = validate_config(config, verbose=False)
        assert any("Type error" in e for e in errors)

    def test_range_error_detected(self):
        config = load_config()
        config["model"]["dropout"] = 5.0  # Should be 0-1
        errors = validate_config(config, verbose=False)
        assert any("Range error" in e for e in errors)

    def test_invalid_pipeline_detected(self):
        config = load_config()
        config["training"]["pipeline"] = "nonexistent"
        errors = validate_config(config, verbose=False)
        assert any("Invalid value" in e for e in errors)


class TestEnvOverrides:
    def test_no_env_vars_no_changes(self):
        config = load_config()
        original_epochs = config["training"]["num_epochs"]
        result = apply_env_overrides(config)
        assert result["training"]["num_epochs"] == original_epochs

    def test_env_override_applied(self, monkeypatch):
        monkeypatch.setenv("AI_MODEL_EPOCHS", "99")
        config = load_config()
        result = apply_env_overrides(config)
        assert result["training"]["num_epochs"] == 99

    def test_env_override_device(self, monkeypatch):
        monkeypatch.setenv("AI_MODEL_DEVICE", "cuda:0")
        config = load_config()
        result = apply_env_overrides(config)
        assert result["device"]["use_cuda"] is True

    def test_env_override_cpu(self, monkeypatch):
        monkeypatch.setenv("AI_MODEL_DEVICE", "cpu")
        config = load_config()
        result = apply_env_overrides(config)
        assert result["device"]["use_cuda"] is False

    def test_original_config_unchanged(self, monkeypatch):
        monkeypatch.setenv("AI_MODEL_EPOCHS", "99")
        config = load_config()
        original_epochs = config["training"]["num_epochs"]
        apply_env_overrides(config)
        assert config["training"]["num_epochs"] == original_epochs


class TestProfiles:
    def test_list_profiles_in_empty_dir(self, tmp_path):
        profiles = list_profiles(str(tmp_path))
        assert profiles == []

    def test_create_and_list_profiles(self, tmp_path):
        # Copy config.yaml to tmp_path
        import shutil
        shutil.copy("config.yaml", str(tmp_path / "config.yaml"))
        created = create_default_profiles(str(tmp_path))
        assert created == 3
        profiles = list_profiles(str(tmp_path))
        assert "fast" in profiles
        assert "quality" in profiles
        assert "gpu" in profiles

    def test_load_nonexistent_profile(self):
        result = load_profile("nonexistent_xyz")
        assert result is None

    def test_save_and_load_profile(self, tmp_path):
        config = {"test": True, "value": 42}
        save_profile("myprofile", config, str(tmp_path))
        loaded = load_profile("myprofile", str(tmp_path))
        assert loaded is not None
        assert loaded["test"] is True
        assert loaded["value"] == 42


class TestDiffConfigs:
    def test_identical_configs(self):
        config = load_config()
        diffs = diff_configs(config, config)
        assert len(diffs) == 0

    def test_different_values(self):
        a = {"model": {"epochs": 5}}
        b = {"model": {"epochs": 10}}
        diffs = diff_configs(a, b)
        assert len(diffs) == 1
        assert "5" in diffs[0] and "10" in diffs[0]

    def test_missing_key(self):
        a = {"model": {"epochs": 5}}
        b = {"model": {"epochs": 5, "lr": 0.001}}
        diffs = diff_configs(a, b)
        assert len(diffs) == 1
        assert "only in" in diffs[0]
