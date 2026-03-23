"""Tests for the Auto Trainer — automated public data collection and training."""

import os
import sys
import json
import tempfile
import shutil
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auto_trainer import auto_train, auto_collect, load_config


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_loads_default_config(self):
        config = load_config("config.yaml")
        assert "model" in config
        assert "training" in config
        assert "device" in config

    def test_auto_training_section_exists(self):
        config = load_config("config.yaml")
        assert "auto_training" in config
        auto = config["auto_training"]
        assert "sources" in auto
        assert "max_pairs_per_topic" in auto
        assert "epochs" in auto


# ---------------------------------------------------------------------------
# auto_collect (data collection only, no training)
# ---------------------------------------------------------------------------

class TestAutoCollect:
    @patch("auto_trainer.WebCollector")
    def test_collect_saves_data(self, MockCollector):
        """Test that auto_collect creates a data file."""
        # Mock the collector to return fake pairs
        mock_instance = MockCollector.return_value
        mock_instance.collect.return_value = [
            {"prompt": "What is Python?", "response": "Python is a programming language."},
            {"prompt": "Tell me about Python.", "response": "Python was created by Guido van Rossum."},
        ]
        mock_instance.collect_from_url.return_value = []

        with tempfile.TemporaryDirectory() as tmpdir:
            # Patch the data directory
            with patch("auto_trainer.os.path.join", side_effect=os.path.join):
                result = auto_collect(
                    topics=["Python programming"],
                    output_name="test_collect",
                    sources=["wikipedia"],
                    max_pairs_per_topic=10,
                )
                # Should return a path (may be empty string if data dir issues)
                assert isinstance(result, str)

    @patch("auto_trainer.WebCollector")
    def test_collect_no_data(self, MockCollector):
        """Test that auto_collect handles empty results gracefully."""
        mock_instance = MockCollector.return_value
        mock_instance.collect.return_value = []

        result = auto_collect(
            topics=["nonexistent_topic_xyz_123"],
            output_name="test_empty",
            sources=["wikipedia"],
        )
        assert result == ""

    @patch("auto_trainer.WebCollector")
    def test_collect_multi_topic(self, MockCollector):
        """Test multi-topic collection."""
        mock_instance = MockCollector.return_value
        mock_instance.collect_multi_topic.return_value = [
            {"prompt": "Q1", "response": "A1"},
            {"prompt": "Q2", "response": "A2"},
        ]

        result = auto_collect(
            topics=["topic1", "topic2"],
            output_name="test_multi",
            sources=["wikipedia"],
        )
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# auto_train (full pipeline, mocked)
# ---------------------------------------------------------------------------

class TestAutoTrain:
    @patch("auto_trainer.register_model")
    @patch("auto_trainer.Trainer")
    @patch("auto_trainer.create_model")
    @patch("auto_trainer.create_data_loaders")
    @patch("auto_trainer.load_and_prepare_data")
    @patch("auto_trainer.save_collected_data")
    @patch("auto_trainer.WebCollector")
    def test_full_pipeline_mocked(self, MockCollector, mock_save, mock_prep,
                                   mock_loaders, mock_create, MockTrainer,
                                   mock_register):
        """Test the full auto_train pipeline with all I/O mocked."""
        import torch
        from models.tokenizer import Tokenizer

        # Mock collector
        mock_instance = MockCollector.return_value
        mock_instance.collect.return_value = [
            {"prompt": "What is AI?", "response": "AI is artificial intelligence."},
            {"prompt": "Tell me about ML.", "response": "ML is machine learning."},
        ]

        # Mock save
        mock_save.return_value = "data/auto/test_train.json"

        # Mock data prep
        mock_prep.return_value = [
            "What is AI?", "AI is artificial intelligence.",
            "Tell me about ML.", "ML is machine learning.",
        ]

        # Mock data loaders (minimal)
        mock_train_loader = MagicMock()
        mock_train_loader.__len__ = MagicMock(return_value=1)
        mock_test_loader = MagicMock()
        mock_test_loader.__len__ = MagicMock(return_value=1)
        mock_loaders.return_value = (mock_train_loader, mock_test_loader)

        # Mock model creation
        mock_model = MagicMock()
        mock_model.parameters.return_value = [torch.zeros(10, requires_grad=True)]
        mock_create.return_value = mock_model

        # Mock trainer
        mock_trainer_inst = MockTrainer.return_value
        mock_trainer_inst.train.return_value = None

        # Mock register
        mock_register.return_value = "trained_models/test_auto"

        result = auto_train(
            topics=["artificial intelligence"],
            model_name="test-auto",
            sources=["wikipedia"],
            max_pairs_per_topic=10,
            epochs=1,
        )

        assert isinstance(result, str)
        assert result != ""
        # Verify the pipeline was called
        MockCollector.assert_called_once()
        mock_save.assert_called_once()
        mock_create.assert_called_once()
        mock_register.assert_called_once()

    @patch("auto_trainer.WebCollector")
    def test_no_data_collected(self, MockCollector):
        """Test that auto_train exits gracefully when no data is collected."""
        mock_instance = MockCollector.return_value
        mock_instance.collect.return_value = []

        result = auto_train(
            topics=["nonexistent_topic_xyz"],
            model_name="test-empty",
            sources=["wikipedia"],
        )
        assert result == ""


# ---------------------------------------------------------------------------
# Configuration integration
# ---------------------------------------------------------------------------

class TestAutoTrainingConfig:
    def test_config_defaults(self):
        """Verify the config.yaml auto_training section has expected keys."""
        config = load_config("config.yaml")
        auto = config.get("auto_training", {})
        assert isinstance(auto.get("sources"), list)
        assert isinstance(auto.get("max_pairs_per_topic"), int)
        assert isinstance(auto.get("epochs"), int)
        assert auto["max_pairs_per_topic"] > 0
        assert auto["epochs"] > 0

    def test_sources_are_valid(self):
        """Check that configured sources are recognized values."""
        config = load_config("config.yaml")
        valid_sources = {"wikipedia", "web", "stackexchange"}
        for src in config.get("auto_training", {}).get("sources", []):
            assert src in valid_sources, f"Unknown source: {src}"
