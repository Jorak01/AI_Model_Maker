"""Tests for the prompt trainer."""

import os
import sys
import json
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.prompt_trainer import save_training_data, _show_pairs, collect_prompts


class TestSaveTrainingData:
    def test_save_new_file(self, tmp_path):
        pairs = [
            {"prompt": "Hello", "response": "Hi there"},
            {"prompt": "How are you", "response": "Good"},
        ]
        path = str(tmp_path / "training.train.json")
        result = save_training_data(pairs, path, merge_existing=False)
        assert os.path.exists(result)
        with open(result, 'r') as f:
            data = json.load(f)
        assert len(data) == 2
        assert data[0]["prompt"] == "Hello"

    def test_merge_with_existing(self, tmp_path):
        # Create existing file
        path = str(tmp_path / "training.train.json")
        existing = [{"prompt": "Existing", "response": "Data"}]
        with open(path, 'w') as f:
            json.dump(existing, f)

        new_pairs = [{"prompt": "New", "response": "Pair"}]
        save_training_data(new_pairs, path, merge_existing=True)

        with open(path, 'r') as f:
            data = json.load(f)
        assert len(data) == 2
        assert data[0]["prompt"] == "Existing"
        assert data[1]["prompt"] == "New"

    def test_no_merge(self, tmp_path):
        path = str(tmp_path / "training.train.json")
        existing = [{"prompt": "Old", "response": "Data"}]
        with open(path, 'w') as f:
            json.dump(existing, f)

        new_pairs = [{"prompt": "New", "response": "Only"}]
        save_training_data(new_pairs, path, merge_existing=False)

        with open(path, 'r') as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["prompt"] == "New"

    def test_creates_directories(self, tmp_path):
        path = str(tmp_path / "sub" / "dir" / "training.train.json")
        pairs = [{"prompt": "Hi", "response": "Hello"}]
        save_training_data(pairs, path, merge_existing=False)
        assert os.path.exists(path)

    def test_empty_pairs(self, tmp_path):
        path = str(tmp_path / "empty.json")
        save_training_data([], path, merge_existing=False)
        with open(path, 'r') as f:
            data = json.load(f)
        assert data == []


class TestShowPairs:
    def test_show_empty(self, capsys):
        _show_pairs([])
        captured = capsys.readouterr()
        assert "No pairs" in captured.out

    def test_show_pairs(self, capsys):
        pairs = [
            {"prompt": "Hello", "response": "Hi there"},
            {"prompt": "What is AI", "response": "Artificial Intelligence"},
        ]
        _show_pairs(pairs)
        captured = capsys.readouterr()
        assert "Hello" in captured.out
        assert "Hi there" in captured.out
        assert "2" in captured.out

    def test_long_text_truncated(self, capsys):
        pairs = [{"prompt": "x" * 100, "response": "y" * 100}]
        _show_pairs(pairs)
        captured = capsys.readouterr()
        assert "..." in captured.out
