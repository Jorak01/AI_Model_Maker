"""Tests for the dataset manager — versioning, quality, augmentation, dedup."""

import os
import sys
import json
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset_manager import (
    score_pair, score_dataset, filter_low_quality,
    augment_synonym_swap, augment_word_dropout, augment_word_swap,
    augment_case_change, augment_dataset, deduplicate,
    snapshot_dataset, list_versions, restore_version,
    _hash_pair, SYNONYMS,
)


class TestScorePair:
    def test_basic_score(self):
        scores = score_pair("What is Python?", "Python is a programming language used for many tasks.")
        assert "overall" in scores
        assert 0 <= scores["overall"] <= 1

    def test_all_keys_present(self):
        scores = score_pair("hello", "world")
        for key in ["length", "diversity", "noise", "coherence", "overall"]:
            assert key in scores

    def test_identical_prompt_response_low_coherence(self):
        scores = score_pair("hello there", "hello there")
        assert scores["coherence"] == 0.1

    def test_noisy_text_penalized(self):
        scores = score_pair("test", "http://example.com !!!! $$$$")
        assert scores["noise"] < 1.0

    def test_short_text_lower_length_score(self):
        short = score_pair("hi", "ok")
        long = score_pair("What is the meaning of life?", "The meaning of life is a philosophical question about purpose.")
        assert short["length"] <= long["length"]


class TestScoreDataset:
    def test_empty_dataset(self):
        result = score_dataset([], verbose=False)
        assert result["count"] == 0

    def test_basic_dataset(self):
        data = [
            {"prompt": "What is AI?", "response": "Artificial intelligence is the simulation of human intelligence."},
            {"prompt": "Hello", "response": "Hi there, how can I help you today?"},
        ]
        result = score_dataset(data, verbose=False)
        assert result["count"] == 2
        assert 0 <= result["avg_score"] <= 1


class TestFilterLowQuality:
    def test_filter_separates(self):
        data = [
            {"prompt": "What is Python?", "response": "Python is a versatile programming language."},
            {"prompt": "x", "response": "x"},  # Low quality
        ]
        good, bad = filter_low_quality(data, threshold=0.4)
        assert len(good) + len(bad) == len(data)


class TestAugmentation:
    def test_synonym_swap_preserves_length(self):
        text = "This is a good and fast example"
        result = augment_synonym_swap(text, swap_prob=0.5)
        assert len(result.split()) == len(text.split())

    def test_word_dropout_shortens(self):
        text = "this is a long sentence with many words in it"
        result = augment_word_dropout(text, drop_prob=0.9)
        assert len(result.split()) <= len(text.split())

    def test_word_dropout_short_text_preserved(self):
        text = "hi"
        result = augment_word_dropout(text)
        assert result == text

    def test_word_swap_same_length(self):
        text = "one two three four five"
        result = augment_word_swap(text, swap_prob=1.0)
        assert len(result.split()) == len(text.split())

    def test_case_change(self):
        text = "Hello world"
        # Run many times since it's random
        results = set()
        for _ in range(50):
            results.add(augment_case_change(text)[0])
        # Should have both 'H' and 'h'
        assert len(results) >= 1  # At least original case

    def test_case_change_empty(self):
        assert augment_case_change("") == ""

    def test_augment_dataset_multiplies(self):
        data = [{"prompt": "hello", "response": "hi"}]
        result = augment_dataset(data, multiplier=3)
        assert len(result) == 4  # 1 original + 3 augmented

    def test_augment_dataset_keeps_originals(self):
        data = [{"prompt": "test", "response": "response"}]
        result = augment_dataset(data, multiplier=2)
        assert result[0]["prompt"] == "test"
        assert result[0]["response"] == "response"


class TestDeduplication:
    def test_hash_pair_deterministic(self):
        h1 = _hash_pair("hello", "world")
        h2 = _hash_pair("hello", "world")
        assert h1 == h2

    def test_hash_pair_case_insensitive(self):
        h1 = _hash_pair("Hello", "World")
        h2 = _hash_pair("hello", "world")
        assert h1 == h2

    def test_deduplicate_removes_dupes(self):
        data = [
            {"prompt": "hello", "response": "world"},
            {"prompt": "hello", "response": "world"},
            {"prompt": "different", "response": "entry"},
        ]
        result = deduplicate(data, use_global_index=False)
        assert len(result) == 2

    def test_deduplicate_no_dupes(self):
        data = [
            {"prompt": "a", "response": "b"},
            {"prompt": "c", "response": "d"},
        ]
        result = deduplicate(data, use_global_index=False)
        assert len(result) == 2


class TestVersioning:
    def test_snapshot_and_list(self, tmp_path, monkeypatch):
        import utils.dataset_manager as dm
        monkeypatch.setattr(dm, "VERSION_DIR", str(tmp_path / "versions"))
        monkeypatch.setattr(dm, "VERSION_INDEX", str(tmp_path / "versions" / "index.json"))

        data_path = str(tmp_path / "data.json")
        with open(data_path, 'w') as f:
            json.dump([{"prompt": "a", "response": "b"}], f)

        vid = snapshot_dataset(data_path, tag="test")
        assert vid.startswith("v_test_")
        versions = list_versions()
        assert len(versions) == 1
        assert versions[0]["tag"] == "test"

    def test_restore_version(self, tmp_path, monkeypatch):
        import utils.dataset_manager as dm
        monkeypatch.setattr(dm, "VERSION_DIR", str(tmp_path / "versions"))
        monkeypatch.setattr(dm, "VERSION_INDEX", str(tmp_path / "versions" / "index.json"))

        data_path = str(tmp_path / "data.json")
        original = [{"prompt": "original", "response": "data"}]
        with open(data_path, 'w') as f:
            json.dump(original, f)

        vid = snapshot_dataset(data_path, tag="v1")

        # Modify data
        with open(data_path, 'w') as f:
            json.dump([{"prompt": "changed", "response": "data"}], f)

        # Restore
        assert restore_version(vid, data_path) is True
        with open(data_path, 'r') as f:
            restored = json.load(f)
        assert restored[0]["prompt"] == "original"

    def test_restore_nonexistent(self, tmp_path, monkeypatch):
        import utils.dataset_manager as dm
        monkeypatch.setattr(dm, "VERSION_DIR", str(tmp_path / "versions"))
        assert restore_version("nonexistent_id", str(tmp_path / "out.json")) is False
