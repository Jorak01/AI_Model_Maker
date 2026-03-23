"""Tests for utils/eval_suite.py — metrics, scoring, eval logging."""

import os
import json
import math
import pytest
import tempfile
from utils.eval_suite import (
    calculate_perplexity, bleu_score, rouge_score, word_error_rate,
    log_eval, get_eval_history, EVAL_LOG_PATH,
)


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

class TestCalculatePerplexity:
    def test_zero_loss(self):
        assert calculate_perplexity([0.0]) == pytest.approx(1.0)

    def test_known_loss(self):
        assert calculate_perplexity([1.0]) == pytest.approx(math.e, rel=1e-3)

    def test_empty_losses(self):
        assert calculate_perplexity([]) == float('inf')

    def test_multiple_losses(self):
        result = calculate_perplexity([1.0, 2.0, 3.0])
        expected = math.exp(2.0)  # average is 2.0
        assert result == pytest.approx(expected, rel=1e-3)

    def test_very_high_loss_capped(self):
        result = calculate_perplexity([1000.0])
        # Should cap at exp(100) instead of overflow
        assert result == pytest.approx(math.exp(100), rel=1e-3)


# ---------------------------------------------------------------------------
# BLEU Score
# ---------------------------------------------------------------------------

class TestBleuScore:
    def test_identical_strings(self):
        scores = bleu_score("the cat sat on the mat", "the cat sat on the mat")
        assert scores["bleu-1"] == 1.0
        assert scores["bleu"] > 0.9

    def test_completely_different(self):
        scores = bleu_score("hello world", "foo bar baz qux")
        assert scores["bleu-1"] == 0.0
        assert scores["bleu"] == 0.0

    def test_partial_overlap(self):
        scores = bleu_score("the cat sat on the mat", "the cat is on the floor")
        assert 0.0 < scores["bleu-1"] < 1.0

    def test_empty_hypothesis(self):
        scores = bleu_score("hello world", "")
        assert scores["bleu-1"] == 0.0

    def test_empty_reference(self):
        scores = bleu_score("", "hello world")
        assert scores["bleu-1"] == 0.0

    def test_returns_all_ngram_scores(self):
        scores = bleu_score("the cat sat on the mat", "the cat sat on the mat", max_n=4)
        assert "bleu-1" in scores
        assert "bleu-2" in scores
        assert "bleu-3" in scores
        assert "bleu-4" in scores
        assert "bleu" in scores


# ---------------------------------------------------------------------------
# ROUGE Score
# ---------------------------------------------------------------------------

class TestRougeScore:
    def test_identical_strings(self):
        scores = rouge_score("the cat sat on the mat", "the cat sat on the mat")
        assert scores["precision"] == 1.0
        assert scores["recall"] == 1.0
        assert scores["f1"] == 1.0

    def test_completely_different(self):
        scores = rouge_score("hello world", "foo bar")
        assert scores["f1"] == 0.0

    def test_partial_overlap(self):
        scores = rouge_score("the cat sat on the mat", "the cat is on the floor")
        assert 0.0 < scores["f1"] < 1.0

    def test_empty_strings(self):
        scores = rouge_score("", "")
        assert scores["f1"] == 0.0

    def test_returns_dict(self):
        scores = rouge_score("hello", "hello")
        assert "precision" in scores
        assert "recall" in scores
        assert "f1" in scores


# ---------------------------------------------------------------------------
# Word Error Rate
# ---------------------------------------------------------------------------

class TestWordErrorRate:
    def test_identical_strings(self):
        assert word_error_rate("the cat sat", "the cat sat") == 0.0

    def test_completely_different(self):
        wer = word_error_rate("hello world", "foo bar")
        assert wer > 0.0

    def test_empty_reference(self):
        wer = word_error_rate("", "hello")
        # Division by max(0, 1) = 1
        assert wer >= 0.0

    def test_insertions(self):
        wer = word_error_rate("hello", "hello world")
        assert wer > 0.0

    def test_deletions(self):
        wer = word_error_rate("hello world", "hello")
        assert wer > 0.0

    def test_substitutions(self):
        wer = word_error_rate("the cat", "the dog")
        assert wer == pytest.approx(0.5, rel=1e-2)


# ---------------------------------------------------------------------------
# Eval Logging
# ---------------------------------------------------------------------------

class TestEvalLogging:
    @pytest.fixture(autouse=True)
    def setup_teardown(self, tmp_path, monkeypatch):
        self.log_path = str(tmp_path / "eval_history.json")
        monkeypatch.setattr("utils.eval_suite.EVAL_LOG_PATH", self.log_path)

    def test_log_and_retrieve(self):
        log_eval("test_run", {"bleu": 0.5, "rouge": 0.6}, model_name="test-model")
        history = get_eval_history()
        assert len(history) == 1
        assert history[0]["name"] == "test_run"
        assert history[0]["model"] == "test-model"
        assert history[0]["metrics"]["bleu"] == 0.5

    def test_multiple_logs(self):
        log_eval("run1", {"bleu": 0.3})
        log_eval("run2", {"bleu": 0.7})
        history = get_eval_history()
        assert len(history) == 2

    def test_empty_history(self):
        history = get_eval_history()
        assert history == []

    def test_results_field_excluded(self):
        log_eval("run", {"bleu": 0.5, "results": [{"detail": "stuff"}]})
        history = get_eval_history()
        assert "results" not in history[0]["metrics"]
