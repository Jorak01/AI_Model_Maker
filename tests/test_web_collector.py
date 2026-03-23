"""Tests for the WebCollector and public data fetching utilities."""

import os
import sys
import json
import tempfile
import shutil
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.web_collector import (
    _clean_text,
    _split_sentences,
    _cache_key,
    text_to_qa_pairs,
    sections_to_qa_pairs,
    search_results_to_pairs,
    stackexchange_to_pairs,
    save_collected_data,
    search_wikipedia,
    fetch_wikipedia_article,
    fetch_wikipedia_sections,
    search_duckduckgo,
    fetch_url_text,
    WebCollector,
)


# ---------------------------------------------------------------------------
# Helper text processing
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_strips_html_tags(self):
        assert "Hello World" in _clean_text("<p>Hello</p> <b>World</b>")

    def test_removes_references(self):
        assert "[1]" not in _clean_text("Some fact[1] and another[2].")

    def test_collapses_whitespace(self):
        result = _clean_text("  too   many    spaces  ")
        assert "  " not in result
        assert result == "too many spaces"

    def test_empty_string(self):
        assert _clean_text("") == ""


class TestSplitSentences:
    def test_basic_split(self):
        text = "First sentence. Second sentence. Third one here."
        sentences = _split_sentences(text)
        assert len(sentences) >= 2

    def test_filters_short(self):
        text = "Hi. This is a longer sentence that should pass the filter."
        sentences = _split_sentences(text)
        # "Hi." is too short (< 10 chars), should be filtered
        for s in sentences:
            assert len(s) > 10

    def test_empty_text(self):
        assert _split_sentences("") == []


class TestCacheKey:
    def test_deterministic(self):
        assert _cache_key("http://example.com") == _cache_key("http://example.com")

    def test_different_urls(self):
        assert _cache_key("http://a.com") != _cache_key("http://b.com")

    def test_length(self):
        key = _cache_key("http://example.com/some/long/path?query=value")
        assert len(key) == 24


# ---------------------------------------------------------------------------
# Text → Training Pairs
# ---------------------------------------------------------------------------

class TestTextToQaPairs:
    def test_basic_conversion(self):
        text = (
            "Machine learning is a subset of artificial intelligence. "
            "It focuses on building systems that learn from data. "
            "Deep learning is a specialized form of machine learning. "
            "It uses neural networks with many layers to model complex patterns."
        )
        pairs = text_to_qa_pairs(text, topic="machine learning")
        assert len(pairs) > 0
        assert all("prompt" in p and "response" in p for p in pairs)

    def test_with_topic(self):
        text = "Python is a programming language. It was created by Guido van Rossum."
        pairs = text_to_qa_pairs(text, topic="Python")
        assert len(pairs) > 0
        assert any("Python" in p["prompt"] for p in pairs)

    def test_without_topic(self):
        text = "The sun is a star. It provides light and heat to Earth."
        pairs = text_to_qa_pairs(text)
        assert len(pairs) >= 0  # May or may not produce pairs depending on length

    def test_empty_text(self):
        pairs = text_to_qa_pairs("")
        assert pairs == []

    def test_short_text_filtered(self):
        pairs = text_to_qa_pairs("Too short.")
        assert pairs == []

    def test_chunk_size(self):
        # Longer text should produce multiple chunks
        long_text = ". ".join([f"Sentence number {i} with enough words to pass the filter"
                               for i in range(20)])
        pairs = text_to_qa_pairs(long_text, topic="test", chunk_size=100)
        assert len(pairs) > 1


class TestSectionsToQaPairs:
    def test_basic_sections(self):
        sections = [
            {"heading": "Introduction", "text": "This is a long introduction about the topic. " * 5},
            {"heading": "History", "text": "The history goes back many years. " * 5},
        ]
        pairs = sections_to_qa_pairs(sections, topic="Test Subject")
        assert len(pairs) > 0

    def test_short_sections_filtered(self):
        sections = [
            {"heading": "Short", "text": "Too short."},
        ]
        pairs = sections_to_qa_pairs(sections)
        assert pairs == []

    def test_empty_sections(self):
        pairs = sections_to_qa_pairs([])
        assert pairs == []


class TestSearchResultsToPairs:
    def test_basic_conversion(self):
        results = [
            {"title": "Python Programming", "url": "http://example.com",
             "snippet": "Python is a versatile programming language used worldwide."},
            {"title": "Java Language", "url": "http://example.com/java",
             "snippet": "Java is an object-oriented programming language."},
        ]
        pairs = search_results_to_pairs(results)
        assert len(pairs) == 2
        assert "Python Programming" in pairs[0]["prompt"]

    def test_empty_snippets_filtered(self):
        results = [
            {"title": "Test", "url": "http://example.com", "snippet": ""},
        ]
        pairs = search_results_to_pairs(results)
        assert len(pairs) == 0

    def test_short_snippets_filtered(self):
        results = [
            {"title": "Test", "url": "http://example.com", "snippet": "Short"},
        ]
        pairs = search_results_to_pairs(results)
        assert len(pairs) == 0


class TestStackexchangeToPairs:
    def test_basic_conversion(self):
        qa = [
            {"question": "How to sort a list in Python?",
             "answer": "You can use the sorted() function or the .sort() method."},
        ]
        pairs = stackexchange_to_pairs(qa)
        assert len(pairs) == 1
        assert "sort" in pairs[0]["prompt"].lower()

    def test_empty_answer_filtered(self):
        qa = [{"question": "Test?", "answer": ""}]
        pairs = stackexchange_to_pairs(qa)
        assert len(pairs) == 0


# ---------------------------------------------------------------------------
# Save collected data
# ---------------------------------------------------------------------------

class TestSaveCollectedData:
    def test_save_new_file(self):
        pairs = [{"prompt": "Q1", "response": "A1"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.json")
            result = save_collected_data(pairs, path, merge_existing=False)
            assert os.path.exists(result)
            with open(result, "r") as f:
                data = json.load(f)
            assert len(data) == 1

    def test_merge_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.json")
            # Save initial
            save_collected_data([{"prompt": "Q1", "response": "A1"}], path, merge_existing=False)
            # Merge
            save_collected_data([{"prompt": "Q2", "response": "A2"}], path, merge_existing=True)
            with open(path, "r") as f:
                data = json.load(f)
            assert len(data) == 2

    def test_no_merge(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "data.json")
            save_collected_data([{"prompt": "Q1", "response": "A1"}], path)
            save_collected_data([{"prompt": "Q2", "response": "A2"}], path, merge_existing=False)
            with open(path, "r") as f:
                data = json.load(f)
            assert len(data) == 1  # overwritten


# ---------------------------------------------------------------------------
# WebCollector (with mocked HTTP)
# ---------------------------------------------------------------------------

class TestWebCollector:
    def test_creation(self):
        collector = WebCollector(verbose=False)
        assert collector.verbose is False

    def test_add_unique_deduplicates(self):
        collector = WebCollector(verbose=False)
        pairs = []
        new1 = [{"prompt": "Q1", "response": "Answer one"}]
        new2 = [{"prompt": "Q2", "response": "Answer one"}]  # same response
        collector._add_unique(pairs, new1)
        collector._add_unique(pairs, new2)
        assert len(pairs) == 1  # deduplicated

    def test_add_unique_keeps_different(self):
        collector = WebCollector(verbose=False)
        pairs = []
        new1 = [{"prompt": "Q1", "response": "Answer one about topic A"}]
        new2 = [{"prompt": "Q2", "response": "Answer two about topic B"}]
        collector._add_unique(pairs, new1)
        collector._add_unique(pairs, new2)
        assert len(pairs) == 2

    @patch("utils.web_collector._safe_get")
    def test_collect_from_wikipedia_mocked(self, mock_get):
        # Mock Wikipedia search response
        search_response = json.dumps(["test", ["Test article"]])
        # Mock article content
        article_response = json.dumps({
            "query": {
                "pages": {
                    "123": {
                        "extract": (
                            "Test article is about testing software. "
                            "Software testing is important for quality. " * 10
                        )
                    }
                }
            }
        })
        mock_get.side_effect = [search_response, article_response]

        collector = WebCollector(verbose=False)
        pairs = collector.collect_from_wikipedia("test", max_articles=1)
        assert isinstance(pairs, list)

    @patch("utils.web_collector._safe_get")
    def test_collect_returns_list(self, mock_get):
        mock_get.return_value = json.dumps(["test", []])
        collector = WebCollector(verbose=False)
        pairs = collector.collect("nonexistent topic xyz", max_pairs=10,
                                  sources=["wikipedia"])
        assert isinstance(pairs, list)

    def test_collect_multi_topic(self):
        """Test multi-topic collection returns a list (may be empty without network)."""
        collector = WebCollector(verbose=False)
        with patch.object(collector, "collect", return_value=[
            {"prompt": "Q", "response": "A"}
        ]):
            pairs = collector.collect_multi_topic(["topic1", "topic2"],
                                                   max_pairs_per_topic=5)
            assert isinstance(pairs, list)
            assert len(pairs) == 2  # 1 per topic


# ---------------------------------------------------------------------------
# Wikipedia functions (mocked)
# ---------------------------------------------------------------------------

class TestWikipediaFunctions:
    @patch("utils.web_collector._safe_get")
    def test_search_wikipedia(self, mock_get):
        mock_get.return_value = json.dumps([
            "python",
            ["Python (programming language)", "Python (genus)"],
            ["", ""],
            ["https://en.wikipedia.org/wiki/Python_(programming_language)",
             "https://en.wikipedia.org/wiki/Python_(genus)"]
        ])
        titles = search_wikipedia("python", max_results=2)
        assert len(titles) == 2
        assert "Python (programming language)" in titles

    @patch("utils.web_collector._safe_get")
    def test_search_wikipedia_empty(self, mock_get):
        mock_get.return_value = None
        titles = search_wikipedia("nonexistent")
        assert titles == []

    @patch("utils.web_collector._safe_get")
    def test_fetch_wikipedia_article(self, mock_get):
        mock_get.return_value = json.dumps({
            "query": {
                "pages": {
                    "123": {
                        "title": "Test",
                        "extract": "This is a test article about testing."
                    }
                }
            }
        })
        text = fetch_wikipedia_article("Test")
        assert "test article" in text.lower()

    @patch("utils.web_collector._safe_get")
    def test_fetch_wikipedia_article_not_found(self, mock_get):
        mock_get.return_value = None
        text = fetch_wikipedia_article("Nonexistent Page")
        assert text == ""

    @patch("utils.web_collector._safe_get")
    def test_fetch_wikipedia_sections(self, mock_get):
        article_text = (
            "Introduction paragraph.\n\n"
            "== History ==\n\nHistory content goes here and is long enough. " * 3 + "\n\n"
            "== Usage ==\n\nUsage content goes here and is long enough. " * 3
        )
        mock_get.return_value = json.dumps({
            "query": {"pages": {"1": {"extract": article_text}}}
        })
        sections = fetch_wikipedia_sections("Test")
        assert isinstance(sections, list)


# ---------------------------------------------------------------------------
# DuckDuckGo (mocked)
# ---------------------------------------------------------------------------

class TestDuckDuckGo:
    @patch("utils.web_collector._safe_get")
    def test_search_returns_list(self, mock_get):
        mock_get.return_value = '<a class="result__a" href="http://example.com">Example</a>'
        results = search_duckduckgo("test")
        assert isinstance(results, list)

    @patch("utils.web_collector._safe_get")
    def test_search_no_results(self, mock_get):
        mock_get.return_value = None
        results = search_duckduckgo("test")
        assert results == []


# ---------------------------------------------------------------------------
# URL fetching (mocked)
# ---------------------------------------------------------------------------

class TestFetchUrl:
    @patch("utils.web_collector._safe_get")
    def test_fetch_html(self, mock_get):
        mock_get.return_value = "<html><body><p>Hello world</p></body></html>"
        text = fetch_url_text("http://example.com")
        assert "Hello world" in text

    @patch("utils.web_collector._safe_get")
    def test_fetch_json(self, mock_get):
        mock_get.return_value = '{"key": "value"}'
        text = fetch_url_text("http://api.example.com/data")
        assert "key" in text
        assert "value" in text

    @patch("utils.web_collector._safe_get")
    def test_fetch_failure(self, mock_get):
        mock_get.return_value = None
        text = fetch_url_text("http://bad-url.example.com")
        assert text == ""
