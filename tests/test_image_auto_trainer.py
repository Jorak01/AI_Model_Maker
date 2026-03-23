"""Tests for the Image Auto Trainer — tag collection and dataset building."""

import os
import sys
import json
import tempfile
import random
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from image_auto_trainer import (
    STYLE_CATEGORIES, QUALITY_TAGS, NEGATIVE_QUALITY_TAGS,
    build_tag_captions, build_negative_prompts,
    _extract_tags_from_text, _extract_art_terms,
    collect_tags_from_danbooru, collect_tags_from_web,
    collect_tags_from_wikipedia, auto_collect_tags,
    fetch_danbooru_related_tags, fetch_danbooru_popular_tags,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_style_categories_not_empty(self):
        assert len(STYLE_CATEGORIES) > 0

    def test_all_styles_have_required_keys(self):
        for name, cfg in STYLE_CATEGORIES.items():
            assert "search_terms" in cfg, f"{name} missing search_terms"
            assert "wiki_topics" in cfg, f"{name} missing wiki_topics"
            assert "seed_tags" in cfg, f"{name} missing seed_tags"
            assert "danbooru_tags" in cfg, f"{name} missing danbooru_tags"
            assert len(cfg["seed_tags"]) > 0, f"{name} has empty seed_tags"

    def test_known_styles_exist(self):
        assert "anime" in STYLE_CATEGORIES
        assert "landscape" in STYLE_CATEGORIES
        assert "portrait" in STYLE_CATEGORIES
        assert "cyberpunk" in STYLE_CATEGORIES
        assert "fantasy" in STYLE_CATEGORIES
        assert "realistic" in STYLE_CATEGORIES
        assert "pixel-art" in STYLE_CATEGORIES

    def test_quality_tags_not_empty(self):
        assert len(QUALITY_TAGS) > 0
        assert len(NEGATIVE_QUALITY_TAGS) > 0


# ---------------------------------------------------------------------------
# Tag extraction helpers
# ---------------------------------------------------------------------------

class TestExtractTagsFromText:
    def test_extracts_comma_lists(self):
        text = "Tags: anime, fantasy, landscape, portrait, cyberpunk"
        tags = _extract_tags_from_text(text)
        assert len(tags) > 0

    def test_filters_short_tags(self):
        text = "a, b, c, hello world, test tag, another one"
        tags = _extract_tags_from_text(text)
        for t in tags:
            assert len(t) > 2

    def test_empty_text(self):
        assert _extract_tags_from_text("") == set()

    def test_no_comma_lists(self):
        text = "This is a plain paragraph with no comma-separated lists."
        tags = _extract_tags_from_text(text)
        assert isinstance(tags, set)


class TestExtractArtTerms:
    def test_extracts_style_terms(self):
        text = "The impressionist style is characterized by visible brushstrokes, open composition, and light."
        terms = _extract_art_terms(text)
        assert isinstance(terms, set)

    def test_extracts_technique_terms(self):
        text = "A technique called chiaroscuro was widely used in Renaissance painting."
        terms = _extract_art_terms(text)
        assert len(terms) > 0

    def test_empty_text(self):
        assert _extract_art_terms("") == set()


# ---------------------------------------------------------------------------
# Caption building
# ---------------------------------------------------------------------------

class TestBuildTagCaptions:
    def test_basic_generation(self):
        random.seed(42)
        tags = ["anime", "fantasy", "portrait", "landscape", "detailed"]
        entries = build_tag_captions(tags, "test", ["anime"], num_captions=10)
        assert len(entries) == 10
        assert all("caption" in e for e in entries)
        assert all("raw_tags" in e for e in entries)

    def test_includes_seed_tags(self):
        random.seed(42)
        tags = ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6"]
        seed = ["must_include"]
        entries = build_tag_captions(tags, "test", seed, num_captions=20)
        # Most entries should contain the seed tag
        has_seed = sum(1 for e in entries if "must_include" in e["caption"])
        assert has_seed > 0

    def test_with_quality_tags(self):
        random.seed(42)
        tags = ["test_tag_1", "test_tag_2", "test_tag_3"]
        entries = build_tag_captions(tags, "test", ["test_tag_1"],
                                     num_captions=5, include_quality=True)
        # Quality tags should appear in some entries
        all_captions = " ".join(e["caption"] for e in entries)
        has_quality = any(q in all_captions for q in QUALITY_TAGS)
        assert has_quality

    def test_without_quality_tags(self):
        random.seed(42)
        tags = ["unique_test_a", "unique_test_b", "unique_test_c"]
        entries = build_tag_captions(tags, "test", ["unique_test_a"],
                                     num_captions=5, include_quality=False)
        all_captions = " ".join(e["caption"] for e in entries)
        has_quality = any(q in all_captions for q in QUALITY_TAGS)
        assert not has_quality

    def test_empty_tags(self):
        entries = build_tag_captions([], "test", [], num_captions=5)
        assert entries == []

    def test_image_path_empty(self):
        random.seed(42)
        entries = build_tag_captions(["tag"], "test", ["tag"], num_captions=3)
        for e in entries:
            assert e["image_path"] == ""


class TestBuildNegativePrompts:
    def test_generates_entries(self):
        random.seed(42)
        entries = build_negative_prompts(num_entries=10)
        assert len(entries) == 10

    def test_negative_prefix(self):
        random.seed(42)
        entries = build_negative_prompts(num_entries=5)
        for e in entries:
            assert e["caption"].startswith("[negative]")

    def test_contains_negative_tags(self):
        random.seed(42)
        entries = build_negative_prompts(num_entries=5)
        for e in entries:
            assert any(t in e["raw_tags"] for t in NEGATIVE_QUALITY_TAGS)


# ---------------------------------------------------------------------------
# Danbooru API (mocked)
# ---------------------------------------------------------------------------

class TestDanbooruAPI:
    @patch("image_auto_trainer._safe_get")
    def test_fetch_related_tags(self, mock_get):
        mock_get.return_value = json.dumps({
            "related_tags": [
                {"tag": {"name": "blue_hair"}},
                {"tag": {"name": "school_uniform"}},
            ]
        })
        tags = fetch_danbooru_related_tags("1girl", limit=5)
        assert isinstance(tags, list)

    @patch("image_auto_trainer._safe_get")
    def test_fetch_related_empty(self, mock_get):
        mock_get.return_value = None
        tags = fetch_danbooru_related_tags("nonexistent")
        assert tags == []

    @patch("image_auto_trainer._safe_get")
    def test_fetch_popular_tags(self, mock_get):
        mock_get.return_value = json.dumps([
            {"name": "solo"}, {"name": "1girl"}, {"name": "highres"},
        ])
        tags = fetch_danbooru_popular_tags(category=0, limit=3)
        assert len(tags) == 3
        assert "solo" in tags

    @patch("image_auto_trainer._safe_get")
    def test_fetch_popular_empty(self, mock_get):
        mock_get.return_value = None
        tags = fetch_danbooru_popular_tags()
        assert tags == []


# ---------------------------------------------------------------------------
# Collection functions (mocked)
# ---------------------------------------------------------------------------

class TestCollectFromDanbooru:
    @patch("image_auto_trainer.fetch_danbooru_related_tags")
    def test_collects_tags(self, mock_fetch):
        mock_fetch.return_value = ["tag_a", "tag_b", "tag_c"]
        tags = collect_tags_from_danbooru(["anime"], max_tags=10, verbose=False)
        assert isinstance(tags, list)
        assert len(tags) > 0

    @patch("image_auto_trainer.fetch_danbooru_related_tags")
    def test_empty_result(self, mock_fetch):
        mock_fetch.return_value = []
        tags = collect_tags_from_danbooru(["nonexistent"], max_tags=10, verbose=False)
        assert tags == []


class TestCollectFromWeb:
    @patch("image_auto_trainer.fetch_url_text")
    @patch("image_auto_trainer.search_duckduckgo")
    def test_collects_tags(self, mock_search, mock_fetch):
        mock_search.return_value = [
            {"title": "Art Tags", "url": "http://example.com",
             "snippet": "anime, fantasy, landscape, portrait, detailed"}
        ]
        mock_fetch.return_value = "tag_one, tag_two, tag_three, tag_four"
        tags = collect_tags_from_web(["art prompts"], max_tags=10, verbose=False)
        assert isinstance(tags, list)

    @patch("image_auto_trainer.search_duckduckgo")
    def test_empty_results(self, mock_search):
        mock_search.return_value = []
        tags = collect_tags_from_web(["nonexistent"], max_tags=10, verbose=False)
        assert isinstance(tags, list)


class TestCollectFromWikipedia:
    @patch("image_auto_trainer.fetch_wikipedia_article")
    def test_collects_terms(self, mock_fetch):
        mock_fetch.return_value = (
            "The impressionist style is characterized by visible brushstrokes. "
            "This technique called plein air was widely used. "
            "Features such as bright colors, soft edges, and natural light."
        )
        tags = collect_tags_from_wikipedia(["Impressionism"], max_tags=20, verbose=False)
        assert isinstance(tags, list)

    @patch("image_auto_trainer.fetch_wikipedia_article")
    def test_empty_article(self, mock_fetch):
        mock_fetch.return_value = ""
        tags = collect_tags_from_wikipedia(["Nonexistent"], max_tags=10, verbose=False)
        assert tags == []


# ---------------------------------------------------------------------------
# auto_collect_tags (high-level, mocked)
# ---------------------------------------------------------------------------

class TestAutoCollectTags:
    @patch("image_auto_trainer.collect_tags_from_wikipedia")
    @patch("image_auto_trainer.collect_tags_from_web")
    @patch("image_auto_trainer.collect_tags_from_danbooru")
    def test_produces_dataset(self, mock_dan, mock_web, mock_wiki):
        mock_dan.return_value = ["danbooru_tag1", "danbooru_tag2"]
        mock_web.return_value = ["web_tag1", "web_tag2"]
        mock_wiki.return_value = ["wiki_term1"]

        result = auto_collect_tags(
            styles=["anime"],
            model_name="test-image",
            num_captions=10,
            sources=["danbooru", "web", "wikipedia"],
            verbose=False,
        )
        assert isinstance(result, str)
        # Should have created a file
        if result:
            assert os.path.exists(result)
            # Cleanup
            os.unlink(result)

    @patch("image_auto_trainer.collect_tags_from_danbooru")
    def test_custom_style(self, mock_dan):
        mock_dan.return_value = ["custom_tag"]
        result = auto_collect_tags(
            styles=["steampunk"],  # Not a preset
            model_name="test-custom",
            num_captions=5,
            sources=["danbooru"],
            verbose=False,
        )
        assert isinstance(result, str)
        if result:
            os.unlink(result)


# ---------------------------------------------------------------------------
# Import & integration
# ---------------------------------------------------------------------------

class TestImports:
    def test_import_image_auto_trainer(self):
        import image_auto_trainer
        assert hasattr(image_auto_trainer, 'STYLE_CATEGORIES')
        assert hasattr(image_auto_trainer, 'auto_collect_tags')
        assert hasattr(image_auto_trainer, 'auto_train_image')
        assert hasattr(image_auto_trainer, 'auto_image_train_interactive')
        assert hasattr(image_auto_trainer, 'build_tag_captions')

    def test_import_from_run(self):
        import run
        assert hasattr(run, 'cmd_auto_image')
