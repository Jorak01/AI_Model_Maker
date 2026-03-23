"""Tests for utils/tag_manager.py — frequency, hierarchy, negatives."""

import os
import json
import pytest
from utils.tag_manager import (
    analyze_tag_frequency, find_rare_tags, find_overrepresented,
    TAG_HIERARCHY, get_tag_category, categorize_tags,
    suggest_negatives, NEGATIVE_PROMPT_SETS, display_frequency,
)


# ---------------------------------------------------------------------------
# Tag Frequency Analysis
# ---------------------------------------------------------------------------

class TestAnalyzeTagFrequency:
    def test_basic_frequency(self, tmp_path):
        data = [
            {"caption": "red, blue, green"},
            {"caption": "red, yellow"},
            {"caption": "red, blue"},
        ]
        path = str(tmp_path / "tags.json")
        with open(path, 'w') as f:
            json.dump(data, f)

        freq = analyze_tag_frequency(path)
        assert freq["red"] == 3
        assert freq["blue"] == 2
        assert freq["yellow"] == 1

    def test_tag_field_override(self, tmp_path):
        data = [{"tags": "cat, dog"}, {"tags": "cat, bird"}]
        path = str(tmp_path / "tags.json")
        with open(path, 'w') as f:
            json.dump(data, f)

        freq = analyze_tag_frequency(path, tag_field="tags")
        assert freq["cat"] == 2

    def test_missing_file(self):
        freq = analyze_tag_frequency("nonexistent.json")
        assert freq == {}

    def test_list_tags(self, tmp_path):
        data = [{"caption": ["red", "blue"]}, {"caption": ["red"]}]
        path = str(tmp_path / "tags.json")
        with open(path, 'w') as f:
            json.dump(data, f)

        freq = analyze_tag_frequency(path)
        assert freq["red"] == 2

    def test_dict_with_entries_key(self, tmp_path):
        data = {"entries": [{"caption": "a, b"}, {"caption": "a, c"}]}
        path = str(tmp_path / "tags.json")
        with open(path, 'w') as f:
            json.dump(data, f)

        freq = analyze_tag_frequency(path)
        assert freq["a"] == 2


class TestFindRareTags:
    def test_finds_rare(self):
        freq = {"common": 10, "rare1": 1, "rare2": 1, "medium": 3}
        rare = find_rare_tags(freq, threshold=2)
        assert "rare1" in rare
        assert "rare2" in rare
        assert "common" not in rare

    def test_no_rare(self):
        freq = {"a": 5, "b": 10}
        assert find_rare_tags(freq, threshold=2) == []


class TestFindOverrepresented:
    def test_finds_overrepresented(self):
        freq = {"dominant": 80, "small": 10, "tiny": 5}
        over = find_overrepresented(freq, ratio=0.3)
        assert "dominant" in over
        assert "small" not in over

    def test_empty(self):
        assert find_overrepresented({}) == []


# ---------------------------------------------------------------------------
# Tag Hierarchy
# ---------------------------------------------------------------------------

class TestTagHierarchy:
    def test_hierarchy_structure(self):
        assert "color" in TAG_HIERARCHY
        assert "lighting" in TAG_HIERARCHY
        assert "style" in TAG_HIERARCHY
        assert "subject" in TAG_HIERARCHY
        assert "quality" in TAG_HIERARCHY
        assert "composition" in TAG_HIERARCHY

    def test_get_tag_category_found(self):
        result = get_tag_category("red")
        assert result is not None
        assert result[0] == "color"
        assert result[1] == "warm"

    def test_get_tag_category_not_found(self):
        result = get_tag_category("xyzzy_nonexistent")
        assert result is None

    def test_categorize_tags(self):
        tags = ["red", "blue", "sunlight", "oil painting", "unknown_tag"]
        cats = categorize_tags(tags)
        assert "color" in cats
        assert "lighting" in cats
        assert "style" in cats
        assert "uncategorized" in cats

    def test_categorize_empty(self):
        assert categorize_tags([]) == {}


# ---------------------------------------------------------------------------
# Negative Prompts
# ---------------------------------------------------------------------------

class TestSuggestNegatives:
    def test_general_negatives(self):
        negs = suggest_negatives("general")
        assert len(negs) > 0
        assert "low quality" in negs or "blurry" in negs

    def test_portrait_negatives(self):
        negs = suggest_negatives("portrait")
        assert len(negs) > len(suggest_negatives("nonexistent_style"))

    def test_filters_existing(self):
        negs = suggest_negatives("general", existing_tags=["low quality", "blurry"])
        assert "low quality" not in negs
        assert "blurry" not in negs

    def test_unknown_style_returns_general(self):
        negs = suggest_negatives("nonexistent_style_xyz")
        # Should still return general negatives
        assert len(negs) > 0

    def test_anime_negatives(self):
        negs = suggest_negatives("anime")
        assert len(negs) > 0


class TestDisplayFrequency:
    def test_empty_no_crash(self, capsys):
        display_frequency({})
        captured = capsys.readouterr()
        assert "No tags found" in captured.out

    def test_basic_display(self, capsys):
        display_frequency({"tag1": 10, "tag2": 5}, top_n=2)
        captured = capsys.readouterr()
        assert "tag1" in captured.out
