"""Tests for tutorial.py — tutorial steps and structure."""

import pytest
from docs.tutorial import STEPS, run_tutorial


class TestTutorialSteps:
    def test_steps_is_list(self):
        assert isinstance(STEPS, list)

    def test_has_multiple_steps(self):
        assert len(STEPS) >= 10

    def test_each_step_is_tuple(self):
        for step in STEPS:
            assert isinstance(step, tuple)
            assert len(step) == 2

    def test_each_step_has_title_and_content(self):
        for title, content in STEPS:
            assert isinstance(title, str)
            assert isinstance(content, str)
            assert len(title) > 0
            assert len(content) > 0

    def test_first_step_is_welcome(self):
        assert "Welcome" in STEPS[0][0] or "welcome" in STEPS[0][0].lower()

    def test_last_step_is_conclusion(self):
        last_title = STEPS[-1][0].lower()
        assert "next" in last_title or "complete" in last_title or "what" in last_title

    def test_steps_cover_key_topics(self):
        all_content = " ".join(content for _, content in STEPS)
        assert "train" in all_content.lower()
        assert "chat" in all_content.lower()
        assert "rag" in all_content.lower()
        assert "agent" in all_content.lower()
        assert "plugin" in all_content.lower()

    def test_steps_are_numbered_or_named(self):
        # Each step should have a meaningful title
        for title, _ in STEPS:
            assert len(title) >= 5

    def test_no_empty_content(self):
        for _, content in STEPS:
            # Content should have real text, not just whitespace
            assert len(content.strip()) > 20
