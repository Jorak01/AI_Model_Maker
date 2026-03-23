"""Tests for utils/curriculum.py — difficulty estimation, scheduling, domain mixing."""

import pytest
from utils.curriculum import (
    estimate_difficulty, sort_by_difficulty, assign_difficulty_levels,
    CurriculumScheduler, detect_domain, analyze_domain_distribution,
    mix_by_domain,
)


# ---------------------------------------------------------------------------
# Difficulty Estimation
# ---------------------------------------------------------------------------

class TestEstimateDifficulty:
    def test_empty_strings(self):
        assert estimate_difficulty("", "") == 0.0

    def test_simple_pair(self):
        d = estimate_difficulty("Hi", "Hello")
        assert 0.0 <= d <= 1.0

    def test_complex_pair_harder(self):
        simple = estimate_difficulty("Hi", "Hello")
        complex_ = estimate_difficulty(
            "Explain the theoretical underpinnings of quantum entanglement.",
            "Quantum entanglement is a phenomenon in quantum mechanics where "
            "two particles become interconnected and the quantum state of each "
            "particle cannot be described independently of the state of the others."
        )
        assert complex_ > simple

    def test_difficulty_range(self):
        d = estimate_difficulty("What is 2+2?", "4")
        assert 0.0 <= d <= 1.0

    def test_long_text_caps_at_one(self):
        long_text = " ".join(["word"] * 500)
        d = estimate_difficulty(long_text, long_text)
        assert d <= 1.0


class TestSortByDifficulty:
    def test_sorts_easy_first(self):
        data = [
            {"prompt": "Explain quantum physics in detail with mathematical formulations and proofs.",
             "response": "Quantum physics involves wave functions, Schrodinger equation, and probability amplitudes."},
            {"prompt": "Hi", "response": "Hello"},
        ]
        result = sort_by_difficulty(data)
        # Simple pair should come first
        assert result[0]["prompt"] == "Hi"

    def test_reverse_sorts_hard_first(self):
        data = [
            {"prompt": "Hi", "response": "Hello"},
            {"prompt": "Explain quantum physics.", "response": "Quantum physics is complex."},
        ]
        result = sort_by_difficulty(data, reverse=True)
        assert result[0]["prompt"] != "Hi"

    def test_empty_list(self):
        assert sort_by_difficulty([]) == []


class TestAssignDifficultyLevels:
    def test_assigns_levels(self):
        data = [{"prompt": f"q{i}", "response": f"a{i}" * (i + 1)} for i in range(10)]
        result = assign_difficulty_levels(data, num_levels=3)
        assert len(result) == 10
        assert all("_difficulty" in item for item in result)
        assert all("_level" in item for item in result)
        assert all(0 <= item["_level"] <= 2 for item in result)

    def test_single_item(self):
        data = [{"prompt": "Hi", "response": "Hello"}]
        result = assign_difficulty_levels(data, num_levels=5)
        assert len(result) == 1
        assert result[0]["_level"] == 0


# ---------------------------------------------------------------------------
# Curriculum Scheduler
# ---------------------------------------------------------------------------

class TestCurriculumScheduler:
    @pytest.fixture
    def sample_data(self):
        return [{"prompt": f"q{i}", "response": f"a{i}" * (i + 1)} for i in range(20)]

    def test_linear_strategy(self, sample_data):
        sched = CurriculumScheduler(sample_data, num_epochs=10, warmup_epochs=5, strategy="linear")
        # Early epoch should have fewer items
        early = sched.get_epoch_data(0)
        late = sched.get_epoch_data(9)
        assert len(early) <= len(late)

    def test_exponential_strategy(self, sample_data):
        sched = CurriculumScheduler(sample_data, num_epochs=10, warmup_epochs=5, strategy="exponential")
        data = sched.get_epoch_data(0)
        assert len(data) > 0

    def test_step_strategy(self, sample_data):
        sched = CurriculumScheduler(sample_data, num_epochs=10, warmup_epochs=5, strategy="step")
        data = sched.get_epoch_data(0)
        assert len(data) > 0

    def test_after_warmup_all_data(self, sample_data):
        sched = CurriculumScheduler(sample_data, num_epochs=10, warmup_epochs=3, strategy="linear")
        data = sched.get_epoch_data(5)
        assert len(data) == len(sample_data)

    def test_get_schedule_info(self, sample_data):
        sched = CurriculumScheduler(sample_data, num_epochs=5, warmup_epochs=3)
        info = sched.get_schedule_info()
        assert len(info) == 5
        assert all("epoch" in s and "fraction" in s and "items" in s for s in info)
        # Last epoch should use all data
        assert info[-1]["fraction"] == 1.0

    def test_unknown_strategy_uses_all(self, sample_data):
        sched = CurriculumScheduler(sample_data, num_epochs=5, warmup_epochs=3, strategy="unknown")
        frac = sched._get_fraction(0)
        assert frac == 1.0


# ---------------------------------------------------------------------------
# Domain Detection
# ---------------------------------------------------------------------------

class TestDetectDomain:
    def test_code_domain(self):
        assert detect_domain("Write a Python function to sort a list") == "code"

    def test_math_domain(self):
        assert detect_domain("Calculate the sum of these numbers using algebra") == "math"

    def test_science_domain(self):
        assert detect_domain("Explain the hypothesis about DNA and evolution") == "science"

    def test_history_domain(self):
        assert detect_domain("The ancient empire fell after the war in the century") == "history"

    def test_creative_domain(self):
        assert detect_domain("Write a creative story with interesting characters") == "creative"

    def test_chat_domain(self):
        assert detect_domain("Hello, how are you? Can you help me?") == "chat"

    def test_general_fallback(self):
        assert detect_domain("xyzzy foobar baz") == "general"


class TestAnalyzeDomainDistribution:
    def test_basic_distribution(self):
        data = [
            {"prompt": "Write Python code", "response": "def foo(): pass"},
            {"prompt": "Hello how are you", "response": "I'm fine thanks"},
        ]
        dist = analyze_domain_distribution(data)
        assert isinstance(dist, dict)
        assert sum(dist.values()) == 2


class TestMixByDomain:
    def test_equal_mixing(self):
        data = [
            {"prompt": "Write a Python function", "response": "def foo(): pass"},
            {"prompt": "Hello how are you", "response": "I'm good"},
            {"prompt": "Calculate the sum", "response": "The sum is 10"},
        ]
        mixed = mix_by_domain(data, target_size=3)
        assert len(mixed) <= 3

    def test_with_ratios(self):
        data = [
            {"prompt": "Write Python code", "response": "def foo(): pass"},
            {"prompt": "Write JavaScript code", "response": "function foo() {}"},
            {"prompt": "Hello", "response": "Hi"},
        ]
        mixed = mix_by_domain(data, ratios={"code": 0.8, "chat": 0.2}, target_size=10)
        assert len(mixed) <= 10

    def test_empty_data(self):
        mixed = mix_by_domain([], target_size=5)
        assert mixed == []
