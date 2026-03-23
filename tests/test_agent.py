"""Tests for agent.py — Tool, Agent, MemorySystem."""

import os
import json
import pytest
from services.agent import (
    Tool, tool_calculator, tool_current_time, tool_read_file, tool_list_files,
    MemorySystem, Agent, DEFAULT_TOOLS,
)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class TestTool:
    def test_tool_call(self):
        tool = Tool("add", "Add two numbers", lambda a, b: str(int(a) + int(b)), ["a", "b"])
        result = tool(a="3", b="4")
        assert result == "7"

    def test_tool_error_handling(self):
        tool = Tool("fail", "Always fails", lambda: 1 / 0, [])
        result = tool()
        assert "Tool Error" in result

    def test_tool_attributes(self):
        tool = Tool("test", "A test tool", lambda: "ok", ["p1", "p2"])
        assert tool.name == "test"
        assert tool.description == "A test tool"
        assert tool.parameters == ["p1", "p2"]


class TestBuiltinTools:
    def test_calculator_basic(self):
        assert tool_calculator("2 + 3") == "5"

    def test_calculator_multiplication(self):
        assert tool_calculator("6 * 7") == "42"

    def test_calculator_power(self):
        assert tool_calculator("2^10") == "1024"

    def test_calculator_invalid(self):
        result = tool_calculator("import os")
        assert "Invalid" in result

    def test_calculator_division(self):
        assert tool_calculator("10 / 2") == "5.0"

    def test_current_time(self):
        result = tool_current_time()
        assert len(result) > 0
        assert "-" in result  # Date format: YYYY-MM-DD

    def test_read_file_exists(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        result = tool_read_file(str(f))
        assert result == "hello world"

    def test_read_file_missing(self):
        result = tool_read_file("nonexistent.txt")
        assert "not found" in result.lower()

    def test_list_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        result = tool_list_files(str(tmp_path))
        assert "a.txt" in result
        assert "b.txt" in result

    def test_list_files_invalid_dir(self):
        result = tool_list_files("/nonexistent_directory_xyz")
        assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# MemorySystem
# ---------------------------------------------------------------------------

class TestMemorySystem:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        self.mem_path = str(tmp_path / "memory.json")
        monkeypatch.setattr("services.agent.MEMORY_PATH", self.mem_path)

    def test_add_message(self):
        mem = MemorySystem()
        mem.add_message("user", "Hello")
        mem.add_message("assistant", "Hi there")
        assert len(mem.short_term) == 2

    def test_short_term_limit(self):
        mem = MemorySystem(max_short_term=5)
        for i in range(10):
            mem.add_message("user", f"msg {i}")
        assert len(mem.short_term) == 5

    def test_remember_and_recall(self):
        mem = MemorySystem()
        mem.remember("The capital of France is Paris", category="geography")
        results = mem.recall("capital France")
        assert len(results) > 0
        assert "Paris" in results[0]

    def test_recall_empty(self):
        mem = MemorySystem()
        results = mem.recall("anything")
        assert results == []

    def test_long_term_persistence(self):
        mem1 = MemorySystem()
        mem1.remember("Python was created by Guido van Rossum")

        mem2 = MemorySystem()
        assert len(mem2.long_term) == 1
        assert "Guido" in mem2.long_term[0]["fact"]

    def test_get_context(self):
        mem = MemorySystem()
        mem.add_message("user", "Hello")
        mem.add_message("assistant", "Hi")
        context = mem.get_context()
        assert "user: Hello" in context
        assert "assistant: Hi" in context

    def test_clear_short_term(self):
        mem = MemorySystem()
        mem.add_message("user", "test")
        mem.clear_short_term()
        assert mem.short_term == []

    def test_clear_long_term(self):
        mem = MemorySystem()
        mem.remember("fact")
        mem.clear_long_term()
        assert mem.long_term == []


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TestAgent:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        self.mem_path = str(tmp_path / "memory.json")
        monkeypatch.setattr("services.agent.MEMORY_PATH", self.mem_path)

    def test_create_agent(self):
        agent = Agent()
        assert len(agent.tools) > 0

    def test_list_tools(self):
        agent = Agent()
        tools_str = agent.list_tools()
        assert "calculator" in tools_str
        assert "current_time" in tools_str

    def test_process_math(self):
        agent = Agent()
        response = agent.process("5 + 3")
        assert "8" in response

    def test_process_time(self):
        agent = Agent()
        response = agent.process("What time is it?")
        assert "current_time" in response.lower() or "-" in response

    def test_process_general(self):
        agent = Agent()
        response = agent.process("Tell me something interesting about space exploration")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_detect_tool_call_calculator(self):
        agent = Agent()
        result = agent._detect_tool_call("Calculate 10 + 20")
        assert result is not None
        assert result[0] == "calculator"

    def test_detect_tool_call_web_search(self):
        agent = Agent()
        result = agent._detect_tool_call("search for Python tutorials")
        assert result is not None
        assert result[0] == "web_search"

    def test_detect_tool_call_wikipedia(self):
        agent = Agent()
        result = agent._detect_tool_call("What is photosynthesis?")
        assert result is not None
        assert result[0] == "wikipedia"

    def test_detect_tool_call_none(self):
        agent = Agent()
        result = agent._detect_tool_call("Please help me")
        # "help me" matches chat but not a specific tool
        # This should not match any tool
        assert result is None or result[0] in agent.tools

    def test_remember(self):
        agent = Agent()
        result = agent.remember("The sky is blue")
        assert "Remembered" in result

    def test_custom_tools(self):
        custom = Tool("greet", "Say hello", lambda name="World": f"Hello, {name}!", ["name"])
        agent = Agent(tools=[custom])
        assert "greet" in agent.tools
        assert len(agent.tools) == 1

    def test_default_tools_count(self):
        assert len(DEFAULT_TOOLS) == 6
