"""Agent Framework — Tool-use agent, multi-step planning, memory system.

Features:
  - Tool registry and execution
  - Built-in tools: calculator, web search, code execution, file read
  - Plan-execute-reflect loop
  - Conversation memory (short-term buffer + long-term store)
"""

import os
import re
import json
import math
import time
from datetime import datetime
from typing import List, Dict, Optional, Callable, Any, Tuple


# ---------------------------------------------------------------------------
# Tool Registry
# ---------------------------------------------------------------------------

class Tool:
    """A callable tool the agent can use."""
    def __init__(self, name: str, description: str, func: Callable,
                 parameters: Optional[List[str]] = None):
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters or []

    def __call__(self, **kwargs) -> str:
        try:
            return str(self.func(**kwargs))
        except Exception as e:
            return f"[Tool Error: {e}]"


# Built-in tools
def tool_calculator(expression: str = "") -> str:
    """Evaluate a math expression safely."""
    allowed = set("0123456789+-*/().%^ ")
    expr = expression.replace("^", "**")
    if not all(c in allowed for c in expr):
        return "Invalid expression"
    try:
        result = eval(expr, {"__builtins__": {}}, {"math": math})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def tool_web_search(query: str = "") -> str:
    """Search the web using DuckDuckGo."""
    try:
        from utils.web_collector import search_duckduckgo
        results = search_duckduckgo(query, max_results=3)
        if results:
            return "\n".join(f"- {r['title']}: {r['snippet']}" for r in results)
        return "No results found"
    except Exception as e:
        return f"Search error: {e}"


def tool_read_file(path: str = "") -> str:
    """Read contents of a file."""
    if not os.path.exists(path):
        return f"File not found: {path}"
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(5000)
        return content
    except Exception as e:
        return f"Error: {e}"


def tool_list_files(directory: str = ".") -> str:
    """List files in a directory."""
    if not os.path.isdir(directory):
        return f"Directory not found: {directory}"
    files = os.listdir(directory)[:50]
    return "\n".join(files)


def tool_current_time() -> str:
    """Get current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def tool_wikipedia(topic: str = "") -> str:
    """Fetch a Wikipedia summary."""
    try:
        from utils.web_collector import fetch_wikipedia_article
        text = fetch_wikipedia_article(topic, max_chars=1000)
        return text if text else f"No article found for: {topic}"
    except Exception as e:
        return f"Wikipedia error: {e}"


DEFAULT_TOOLS = [
    Tool("calculator", "Evaluate math expressions", tool_calculator, ["expression"]),
    Tool("web_search", "Search the web for information", tool_web_search, ["query"]),
    Tool("read_file", "Read a file's contents", tool_read_file, ["path"]),
    Tool("list_files", "List files in a directory", tool_list_files, ["directory"]),
    Tool("current_time", "Get current date and time", tool_current_time, []),
    Tool("wikipedia", "Look up a topic on Wikipedia", tool_wikipedia, ["topic"]),
]


# ---------------------------------------------------------------------------
# Memory System
# ---------------------------------------------------------------------------

MEMORY_PATH = "data/agent_memory.json"


class MemorySystem:
    """Short-term (conversation) and long-term (persistent) memory."""

    def __init__(self, max_short_term: int = 20):
        self.short_term: List[Dict] = []  # Recent conversation
        self.max_short_term = max_short_term
        self.long_term: List[Dict] = []  # Persistent facts
        self._load_long_term()

    def _load_long_term(self):
        if os.path.exists(MEMORY_PATH):
            try:
                with open(MEMORY_PATH, 'r') as f:
                    self.long_term = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.long_term = []

    def _save_long_term(self):
        os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
        with open(MEMORY_PATH, 'w') as f:
            json.dump(self.long_term, f, indent=2)

    def add_message(self, role: str, content: str):
        self.short_term.append({"role": role, "content": content, "time": time.time()})
        if len(self.short_term) > self.max_short_term:
            self.short_term = self.short_term[-self.max_short_term:]

    def remember(self, fact: str, category: str = "general"):
        """Store a fact in long-term memory."""
        self.long_term.append({
            "fact": fact, "category": category,
            "timestamp": datetime.now().isoformat(),
        })
        self._save_long_term()

    def recall(self, query: str, top_k: int = 5) -> List[str]:
        """Recall relevant facts from long-term memory."""
        query_words = set(query.lower().split())
        scored = []
        for mem in self.long_term:
            fact_words = set(mem["fact"].lower().split())
            overlap = len(query_words & fact_words)
            if overlap > 0:
                scored.append((overlap, mem["fact"]))
        scored.sort(key=lambda x: -x[0])
        return [fact for _, fact in scored[:top_k]]

    def get_context(self, max_messages: int = 10) -> str:
        """Get recent conversation context as string."""
        recent = self.short_term[-max_messages:]
        return "\n".join(f"{m['role']}: {m['content']}" for m in recent)

    def clear_short_term(self):
        self.short_term = []

    def clear_long_term(self):
        self.long_term = []
        self._save_long_term()


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """Tool-use agent with planning and memory."""

    def __init__(self, tools: Optional[List[Tool]] = None, model=None,
                 tokenizer=None, config=None, device: str = "cpu"):
        self.tools = {t.name: t for t in (tools or DEFAULT_TOOLS)}
        self.memory = MemorySystem()
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device

    def _detect_tool_call(self, text: str) -> Optional[Tuple[str, Dict]]:
        """Detect if the user's message needs a tool call."""
        text_lower = text.lower()

        # Calculator
        if re.search(r'\d+\s*[\+\-\*/\^]\s*\d+', text):
            expr = re.search(r'[\d\+\-\*/\^\.\(\)\s]+', text)
            if expr:
                return ("calculator", {"expression": expr.group().strip()})

        # Web search
        if any(w in text_lower for w in ["search for", "look up", "find info", "search the web"]):
            query = re.sub(r'(search for|look up|find info on|search the web for)', '', text_lower).strip()
            return ("web_search", {"query": query})

        # Wikipedia
        if any(w in text_lower for w in ["what is", "who is", "tell me about", "define"]):
            topic = re.sub(r'(what is|who is|tell me about|define)\s*', '', text_lower).strip().rstrip('?.')
            if len(topic) > 2:
                return ("wikipedia", {"topic": topic})

        # Time
        if any(w in text_lower for w in ["what time", "current time", "what date", "today"]):
            return ("current_time", {})

        # File operations
        if "read file" in text_lower or "show file" in text_lower:
            match = re.search(r'(?:read|show)\s+file\s+(\S+)', text_lower)
            if match:
                return ("read_file", {"path": match.group(1)})

        if "list files" in text_lower:
            match = re.search(r'list files\s*(?:in\s+)?(\S+)?', text_lower)
            directory = match.group(1) if match and match.group(1) else "."
            return ("list_files", {"directory": directory})

        return None

    def process(self, user_input: str) -> str:
        """Process user input — detect tools, generate response."""
        self.memory.add_message("user", user_input)

        # Check for tool use
        tool_call = self._detect_tool_call(user_input)
        if tool_call:
            tool_name, params = tool_call
            tool = self.tools.get(tool_name)
            if tool:
                result = tool(**params)
                response = f"[Used {tool_name}]\n{result}"
                self.memory.add_message("assistant", response)
                return response

        # Check long-term memory
        memories = self.memory.recall(user_input, top_k=3)
        memory_context = ""
        if memories:
            memory_context = "\n[Recalled]: " + "; ".join(memories)

        # Generate with model if available
        if self.model and self.tokenizer and self.config:
            from chat import generate_response
            augmented = user_input
            if memory_context:
                augmented = f"{memory_context}\n{user_input}"
            response = generate_response(self.model, self.tokenizer,
                                          augmented, self.config, self.device)
        else:
            # Fallback
            response = f"I understand you're asking about: {user_input}"
            if memory_context:
                response += memory_context
            response += "\n(Load a model for AI-generated responses)"

        self.memory.add_message("assistant", response)
        return response

    def remember(self, fact: str):
        """Store a fact in long-term memory."""
        self.memory.remember(fact)
        return f"Remembered: {fact}"

    def list_tools(self) -> str:
        lines = ["Available tools:"]
        for name, tool in self.tools.items():
            params = ", ".join(tool.parameters) if tool.parameters else "none"
            lines.append(f"  {name}: {tool.description} (params: {params})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Interactive Agent Chat
# ---------------------------------------------------------------------------

def interactive_agent():
    """Interactive agent chat with tools."""
    print("\n" + "=" * 55)
    print("       AI Agent — Tool-Augmented Chat")
    print("=" * 55)
    print("\n  The agent can use tools automatically:")
    print("  • Calculator: ask math questions")
    print("  • Web search: 'search for <topic>'")
    print("  • Wikipedia: 'what is <topic>'")
    print("  • Files: 'read file <path>' or 'list files'")
    print("  • Time: 'what time is it'")
    print("  • Memory: 'remember <fact>' to store, auto-recalls")
    print("\n  Commands: 'tools', 'memory', 'clear', 'quit'\n")

    agent = Agent()

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Agent session ended.")
            break

        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit', 'stop', 'back'):
            break
        if user_input.lower() == 'tools':
            print(f"\n{agent.list_tools()}\n")
            continue
        if user_input.lower() == 'memory':
            memories = agent.memory.long_term[-10:]
            if memories:
                print("\n  Long-term memories:")
                for m in memories:
                    print(f"    [{m.get('category', '?')}] {m['fact']}")
            else:
                print("  No long-term memories stored.")
            print()
            continue
        if user_input.lower() == 'clear':
            agent.memory.clear_short_term()
            print("  [Conversation cleared]\n")
            continue
        if user_input.lower().startswith('remember '):
            fact = user_input[9:].strip()
            result = agent.remember(fact)
            print(f"  {result}\n")
            continue

        response = agent.process(user_input)
        print(f"\nAgent: {response}\n")


if __name__ == "__main__":
    interactive_agent()
