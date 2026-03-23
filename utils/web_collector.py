"""Web Collector — Search and fetch public domain information for training data.

Collects text from public sources (Wikipedia, web search, direct URLs) and
converts it into prompt/response training pairs automatically.
"""

import re
import json
import os
import time
import hashlib
import logging
from typing import List, Dict, Optional, Tuple
from urllib.parse import quote_plus, urlparse, urljoin

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; AIModelTrainer/1.0; "
        "+https://github.com/Jorak01/AI_Model)"
    ),
    "Accept": "text/html,application/json,text/plain;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

CACHE_DIR = os.path.join("data", ".web_cache")

# Public domain / open-access APIs that do NOT require keys
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
DUCKDUCKGO_HTML = "https://html.duckduckgo.com/html/"
STACKEXCHANGE_API = "https://api.stackexchange.com/2.3"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cache_key(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:24]


def _get_cached(url: str, max_age: int = 86400) -> Optional[str]:
    """Return cached response text if fresh enough, else None."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, _cache_key(url) + ".txt")
    if os.path.exists(path):
        age = time.time() - os.path.getmtime(path)
        if age < max_age:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    return None


def _put_cache(url: str, text: str):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, _cache_key(url) + ".txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _safe_get(url: str, params: Optional[dict] = None,
              timeout: int = 15, cache: bool = True) -> Optional[str]:
    """HTTP GET with caching, retries, and polite rate-limiting."""
    full = url + ("?" + "&".join(f"{k}={v}" for k, v in params.items()) if params else "")
    if cache:
        cached = _get_cached(full)
        if cached is not None:
            return cached

    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, headers=DEFAULT_HEADERS,
                                timeout=timeout)
            resp.raise_for_status()
            text = resp.text
            if cache:
                _put_cache(full, text)
            time.sleep(0.5)  # polite delay
            return text
        except requests.RequestException as exc:
            logger.warning("GET %s attempt %d failed: %s", url, attempt + 1, exc)
            time.sleep(1.5 * (attempt + 1))
    return None


def _clean_text(raw: str) -> str:
    """Strip HTML tags, collapse whitespace, and basic cleanup."""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", raw)
    # Remove references like [1], [2], etc.
    text = re.sub(r"\[\d+\]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences (simple heuristic)."""
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in parts if len(s.strip()) > 10]


# ---------------------------------------------------------------------------
# Wikipedia
# ---------------------------------------------------------------------------

def search_wikipedia(query: str, max_results: int = 5) -> List[str]:
    """Search Wikipedia and return a list of page titles."""
    params = {
        "action": "opensearch",
        "search": query,
        "limit": str(max_results),
        "namespace": "0",
        "format": "json",
    }
    raw = _safe_get(WIKIPEDIA_API, params)
    if not raw:
        return []
    try:
        data = json.loads(raw)
        return data[1] if len(data) > 1 else []
    except (json.JSONDecodeError, IndexError):
        return []


def fetch_wikipedia_article(title: str, max_chars: int = 15000) -> str:
    """Fetch the plain-text extract of a Wikipedia article."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": "1",
        "exlimit": "1",
        "format": "json",
    }
    raw = _safe_get(WIKIPEDIA_API, params)
    if not raw:
        return ""
    try:
        data = json.loads(raw)
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            extract = page.get("extract", "")
            if extract:
                return extract[:max_chars]
    except (json.JSONDecodeError, KeyError):
        pass
    return ""


def fetch_wikipedia_sections(title: str) -> List[Dict[str, str]]:
    """Fetch article split into sections (heading + text)."""
    full_text = fetch_wikipedia_article(title, max_chars=50000)
    if not full_text:
        return []

    sections = []
    # Split on section headers (== Header ==)
    parts = re.split(r'\n\s*(={2,})\s*(.+?)\s*\1\s*\n', full_text)

    # First part is the intro
    if parts[0].strip():
        sections.append({"heading": title, "text": parts[0].strip()})

    # Remaining are (=level=, heading, text) triples
    i = 1
    while i + 2 <= len(parts):
        heading = parts[i + 1].strip() if i + 1 < len(parts) else ""
        text = parts[i + 2].strip() if i + 2 < len(parts) else ""
        if heading and text and len(text) > 50:
            sections.append({"heading": heading, "text": text})
        i += 3

    return sections


# ---------------------------------------------------------------------------
# DuckDuckGo (HTML version — no API key needed)
# ---------------------------------------------------------------------------

def search_duckduckgo(query: str, max_results: int = 8) -> List[Dict[str, str]]:
    """Search DuckDuckGo and return list of {title, url, snippet}."""
    params = {"q": query}
    raw = _safe_get(DUCKDUCKGO_HTML, params, cache=True)
    if not raw:
        return []

    results = []
    # Parse result links from HTML
    # DuckDuckGo HTML results have class="result__a" for links
    link_pattern = re.compile(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL
    )
    snippet_pattern = re.compile(
        r'class="result__snippet"[^>]*>(.*?)</(?:a|td|div|span)', re.DOTALL
    )

    links = link_pattern.findall(raw)
    snippets = snippet_pattern.findall(raw)

    for i, (url, title) in enumerate(links[:max_results]):
        snippet = _clean_text(snippets[i]) if i < len(snippets) else ""
        clean_title = _clean_text(title)
        # DuckDuckGo wraps URLs through a redirect — extract actual URL
        actual_url = url
        if "uddg=" in url:
            match = re.search(r'uddg=([^&]+)', url)
            if match:
                from urllib.parse import unquote
                actual_url = unquote(match.group(1))
        if clean_title and actual_url:
            results.append({
                "title": clean_title,
                "url": actual_url,
                "snippet": snippet,
            })

    return results


# ---------------------------------------------------------------------------
# Generic URL fetching
# ---------------------------------------------------------------------------

def fetch_url_text(url: str, max_chars: int = 20000) -> str:
    """Fetch a URL and extract readable text content."""
    raw = _safe_get(url, timeout=20)
    if not raw:
        return ""

    # If it looks like JSON, return it formatted
    if raw.strip().startswith("{") or raw.strip().startswith("["):
        try:
            data = json.loads(raw)
            return json.dumps(data, indent=2)[:max_chars]
        except json.JSONDecodeError:
            pass

    # Extract text from HTML
    text = _clean_text(raw)
    return text[:max_chars]


# ---------------------------------------------------------------------------
# StackExchange (public, no key needed for read-only)
# ---------------------------------------------------------------------------

def search_stackexchange(query: str, site: str = "stackoverflow",
                         max_results: int = 5) -> List[Dict[str, str]]:
    """Search StackExchange and return Q&A pairs."""
    params = {
        "order": "desc",
        "sort": "relevance",
        "intitle": query,
        "site": site,
        "filter": "withbody",
        "pagesize": str(max_results),
    }
    raw = _safe_get(f"{STACKEXCHANGE_API}/search/advanced", params)
    if not raw:
        return []

    results = []
    try:
        data = json.loads(raw)
        for item in data.get("items", [])[:max_results]:
            title = _clean_text(item.get("title", ""))
            body = _clean_text(item.get("body", ""))
            if title and body:
                results.append({"question": title, "answer": body[:2000]})
    except (json.JSONDecodeError, KeyError):
        pass
    return results


# ---------------------------------------------------------------------------
# Text → Training Pairs Conversion
# ---------------------------------------------------------------------------

def text_to_qa_pairs(text: str, topic: str = "",
                     chunk_size: int = 300) -> List[Dict[str, str]]:
    """Convert a block of text into question/answer training pairs.

    Strategy:
    - Split text into paragraph-sized chunks
    - Generate factual Q&A pairs from each chunk
    - Use the topic to create contextual questions
    """
    pairs: List[Dict[str, str]] = []
    sentences = _split_sentences(text)
    if not sentences:
        return pairs

    # Group sentences into chunks
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for sent in sentences:
        current.append(sent)
        current_len += len(sent)
        if current_len >= chunk_size:
            chunks.append(" ".join(current))
            current = []
            current_len = 0
    if current:
        chunks.append(" ".join(current))

    # Question templates for generating diverse training pairs
    templates = [
        ("What is {topic}?", None),
        ("Tell me about {topic}.", None),
        ("Explain {topic}.", None),
        ("Can you describe {topic}?", None),
        ("What do you know about {topic}?", None),
        ("Give me information about {topic}.", None),
        ("How does {topic} work?", None),
        ("What are the key facts about {topic}?", None),
        ("Summarize {topic} for me.", None),
        ("What should I know about {topic}?", None),
    ]

    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) < 30:
            continue

        # Use rotating question templates
        template_idx = i % len(templates)
        question_template = templates[template_idx][0]

        # Create the prompt
        if topic:
            prompt = question_template.format(topic=topic)
        else:
            # Extract a topic-like phrase from the chunk
            words = chunk.split()[:5]
            inferred_topic = " ".join(words)
            prompt = question_template.format(topic=inferred_topic)

        pairs.append({
            "prompt": prompt,
            "response": chunk.strip(),
        })

    return pairs


def sections_to_qa_pairs(sections: List[Dict[str, str]],
                         topic: str = "") -> List[Dict[str, str]]:
    """Convert article sections into Q&A training pairs."""
    pairs: List[Dict[str, str]] = []
    for section in sections:
        heading = section.get("heading", "")
        text = section.get("text", "")
        if not text or len(text) < 50:
            continue

        # Create diverse question forms from the heading
        questions = [
            f"What is {heading}?",
            f"Tell me about {heading}.",
            f"Explain {heading} in the context of {topic}." if topic else f"Explain {heading}.",
        ]

        # Use each sentence grouping as a response
        sub_pairs = text_to_qa_pairs(text, topic=heading)
        if sub_pairs:
            pairs.extend(sub_pairs)
        else:
            # Fallback: use the full section text
            pairs.append({
                "prompt": questions[0],
                "response": text[:1500].strip(),
            })

    return pairs


def search_results_to_pairs(results: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Convert search result snippets into training pairs."""
    pairs = []
    for r in results:
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        if title and snippet and len(snippet) > 20:
            pairs.append({
                "prompt": f"What is {title}?",
                "response": snippet,
            })
    return pairs


def stackexchange_to_pairs(qa_items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Convert StackExchange Q&A into training pairs."""
    pairs = []
    for item in qa_items:
        q = item.get("question", "")
        a = item.get("answer", "")
        if q and a and len(a) > 20:
            pairs.append({"prompt": q, "response": a})
    return pairs


# ---------------------------------------------------------------------------
# High-Level Collector
# ---------------------------------------------------------------------------

class WebCollector:
    """Orchestrates searching and collecting training data from public sources.

    Usage:
        collector = WebCollector()
        pairs = collector.collect("machine learning", max_pairs=100)
    """

    def __init__(self, cache_age: int = 86400, verbose: bool = True):
        self.cache_age = cache_age
        self.verbose = verbose
        self._seen_responses: set = set()  # deduplicate

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [WebCollector] {msg}")

    def _add_unique(self, pairs: List[Dict[str, str]],
                    new_pairs: List[Dict[str, str]]):
        """Add only pairs with unique responses."""
        for p in new_pairs:
            key = p["response"][:100]
            if key not in self._seen_responses:
                self._seen_responses.add(key)
                pairs.append(p)

    def collect_from_wikipedia(self, query: str,
                               max_articles: int = 3) -> List[Dict[str, str]]:
        """Search Wikipedia and build training pairs from articles."""
        self._log(f"Searching Wikipedia for: '{query}'")
        titles = search_wikipedia(query, max_results=max_articles)
        self._log(f"  Found {len(titles)} articles: {titles}")

        all_pairs: List[Dict[str, str]] = []
        for title in titles:
            self._log(f"  Fetching: {title}")
            sections = fetch_wikipedia_sections(title)
            if sections:
                pairs = sections_to_qa_pairs(sections, topic=title)
                self._add_unique(all_pairs, pairs)
                self._log(f"    → {len(pairs)} pairs from sections")
            else:
                text = fetch_wikipedia_article(title)
                if text:
                    pairs = text_to_qa_pairs(text, topic=title)
                    self._add_unique(all_pairs, pairs)
                    self._log(f"    → {len(pairs)} pairs from text")

        return all_pairs

    def collect_from_web_search(self, query: str,
                                max_results: int = 8,
                                fetch_pages: bool = True) -> List[Dict[str, str]]:
        """Search the web and build training pairs from results + page content."""
        self._log(f"Searching web for: '{query}'")
        results = search_duckduckgo(query, max_results=max_results)
        self._log(f"  Found {len(results)} search results")

        all_pairs: List[Dict[str, str]] = []

        # Pairs from search snippets
        snippet_pairs = search_results_to_pairs(results)
        self._add_unique(all_pairs, snippet_pairs)

        # Optionally fetch full page content
        if fetch_pages:
            for r in results[:5]:  # Limit to top 5 pages
                url = r.get("url", "")
                if not url or not url.startswith("http"):
                    continue
                # Skip non-text content
                skip_ext = (".pdf", ".jpg", ".png", ".gif", ".mp4", ".zip")
                if any(url.lower().endswith(ext) for ext in skip_ext):
                    continue
                self._log(f"  Fetching page: {url[:80]}...")
                text = fetch_url_text(url, max_chars=10000)
                if text and len(text) > 100:
                    topic = r.get("title", query)
                    pairs = text_to_qa_pairs(text, topic=topic)
                    self._add_unique(all_pairs, pairs)
                    self._log(f"    → {len(pairs)} pairs from page")

        return all_pairs

    def collect_from_stackexchange(self, query: str,
                                   site: str = "stackoverflow",
                                   max_results: int = 5) -> List[Dict[str, str]]:
        """Search StackExchange for Q&A training pairs."""
        self._log(f"Searching StackExchange ({site}) for: '{query}'")
        qa_items = search_stackexchange(query, site=site, max_results=max_results)
        pairs = stackexchange_to_pairs(qa_items)
        self._log(f"  Found {len(pairs)} Q&A pairs")
        return pairs

    def collect_from_url(self, url: str, topic: str = "") -> List[Dict[str, str]]:
        """Fetch a specific URL and build training pairs from its content."""
        self._log(f"Fetching URL: {url}")
        text = fetch_url_text(url, max_chars=20000)
        if not text or len(text) < 50:
            self._log("  No usable content found")
            return []
        pairs = text_to_qa_pairs(text, topic=topic)
        self._log(f"  → {len(pairs)} pairs from URL")
        return pairs

    def collect(self, query: str, max_pairs: int = 200,
                sources: Optional[List[str]] = None,
                urls: Optional[List[str]] = None) -> List[Dict[str, str]]:
        """Collect training data from multiple public sources.

        Args:
            query: The topic/search query to collect data about
            max_pairs: Maximum number of training pairs to collect
            sources: Which sources to use. Default: all.
                     Options: "wikipedia", "web", "stackexchange"
            urls: Additional specific URLs to fetch

        Returns:
            List of {"prompt": ..., "response": ...} training pairs
        """
        if sources is None:
            sources = ["wikipedia", "web"]

        all_pairs: List[Dict[str, str]] = []
        self._seen_responses.clear()

        # 1. Wikipedia (highest quality for factual data)
        if "wikipedia" in sources and len(all_pairs) < max_pairs:
            wiki_pairs = self.collect_from_wikipedia(query, max_articles=3)
            self._add_unique(all_pairs, wiki_pairs)
            self._log(f"  Wikipedia total: {len(wiki_pairs)} pairs")

        # 2. Web search
        if "web" in sources and len(all_pairs) < max_pairs:
            web_pairs = self.collect_from_web_search(query, fetch_pages=True)
            self._add_unique(all_pairs, web_pairs)
            self._log(f"  Web total: {len(web_pairs)} pairs")

        # 3. StackExchange (great for technical topics)
        if "stackexchange" in sources and len(all_pairs) < max_pairs:
            se_pairs = self.collect_from_stackexchange(query)
            self._add_unique(all_pairs, se_pairs)
            self._log(f"  StackExchange total: {len(se_pairs)} pairs")

        # 4. Specific URLs
        if urls:
            for url in urls:
                if len(all_pairs) >= max_pairs:
                    break
                url_pairs = self.collect_from_url(url, topic=query)
                self._add_unique(all_pairs, url_pairs)

        # Trim to max
        all_pairs = all_pairs[:max_pairs]
        self._log(f"\n  Total collected: {len(all_pairs)} unique training pairs")
        return all_pairs

    def collect_multi_topic(self, queries: List[str],
                            max_pairs_per_topic: int = 100,
                            sources: Optional[List[str]] = None) -> List[Dict[str, str]]:
        """Collect data across multiple topics/queries.

        Args:
            queries: List of topics to search for
            max_pairs_per_topic: Max pairs per individual topic
            sources: Which sources to use

        Returns:
            Combined list of training pairs from all topics
        """
        all_pairs: List[Dict[str, str]] = []
        for i, query in enumerate(queries, 1):
            self._log(f"\n{'='*50}")
            self._log(f"Topic {i}/{len(queries)}: {query}")
            self._log(f"{'='*50}")
            pairs = self.collect(query, max_pairs=max_pairs_per_topic,
                                 sources=sources)
            all_pairs.extend(pairs)
            self._log(f"Running total: {len(all_pairs)} pairs")

        self._log(f"\nAll topics complete: {len(all_pairs)} total pairs")
        return all_pairs


def save_collected_data(pairs: List[Dict[str, str]], output_path: str,
                        merge_existing: bool = True) -> str:
    """Save collected training pairs to a JSON file.

    Args:
        pairs: List of {"prompt": ..., "response": ...} dicts
        output_path: Path to save JSON
        merge_existing: If True, append to existing data

    Returns:
        Path to saved file
    """
    existing: List[Dict[str, str]] = []
    if merge_existing and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        print(f"  Merging with {len(existing)} existing pairs")

    combined = existing + pairs
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"  Saved {len(combined)} pairs to {output_path}")
    return output_path
