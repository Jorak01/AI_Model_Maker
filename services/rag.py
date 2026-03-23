"""RAG (Retrieval-Augmented Generation) — Document ingestion, vector search, context-aware chat.

Features:
  - Document ingestion (text, markdown, PDF-text)
  - Simple TF-IDF based vector store (no external deps)
  - Context-aware chat with source citations
  - Chunk management and search
"""

import os
import re
import json
import math
import hashlib
from typing import List, Dict, Optional, Tuple
from collections import Counter

RAG_DIR = "data/rag"
INDEX_PATH = os.path.join(RAG_DIR, "index.json")


# ---------------------------------------------------------------------------
# Document Processing
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.split()) > 20:
            chunks.append(chunk)
    return chunks


def load_document(file_path: str) -> str:
    """Load text content from a file."""
    if not os.path.exists(file_path):
        return ""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext in ('.txt', '.md', '.py', '.js', '.html', '.css', '.yaml', '.yml', '.json', '.csv'):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        elif ext == '.pdf':
            # Try basic text extraction
            try:
                import subprocess
                result = subprocess.run(['python', '-c',
                    f'import fitz; doc=fitz.open("{file_path}"); print("\\n".join(p.get_text() for p in doc))'],
                    capture_output=True, text=True, timeout=30)
                if result.stdout:
                    return result.stdout
            except Exception:
                pass
            with open(file_path, 'rb') as f:
                content = f.read()
            # Basic text extraction from PDF bytes
            text_parts = re.findall(rb'\((.*?)\)', content)
            return " ".join(p.decode('utf-8', errors='ignore') for p in text_parts)
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    except Exception as e:
        print(f"  ⚠ Error reading {file_path}: {e}")
        return ""


# ---------------------------------------------------------------------------
# Simple TF-IDF Vector Store
# ---------------------------------------------------------------------------

class SimpleVectorStore:
    """TF-IDF based document retrieval — no external dependencies."""

    def __init__(self):
        self.documents: List[Dict] = []  # {id, text, source, metadata}
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.tfidf_vectors: List[Dict[str, float]] = []

    def _tokenize(self, text: str) -> List[str]:
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return [w for w in text.split() if len(w) > 2]

    def add_document(self, text: str, source: str = "", metadata: Optional[Dict] = None):
        doc_id = hashlib.md5(text[:200].encode()).hexdigest()[:12]
        self.documents.append({
            "id": doc_id, "text": text,
            "source": source, "metadata": metadata or {},
        })

    def build_index(self):
        """Build TF-IDF index from all documents."""
        n_docs = len(self.documents)
        if n_docs == 0:
            return

        # Document frequencies
        df: Counter = Counter()
        doc_tokens = []
        for doc in self.documents:
            tokens = self._tokenize(doc["text"])
            doc_tokens.append(tokens)
            unique = set(tokens)
            for t in unique:
                df[t] += 1

        # IDF
        self.idf = {}
        for term, freq in df.items():
            self.idf[term] = math.log(n_docs / (1 + freq))

        # TF-IDF vectors
        self.tfidf_vectors = []
        for tokens in doc_tokens:
            tf: Counter = Counter(tokens)
            total = len(tokens) if tokens else 1
            vector = {}
            for term, count in tf.items():
                if term in self.idf:
                    vector[term] = (count / total) * self.idf[term]
            self.tfidf_vectors.append(vector)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[float, Dict]]:
        """Search for most relevant documents."""
        if not self.tfidf_vectors:
            self.build_index()

        query_tokens = self._tokenize(query)
        query_tf: Counter = Counter(query_tokens)
        total = len(query_tokens) if query_tokens else 1

        query_vector = {}
        for term, count in query_tf.items():
            if term in self.idf:
                query_vector[term] = (count / total) * self.idf[term]

        # Cosine similarity
        results = []
        for i, doc_vec in enumerate(self.tfidf_vectors):
            dot = sum(query_vector.get(t, 0) * doc_vec.get(t, 0) for t in query_vector)
            mag_q = math.sqrt(sum(v ** 2 for v in query_vector.values())) if query_vector else 1
            mag_d = math.sqrt(sum(v ** 2 for v in doc_vec.values())) if doc_vec else 1
            sim = dot / (mag_q * mag_d) if (mag_q * mag_d) > 0 else 0
            results.append((sim, self.documents[i]))

        results.sort(key=lambda x: -x[0])
        return results[:top_k]

    def save(self, path: str = INDEX_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {"documents": self.documents}
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str = INDEX_PATH):
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            self.documents = data.get("documents", [])
            self.build_index()


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """Full RAG pipeline — ingest, retrieve, generate with context."""

    def __init__(self):
        self.store = SimpleVectorStore()
        self.store.load()

    def ingest_file(self, file_path: str, chunk_size: int = 500):
        """Ingest a document file into the vector store."""
        text = load_document(file_path)
        if not text:
            print(f"  ❌ Could not load: {file_path}")
            return 0

        chunks = chunk_text(text, chunk_size=chunk_size)
        for chunk in chunks:
            self.store.add_document(chunk, source=file_path)

        self.store.build_index()
        self.store.save()
        print(f"  ✓ Ingested: {file_path} ({len(chunks)} chunks)")
        return len(chunks)

    def ingest_directory(self, dir_path: str, extensions: Optional[List[str]] = None):
        """Ingest all documents in a directory."""
        if extensions is None:
            extensions = ['.txt', '.md', '.py', '.json', '.yaml', '.csv']

        total = 0
        for root, _, files in os.walk(dir_path):
            for f in files:
                if any(f.endswith(ext) for ext in extensions):
                    path = os.path.join(root, f)
                    total += self.ingest_file(path)
        print(f"\n  Total: {total} chunks from {dir_path}")
        return total

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve relevant context for a query."""
        results = self.store.search(query, top_k=top_k)
        return [{"score": round(score, 4), "text": doc["text"][:500],
                 "source": doc["source"]}
                for score, doc in results if score > 0.01]

    def augment_prompt(self, query: str, top_k: int = 3) -> str:
        """Create an augmented prompt with retrieved context."""
        contexts = self.retrieve(query, top_k=top_k)
        if not contexts:
            return query

        context_text = "\n\n".join(
            f"[Source: {c['source']}]\n{c['text']}" for c in contexts
        )
        return (
            f"Use the following context to answer the question.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

    def chat_with_context(self, query: str, model=None, tokenizer=None,
                          config=None, device: str = "cpu") -> Tuple[str, List[Dict]]:
        """Chat with RAG context. Returns (response, sources)."""
        contexts = self.retrieve(query)
        augmented = self.augment_prompt(query)

        if model and tokenizer and config:
            from services.chat import generate_response
            response = generate_response(model, tokenizer, augmented, config, device)
        else:
            response = f"[RAG Context Retrieved]\n{augmented}"

        sources = [{"source": c["source"], "relevance": c["score"]} for c in contexts]
        return response, sources

    def get_stats(self) -> Dict:
        return {
            "total_documents": len(self.store.documents),
            "unique_sources": len(set(d["source"] for d in self.store.documents)),
            "index_built": len(self.store.tfidf_vectors) > 0,
        }


# ---------------------------------------------------------------------------
# Interactive
# ---------------------------------------------------------------------------

def interactive_rag():
    """Interactive RAG interface."""
    print("\n" + "=" * 55)
    print("       RAG — Retrieval-Augmented Generation")
    print("=" * 55)

    rag = RAGPipeline()
    stats = rag.get_stats()
    print(f"\n  Documents: {stats['total_documents']} chunks from {stats['unique_sources']} sources")

    while True:
        print("\n  Options:")
        print("  1  Ingest file")
        print("  2  Ingest directory")
        print("  3  Search/retrieve")
        print("  4  RAG-augmented query")
        print("  5  Show stats")
        print("  0  Back")

        try:
            choice = input("\n  rag>> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if choice in ('0', 'back', 'quit', 'q'):
            break
        try:
            if choice == '1':
                path = input("  File path: ").strip()
                if path:
                    rag.ingest_file(path)
            elif choice == '2':
                path = input("  Directory path: ").strip()
                if path:
                    rag.ingest_directory(path)
            elif choice == '3':
                query = input("  Search query: ").strip()
                if query:
                    results = rag.retrieve(query, top_k=5)
                    for i, r in enumerate(results, 1):
                        print(f"\n  [{i}] Score: {r['score']} | Source: {r['source']}")
                        print(f"      {r['text'][:150]}...")
            elif choice == '4':
                query = input("  Question: ").strip()
                if query:
                    augmented = rag.augment_prompt(query)
                    print(f"\n  Augmented prompt:\n  {'─' * 50}")
                    print(f"  {augmented[:500]}...")
            elif choice == '5':
                stats = rag.get_stats()
                print(f"\n  RAG Stats:")
                for k, v in stats.items():
                    print(f"    {k}: {v}")
        except Exception as e:
            print(f"  Error: {e}")
