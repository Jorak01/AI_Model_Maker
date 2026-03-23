"""Tests for rag.py — chunking, vector store, RAG pipeline."""

import os
import json
import pytest
from services.rag import (
    chunk_text, load_document, SimpleVectorStore, RAGPipeline,
)


# ---------------------------------------------------------------------------
# Document Processing
# ---------------------------------------------------------------------------

class TestChunkText:
    def test_basic_chunking(self):
        text = " ".join(f"word{i}" for i in range(100))
        chunks = chunk_text(text, chunk_size=30, overlap=5)
        assert len(chunks) > 1
        assert all(len(c.split()) <= 30 for c in chunks)

    def test_small_text_single_chunk(self):
        # chunk_text filters chunks with <= 20 words, so we need enough words
        text = " ".join(f"word{i}" for i in range(30))
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) == 1

    def test_empty_text(self):
        chunks = chunk_text("")
        assert chunks == []

    def test_overlap_creates_redundancy(self):
        text = " ".join(f"word{i}" for i in range(200))
        chunks = chunk_text(text, chunk_size=50, overlap=10)
        # With overlap, some words should appear in multiple chunks
        all_words = []
        for c in chunks:
            all_words.extend(c.split())
        # Total words > original due to overlap
        assert len(all_words) >= 200


class TestLoadDocument:
    def test_load_text_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("Hello world!")
        assert load_document(str(f)) == "Hello world!"

    def test_load_md_file(self, tmp_path):
        f = tmp_path / "test.md"
        f.write_text("# Title\nContent here")
        result = load_document(str(f))
        assert "Title" in result

    def test_load_missing_file(self):
        assert load_document("nonexistent.txt") == ""

    def test_load_json_file(self, tmp_path):
        f = tmp_path / "test.json"
        f.write_text('{"key": "value"}')
        result = load_document(str(f))
        assert "key" in result


# ---------------------------------------------------------------------------
# SimpleVectorStore
# ---------------------------------------------------------------------------

class TestSimpleVectorStore:
    def test_add_and_search(self):
        store = SimpleVectorStore()
        store.add_document("Python is a programming language", source="doc1")
        store.add_document("The weather is sunny today", source="doc2")
        store.add_document("Java is also a programming language", source="doc3")
        store.build_index()

        results = store.search("programming language", top_k=2)
        assert len(results) == 2
        # Programming-related docs should score higher
        sources = [doc["source"] for _, doc in results]
        assert "doc1" in sources or "doc3" in sources

    def test_empty_store_search(self):
        store = SimpleVectorStore()
        results = store.search("hello")
        assert results == []

    def test_save_and_load(self, tmp_path):
        path = str(tmp_path / "index.json")
        store = SimpleVectorStore()
        store.add_document("Test document content", source="test.txt")
        store.build_index()
        store.save(path)

        store2 = SimpleVectorStore()
        store2.load(path)
        assert len(store2.documents) == 1
        assert store2.documents[0]["source"] == "test.txt"

    def test_search_relevance(self):
        store = SimpleVectorStore()
        store.add_document("Cats are fluffy pets that purr", source="cats")
        store.add_document("Dogs are loyal companions that bark", source="dogs")
        store.add_document("Fish live in water and swim", source="fish")
        store.build_index()

        results = store.search("fluffy purring pet", top_k=1)
        assert len(results) == 1
        assert results[0][1]["source"] == "cats"

    def test_build_index_updates(self):
        store = SimpleVectorStore()
        store.add_document("First doc")
        store.build_index()
        assert len(store.tfidf_vectors) == 1

        store.add_document("Second doc")
        store.build_index()
        assert len(store.tfidf_vectors) == 2

    def test_document_has_id(self):
        store = SimpleVectorStore()
        store.add_document("Test", source="src")
        assert "id" in store.documents[0]
        assert len(store.documents[0]["id"]) > 0


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

class TestRAGPipeline:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, monkeypatch):
        """Use temp directory for RAG index."""
        idx_path = str(tmp_path / "index.json")
        monkeypatch.setattr("services.rag.INDEX_PATH", idx_path)
        self.tmp_path = tmp_path

    def test_ingest_file(self):
        # Create a text file to ingest
        f = self.tmp_path / "doc.txt"
        f.write_text(" ".join(f"word{i}" for i in range(100)))

        rag = RAGPipeline()
        count = rag.ingest_file(str(f))
        assert count > 0

    def test_retrieve(self):
        f = self.tmp_path / "doc.txt"
        # Need enough words (>20) to pass chunk_text filter
        f.write_text(" ".join([
            "Python is a popular programming language used for web development and data science.",
            "It was created by Guido van Rossum and first released in 1991.",
            "Python emphasizes code readability and supports multiple programming paradigms.",
            "It is widely used in machine learning, artificial intelligence, and scientific computing.",
        ]))

        rag = RAGPipeline()
        rag.ingest_file(str(f))
        results = rag.retrieve("programming language")
        assert len(results) > 0

    def test_augment_prompt(self):
        f = self.tmp_path / "doc.txt"
        # Need enough words (>20) to pass chunk_text filter
        f.write_text(" ".join([
            "A pangram is a sentence that uses every letter of the alphabet at least once.",
            "The most famous English pangram is the quick brown fox jumps over the lazy dog.",
            "Pangrams are commonly used in typing tests and font displays to show all characters.",
            "They are also used in handwriting practice and calligraphy exercises around the world.",
        ]))

        rag = RAGPipeline()
        rag.ingest_file(str(f))
        augmented = rag.augment_prompt("What is a pangram?")
        assert "Context:" in augmented
        assert "Question:" in augmented

    def test_augment_prompt_no_context(self):
        rag = RAGPipeline()
        result = rag.augment_prompt("random query")
        assert result == "random query"

    def test_get_stats(self):
        rag = RAGPipeline()
        stats = rag.get_stats()
        assert "total_documents" in stats
        assert "unique_sources" in stats
        assert "index_built" in stats

    def test_ingest_missing_file(self, capsys):
        rag = RAGPipeline()
        count = rag.ingest_file("nonexistent.txt")
        assert count == 0

    def test_chat_without_model(self):
        f = self.tmp_path / "doc.txt"
        f.write_text("Test document about AI and machine learning.")

        rag = RAGPipeline()
        rag.ingest_file(str(f))
        response, sources = rag.chat_with_context("AI")
        assert isinstance(response, str)
        assert isinstance(sources, list)
