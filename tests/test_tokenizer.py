"""Tests for the Tokenizer."""

import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.tokenizer import Tokenizer


class TestTokenizer:
    """Test the tokenizer."""

    def setup_method(self):
        self.tok = Tokenizer(vocab_size=1000, method="word")
        self.tok.build_vocab([
            "Hello world how are you",
            "I am doing great thank you",
            "What is artificial intelligence",
            "Machine learning is a subset of AI"
        ])

    def test_special_tokens(self):
        """Special tokens have correct IDs."""
        assert self.tok.pad_token_id == 0
        assert self.tok.unk_token_id == 1
        assert self.tok.bos_token_id == 2
        assert self.tok.eos_token_id == 3
        assert self.tok.sep_token_id == 4

    def test_vocab_size(self):
        """Vocabulary is built correctly."""
        assert len(self.tok) > 5  # More than just special tokens
        assert len(self.tok) <= 1000

    def test_encode_basic(self):
        """Encode produces token IDs."""
        ids = self.tok.encode("hello world")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert len(ids) > 0

    def test_encode_with_special_tokens(self):
        """Encode adds BOS and EOS."""
        ids = self.tok.encode("hello", add_special=True)
        assert ids[0] == self.tok.bos_token_id
        assert ids[-1] == self.tok.eos_token_id

    def test_encode_without_special_tokens(self):
        """Encode without special tokens."""
        ids = self.tok.encode("hello", add_special=False)
        assert ids[0] != self.tok.bos_token_id

    def test_encode_max_length(self):
        """Encode respects max_length."""
        ids = self.tok.encode("hello world how are you doing today", max_length=5)
        assert len(ids) <= 5

    def test_encode_padding(self):
        """Encode pads to max_length."""
        ids = self.tok.encode("hello", max_length=20, pad=True)
        assert len(ids) == 20
        assert ids[-1] == self.tok.pad_token_id

    def test_decode_basic(self):
        """Decode produces text."""
        ids = self.tok.encode("hello world", add_special=False)
        text = self.tok.decode(ids)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_encode_decode_roundtrip(self):
        """Encode then decode recovers text."""
        original = "hello world"
        ids = self.tok.encode(original, add_special=False)
        decoded = self.tok.decode(ids, skip_special=True)
        assert decoded == original

    def test_decode_skip_special(self):
        """Decode skips special tokens."""
        ids = [self.tok.bos_token_id, self.tok.eos_token_id]
        text = self.tok.decode(ids, skip_special=True)
        assert text == ""

    def test_unknown_token(self):
        """Unknown words get UNK token ID."""
        ids = self.tok.encode("xyzzyspoon", add_special=False)
        assert self.tok.unk_token_id in ids

    def test_encode_conversation(self):
        """Encode conversation produces input/target pairs."""
        input_ids, target_ids = self.tok.encode_conversation(
            "hello", "hi there", max_length=32
        )
        assert len(input_ids) == 32
        assert len(target_ids) == 32
        assert input_ids[0] == self.tok.bos_token_id
        assert self.tok.sep_token_id in input_ids

    def test_encode_conversation_truncation(self):
        """Long conversations are truncated."""
        long_text = " ".join(["word"] * 100)
        input_ids, target_ids = self.tok.encode_conversation(
            long_text, long_text, max_length=32
        )
        assert len(input_ids) == 32

    def test_save_and_load(self):
        """Save and load preserves tokenizer."""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            path = f.name
        try:
            self.tok.save(path)
            loaded = Tokenizer.load(path)
            assert len(loaded) == len(self.tok)
            assert loaded.vocab_size == self.tok.vocab_size
            assert loaded.method == self.tok.method
            # Check same encoding
            ids1 = self.tok.encode("hello world")
            ids2 = loaded.encode("hello world")
            assert ids1 == ids2
        finally:
            os.unlink(path)

    def test_char_tokenizer(self):
        """Character-level tokenizer works."""
        tok = Tokenizer(vocab_size=500, method="char")
        tok.build_vocab(["hello world"])
        ids = tok.encode("hello", add_special=False)
        assert len(ids) == 5  # h-e-l-l-o

    def test_len(self):
        """__len__ returns vocab size."""
        assert len(self.tok) == len(self.tok.token2id)
