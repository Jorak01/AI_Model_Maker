"""Tests for the ConversationalModel."""

import os
import sys
import tempfile
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model import ConversationalModel


class TestConversationalModel:
    """Test the custom transformer model."""

    def setup_method(self):
        self.model = ConversationalModel(
            vocab_size=100, embedding_dim=32, hidden_dim=64,
            num_layers=2, num_heads=4, max_seq_length=32,
            dropout=0.1, pad_token_id=0
        )

    def test_model_creation(self):
        """Model initializes with correct attributes."""
        assert self.model.vocab_size == 100
        assert self.model.embedding_dim == 32
        assert self.model.hidden_dim == 64
        assert self.model.num_layers == 2
        assert self.model.num_heads == 4
        assert self.model.max_seq_length == 32
        assert self.model.pad_token_id == 0

    def test_forward_shape(self):
        """Forward pass returns correct output shape."""
        input_ids = torch.randint(0, 100, (2, 10))  # batch=2, seq=10
        logits = self.model(input_ids)
        assert logits.shape == (2, 10, 100)  # (batch, seq, vocab)

    def test_forward_single_token(self):
        """Forward works with single token input."""
        input_ids = torch.randint(0, 100, (1, 1))
        logits = self.model(input_ids)
        assert logits.shape == (1, 1, 100)

    def test_forward_max_length(self):
        """Forward works at max sequence length."""
        input_ids = torch.randint(0, 100, (1, 32))
        logits = self.model(input_ids)
        assert logits.shape == (1, 32, 100)

    def test_generate_output(self):
        """Generate produces tokens."""
        input_ids = torch.randint(0, 100, (1, 5))
        output = self.model.generate(input_ids, max_length=10, eos_token_id=3)
        assert output.shape[0] == 1
        assert output.shape[1] >= 5  # At least input length

    def test_generate_with_eos(self):
        """Generate respects eos_token_id."""
        input_ids = torch.randint(0, 100, (1, 3))
        output = self.model.generate(input_ids, max_length=50, eos_token_id=3)
        assert output.shape[1] <= 53  # input + max_length

    def test_save_and_load(self):
        """Save and load preserves model."""
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            path = f.name
        try:
            self.model.save(path)
            loaded = ConversationalModel.load(path)
            assert loaded.vocab_size == self.model.vocab_size
            assert loaded.embedding_dim == self.model.embedding_dim
            assert loaded.hidden_dim == self.model.hidden_dim
            assert loaded.num_layers == self.model.num_layers
            assert loaded.num_heads == self.model.num_heads

            # Check weights match
            input_ids = torch.randint(0, 100, (1, 5))
            self.model.eval()
            loaded.eval()
            with torch.no_grad():
                orig_out = self.model(input_ids)
                load_out = loaded(input_ids)
            assert torch.allclose(orig_out, load_out, atol=1e-6)
        finally:
            os.unlink(path)

    def test_parameter_count(self):
        """Model has trainable parameters."""
        params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        assert params > 0

    def test_causal_mask(self):
        """Causal mask has correct shape and values."""
        mask = self.model._causal_mask(5, torch.device('cpu'))
        assert mask.shape == (5, 5)
        assert mask[0, 1] == float('-inf')  # Future token blocked
        assert mask[1, 0] == 0.0            # Past token visible


class TestPositionalEncoding:
    """Test positional encoding."""

    def test_output_shape(self):
        from models.model import PositionalEncoding
        pe = PositionalEncoding(dim=32, max_len=64)
        x = torch.zeros(2, 10, 32)
        out = pe(x)
        assert out.shape == (2, 10, 32)

    def test_adds_position_info(self):
        from models.model import PositionalEncoding
        pe = PositionalEncoding(dim=32, max_len=64, dropout=0.0)
        x = torch.zeros(1, 10, 32)
        out = pe(x)
        # Output should not be all zeros (positional info added)
        assert not torch.allclose(out, torch.zeros_like(out))


class TestTransformerBlock:
    """Test single transformer block."""

    def test_output_shape(self):
        from models.model import TransformerBlock
        block = TransformerBlock(dim=32, heads=4, hidden=64)
        x = torch.randn(2, 10, 32)
        out = block(x)
        assert out.shape == (2, 10, 32)

    def test_with_mask(self):
        from models.model import TransformerBlock
        block = TransformerBlock(dim=32, heads=4, hidden=64)
        x = torch.randn(2, 10, 32)
        mask = torch.zeros(10, 10)
        out = block(x, mask=mask)
        assert out.shape == (2, 10, 32)
