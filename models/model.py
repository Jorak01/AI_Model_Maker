"""Transformer-based conversational AI model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    pe: torch.Tensor

    def __init__(self, dim: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:x.size(1)])


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + out)
        return self.norm2(x + self.ff(x))


class ConversationalModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 256, hidden_dim: int = 512,
                 num_layers: int = 4, num_heads: int = 8, max_seq_length: int = 128,
                 dropout: float = 0.1, pad_token_id: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_length = max_seq_length
        self.dropout = dropout
        self.pad_token_id = pad_token_id

        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        self.pos_enc = PositionalEncoding(embedding_dim, max_seq_length, dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.out = nn.Linear(embedding_dim, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(length, length, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        x = self.pos_enc(self.embed(input_ids))
        mask = self._causal_mask(seq_len, input_ids.device)
        for block in self.blocks:
            x = block(x, mask)
        return self.out(x)

    def generate(self, input_ids: torch.Tensor, max_length: int = 100,
                 temperature: float = 0.8, top_k: int = 50, top_p: float = 0.9,
                 repetition_penalty: float = 1.2, eos_token_id: Optional[int] = None) -> torch.Tensor:
        self.eval()
        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_length):
                logits = self.forward(generated)[:, -1, :] / temperature

                # Repetition penalty
                if repetition_penalty != 1.0:
                    for tid in set(generated[0].tolist()):
                        logits[:, tid] /= repetition_penalty

                # Top-k filtering
                if top_k > 0:
                    k = min(top_k, logits.size(-1))
                    cutoff = torch.topk(logits, k)[0][..., -1, None]
                    logits[logits < cutoff] = float('-inf')

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove = cum_probs > top_p
                    remove[..., 1:] = remove[..., :-1].clone()
                    remove[..., 0] = False
                    logits[remove.scatter(1, sorted_idx, remove)] = float('-inf')

                token = torch.multinomial(F.softmax(logits, dim=-1), 1)
                generated = torch.cat([generated, token], dim=1)

                if eos_token_id is not None and token.item() == eos_token_id:
                    break
                if generated.size(1) > self.max_seq_length:
                    generated = generated[:, -self.max_seq_length:]

        return generated

    def save(self, path: str):
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'vocab_size': self.vocab_size, 'embedding_dim': self.embedding_dim,
                'hidden_dim': self.hidden_dim, 'num_layers': self.num_layers,
                'num_heads': self.num_heads, 'max_seq_length': self.max_seq_length,
                'dropout': self.dropout, 'pad_token_id': self.pad_token_id,
            }
        }, path)

    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'ConversationalModel':
        ckpt = torch.load(path, map_location=device)
        model = cls(**ckpt['config'])
        model.load_state_dict(ckpt['state_dict'])
        return model.to(device)
