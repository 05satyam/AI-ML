"""
Minimal decoder-only Transformer (GPT-style) built from scratch in PyTorch.

Each class maps to one idea from "Attention Is All You Need":
  - Token + positional embeddings
  - Scaled dot-product multi-head self-attention
  - Position-wise feed-forward network
  - Residual connections + layer normalization
  - Causal (look-ahead) mask for autoregressive training
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import VOCAB, PAD_ID


@dataclass
class ModelConfig:
    vocab_size: int = len(VOCAB)
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 192
    max_seq_len: int = 16
    dropout: float = 0.1


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encodings from the original Transformer paper.

    Transformers have no recurrence, so we inject position information by adding
    a unique vector to each token embedding based on its index in the sequence.
    """

    def __init__(self, d_model: int, max_seq_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model)
        )
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # shape: (1, seq, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention in one sentence:
      each token asks every previous token (causal mask) "how relevant are you to me?"
      and builds a weighted summary of their value vectors.

    We split d_model into n_heads independent attention operations, then concat.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # (batch, heads, seq, head_dim)
        def reshape(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = map(reshape, (q, k, v))

        # Attention scores: QK^T / sqrt(d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask: position i may not attend to j > i.
        scores = scores.masked_fill(causal_mask == 0, float("-inf"))

        if key_padding_mask is not None:
            # key_padding_mask: (batch, seq), True where token is PAD
            pad_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(pad_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.out(context)


class TransformerBlock(nn.Module):
    """One encoder-style block used in a decoder-only stack (pre-norm variant)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadSelfAttention(
            config.d_model, config.n_heads, config.dropout
        )
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), causal_mask, key_padding_mask)
        x = x + self.ff(self.ln2(x))
        return x


class AdditionTransformer(nn.Module):
    """Decoder-only transformer that predicts the next character of an addition string."""

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config

        self.token_emb = nn.Embedding(c.vocab_size, c.d_model, padding_idx=PAD_ID)
        self.pos_emb = PositionalEncoding(c.d_model, c.max_seq_len, c.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(c) for _ in range(c.n_layers)])
        self.ln_f = nn.LayerNorm(c.d_model)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)

        # Weight tying: output projection shares weights with input embeddings.
        self.lm_head.weight = self.token_emb.weight

        causal = torch.tril(torch.ones(c.max_seq_len, c.max_seq_len))
        self.register_buffer("causal_mask", causal.unsqueeze(0).unsqueeze(0))

    def forward(
        self,
        input_ids: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Returns logits of shape (batch, seq, vocab_size).
        """
        x = self.token_emb(input_ids)
        x = self.pos_emb(x)

        mask = self.causal_mask[:, :, : x.size(1), : x.size(1)]
        for block in self.blocks:
            x = block(x, mask, key_padding_mask)

        x = self.ln_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 8,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """
        Greedy autoregressive decoding: append one token at a time.

        prompt_ids: (1, prompt_len)
        """
        generated = prompt_ids
        for _ in range(max_new_tokens):
            logits = self.forward(generated)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_id], dim=1)
            if eos_id is not None and next_id.item() == eos_id:
                break
        return generated


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
