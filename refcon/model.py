"""Backbone for the binned CN-ratio model.

Provides the mask-aware Transformer stack (`MaskedConvStemTransformer`)
used by `model_binned_dev.BinnedDevModel`. Self-contained; only depends
on PyTorch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================================================================== #
# Rotary Position Embedding
# ====================================================================== #

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int = 2048):
        super().__init__()
        assert dim % 2 == 0
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos().unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("sin_cached", freqs.sin().unsqueeze(0).unsqueeze(0), persistent=False)

    def forward(self, seq_len: int):
        if seq_len > self.cos_cached.shape[2]:
            self._build_cache(seq_len)
        return self.cos_cached[:, :, :seq_len], self.sin_cached[:, :, :seq_len]


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.repeat(1, 1, 1, 2)
    sin = sin.repeat(1, 1, 1, 2)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


# ====================================================================== #
# Mask-Aware Attention
# ====================================================================== #

class MaskedGenomicSelfAttention(nn.Module):
    """Multi-head self-attention with RoPE and optional padding mask."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None):
        """
        x:         (B, S, d_model)
        attn_mask: (B, S) bool - True = valid, False = padded. None = no masking.
        """
        B, S, _ = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        cos, sin = self.rope(S)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        sdpa_mask = None
        if attn_mask is not None:
            sdpa_mask = attn_mask.unsqueeze(1).unsqueeze(2).expand(-1, -1, S, -1)

        drop = self.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask, dropout_p=drop)
        attn_out = attn_out.transpose(1, 2).reshape(B, S, self.d_model)
        return self.out_proj(attn_out)


# ====================================================================== #
# Transformer Block
# ====================================================================== #

class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with optional attention mask."""

    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MaskedGenomicSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None):
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.ffn(self.ln2(x))
        return x


# ====================================================================== #
# Conv Stem + Transformer Backbone (mask-aware)
# ====================================================================== #

class MaskedConvStemTransformer(nn.Module):
    """Conv1d stem + Transformer stack with attention mask support."""

    def __init__(self, d_model: int = 128, n_heads: int = 4, n_layers: int = 4,
                 dim_ff: int = 512, dropout: float = 0.1, input_dim: int = 1,
                 conv_kernel_size: int = 11):
        super().__init__()
        self.conv_stem = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=conv_kernel_size,
                      padding=conv_kernel_size // 2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=conv_kernel_size,
                      padding=conv_kernel_size // 2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.post_stem_norm = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dim_ff, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, features: torch.Tensor, attn_mask: torch.Tensor | None = None):
        """
        features:  (B, S, input_dim)
        attn_mask: (B, S) bool - True = valid, False = padded
        """
        x = features.transpose(1, 2)       # (B, input_dim, S)
        x = self.conv_stem(x)              # (B, d_model, S)
        x = x.transpose(1, 2)              # (B, S, d_model)
        x = self.post_stem_norm(x)
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)
        return self.final_norm(x)


if __name__ == "__main__":
    # Smoke test: variable-length padded batch through the backbone.
    B, S, D = 4, 32, 64
    lengths = torch.tensor([32, 20, 10, 28])
    x = torch.randn(B, S, D)
    mask = torch.arange(S).unsqueeze(0) < lengths.unsqueeze(1)
    m = MaskedConvStemTransformer(d_model=128, input_dim=D, conv_kernel_size=7)
    out = m(x, attn_mask=mask)
    print(f"backbone out: {out.shape}  params: {sum(p.numel() for p in m.parameters()):,}")
    print("smoke test passed.")
