"""BinnedDevModel - per-bin CN ratio predictor with masked InstanceNorm.

Architecture:
  raw log1p expression (B, max_chunk), zero-padded after `lengths`
  -> masked InstanceNorm (per-sample, on the real-gene prefix only)
  -> Conv1d(1, d_conv, kernel=bin_size, stride=bin_size) + GELU
      -> n_bins = max_chunk // bin_size tokens of d_conv dims each
  -> MaskedConvStemTransformer backbone (d_model=128, 4 layers, RoPE)
  -> ratio_head -> (B, n_bins) per-bin CN/chunk_mean ratios

`predict_per_gene` broadcasts bin predictions back to gene resolution
for downstream gene-level evaluation.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .model import MaskedConvStemTransformer


class BinnedDevModel(nn.Module):

    def __init__(
        self,
        bin_size: int = 16,
        max_chunk: int = 512,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_ff: int = 512,
        dropout: float = 0.1,
        conv_kernel_size: int = 7,
        d_conv: int = 64,
    ):
        super().__init__()
        self.bin_size = bin_size
        self.max_chunk = max_chunk
        self.n_bins = max_chunk // bin_size

        # Learnable affine for masked InstanceNorm
        self.in_scale = nn.Parameter(torch.ones(1))
        self.in_bias = nn.Parameter(torch.zeros(1))

        # Conv binning at input
        self.binner = nn.Sequential(
            nn.Conv1d(1, d_conv, kernel_size=bin_size, stride=bin_size),
            nn.GELU(),
        )

        self.backbone = MaskedConvStemTransformer(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_ff=dim_ff,
            dropout=dropout,
            input_dim=d_conv,
            conv_kernel_size=min(conv_kernel_size, max(3, self.n_bins // 2 * 2 - 1)),
        )

        self.ratio_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, expression: torch.Tensor, lengths: torch.Tensor):
        """
        Args:
            expression: (B, max_chunk) raw log1p, zero-padded
            lengths:    (B,) real gene count

        Returns:
            bin_ratios: (B, n_bins) predicted ratio per bin
            bin_lengths: (B,) number of valid bins
        """
        B, S = expression.shape
        n_bins = S // self.bin_size

        # Masked InstanceNorm: per-sample normalize on real-gene prefix only
        mask = torch.arange(S, device=expression.device).unsqueeze(0) < lengths.unsqueeze(1)
        x = expression.clone()
        for i in range(B):
            L = int(lengths[i].item())
            if L > 1:
                real = x[i, :L]
                x[i, :L] = (real - real.mean()) / (real.std() + 1e-5)
        x = x * self.in_scale + self.in_bias
        x = x * mask.float()

        # Conv binning: (B, 1, S) -> (B, d_conv, n_bins) -> (B, n_bins, d_conv)
        x = x.unsqueeze(1)
        x = self.binner(x)
        x = x.transpose(1, 2)

        # Bin-level mask
        bin_lengths = (lengths.float() / self.bin_size).ceil().long().clamp(min=1, max=n_bins)
        bin_mask = torch.arange(n_bins, device=x.device).unsqueeze(0) < bin_lengths.unsqueeze(1)

        out = self.backbone(x, attn_mask=bin_mask)
        bin_ratios = self.ratio_head(out).squeeze(-1)

        return bin_ratios, bin_lengths

    def predict_per_gene(self, expression: torch.Tensor, lengths: torch.Tensor):
        """Predict per-gene ratios by broadcasting bin predictions.

        Returns (B, max_chunk) where each gene gets its bin's prediction.
        Useful for gene-resolution evaluation and downstream broadcasting.
        """
        bin_ratios, bin_lengths = self.forward(expression, lengths)
        # Repeat each bin prediction for bin_size genes
        per_gene = bin_ratios.unsqueeze(2).expand(-1, -1, self.bin_size)
        per_gene = per_gene.reshape(bin_ratios.shape[0], -1)
        # Pad if max_chunk > n_bins * bin_size
        B, G = per_gene.shape
        if G < self.max_chunk:
            pad = torch.zeros(B, self.max_chunk - G, device=per_gene.device)
            per_gene = torch.cat([per_gene, pad], dim=1)
        return per_gene[:, :self.max_chunk]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    for bs, mc in [(16, 512), (16, 1024), (32, 1024), (8, 512)]:
        model = BinnedDevModel(bin_size=bs, max_chunk=mc)
        B = 4
        lengths = torch.tensor([mc, mc - 100, mc // 2, mc])
        expr = torch.randn(B, mc)
        for i in range(B):
            expr[i, lengths[i]:] = 0.0
        br, bl = model(expr, lengths)
        pg = model.predict_per_gene(expr, lengths)
        print(f"bin={bs} mc={mc}: n_bins={model.n_bins}, params={model.count_parameters():,}, "
              f"bin_ratios={br.shape}, per_gene={pg.shape}")
    print("Smoke test passed.")
