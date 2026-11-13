"""Temporal fusion head shared by the formal wrench model and preflight stack.

Extracted from ``experiments/tactohand_preflight/model.py`` so the production
``training/tactohand_wrench`` task no longer imports from ``experiments/``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalWrenchHead(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        temporal: str = "none",
        temporal_layers: int = 2,
        temporal_heads: int = 8,
        dropout: float = 0.0,
        max_seq_len: int = 64,
    ) -> None:
        super().__init__()
        self.temporal_kind = temporal
        if temporal == "none":
            self.pos_embed = None
            self.temporal = nn.Identity()
        elif temporal == "transformer":
            self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, dim) * 0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=temporal_heads,
                dim_feedforward=dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal = nn.Sequential(
                nn.TransformerEncoder(layer, num_layers=temporal_layers, enable_nested_tensor=False),
                nn.LayerNorm(dim),
            )
        else:
            raise ValueError(f"unsupported temporal module: {temporal}")
        self.head = nn.Linear(dim, 6)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is not None:
            if features.shape[1] > self.pos_embed.shape[1]:
                raise ValueError("sequence is longer than max_seq_len")
            features = features + self.pos_embed[:, : features.shape[1]]
        return self.head(self.temporal(features))
