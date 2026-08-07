from __future__ import annotations

import math

import torch
from torch import nn

from rsna_knee.registry import Registry

AGGREGATORS: Registry[nn.Module] = Registry("aggregator")


def _masked_mean(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(features.dtype)
    return (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _masked_max(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    values = features.masked_fill(~mask.unsqueeze(-1), -torch.inf).max(dim=1).values
    return torch.where(torch.isfinite(values), values, torch.zeros_like(values))


@AGGREGATORS.register("mean_max")
class MeanMaxAggregator(nn.Module):
    def __init__(self, feature_dim: int, num_labels: int, hidden_dim: int = 256, **_: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor, metadata: torch.Tensor | None = None
    ) -> torch.Tensor:
        pooled = torch.cat([_masked_mean(features, mask), _masked_max(features, mask)], dim=-1)
        return self.head(pooled)


@AGGREGATORS.register("attention")
class AttentionAggregator(nn.Module):
    def __init__(self, feature_dim: int, num_labels: int, hidden_dim: int = 128, **_: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )
        self.head = nn.Linear(feature_dim, num_labels)

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor, metadata: torch.Tensor | None = None
    ) -> torch.Tensor:
        scores = self.attention(features).squeeze(-1).masked_fill(~mask, -torch.inf)
        weights = torch.softmax(scores, dim=1).masked_fill(~mask, 0.0)
        pooled = torch.sum(features * weights.unsqueeze(-1), dim=1)
        return self.head(pooled)


@AGGREGATORS.register("per_label_query")
class PerLabelQueryAggregator(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_labels: int,
        metadata_dim: int = 0,
        use_metadata: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_labels, feature_dim) / math.sqrt(feature_dim))
        self.metadata_projection = (
            nn.Linear(metadata_dim, feature_dim) if use_metadata and metadata_dim > 0 else None
        )
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier_weight = nn.Parameter(torch.randn(num_labels, feature_dim) / math.sqrt(feature_dim))
        self.classifier_bias = nn.Parameter(torch.zeros(num_labels))

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor, metadata: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.metadata_projection is not None and metadata is not None:
            features = features + self.metadata_projection(metadata)
        features = self.norm(features)
        attention = torch.einsum("bkd,ld->bkl", features, self.queries)
        attention = attention.masked_fill(~mask.unsqueeze(-1), -torch.inf)
        weights = torch.softmax(attention, dim=1).masked_fill(~mask.unsqueeze(-1), 0.0)
        pooled = torch.einsum("bkl,bkd->bld", weights, features)
        pooled = self.dropout(pooled)
        return torch.einsum("bld,ld->bl", pooled, self.classifier_weight) + self.classifier_bias

