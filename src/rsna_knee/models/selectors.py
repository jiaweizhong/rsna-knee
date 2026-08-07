from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from rsna_knee.registry import Registry

SELECTORS: Registry[nn.Module] = Registry("selector")


@dataclass
class SelectionOutput:
    indices: torch.Tensor
    mask: torch.Tensor
    scores: torch.Tensor | None = None


def _fixed_indices(mask: torch.Tensor, k: int, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    device = mask.device
    batch_size, candidates = mask.shape
    effective_k = min(k, candidates)
    output = torch.zeros(batch_size, effective_k, dtype=torch.long, device=device)
    selected_mask = torch.zeros(batch_size, effective_k, dtype=torch.bool, device=device)
    for batch_index in range(batch_size):
        valid = torch.nonzero(mask[batch_index], as_tuple=False).flatten().cpu().numpy()
        if valid.size == 0:
            continue
        take = min(effective_k, valid.size)
        if mode == "uniform":
            positions = np.linspace(0, valid.size - 1, take).round().astype(int)
            chosen = valid[positions]
        elif mode == "central":
            start = max(0, (valid.size - take) // 2)
            chosen = valid[start : start + take]
        else:
            raise KeyError(mode)
        output[batch_index, :take] = torch.as_tensor(chosen, device=device)
        if take < effective_k:
            output[batch_index, take:] = int(chosen[-1])
        selected_mask[batch_index, :take] = True
    return output, selected_mask


class FixedSelector(nn.Module):
    def __init__(self, k: int, mode: str) -> None:
        super().__init__()
        self.k = int(k)
        self.mode = mode

    def forward(
        self, images: torch.Tensor, metadata: torch.Tensor, mask: torch.Tensor
    ) -> SelectionOutput:
        indices, selected_mask = _fixed_indices(mask, self.k, self.mode)
        return SelectionOutput(indices=indices, mask=selected_mask)


@SELECTORS.register("uniform")
class UniformSelector(FixedSelector):
    def __init__(self, k: int = 15, **_: int) -> None:
        super().__init__(k=k, mode="uniform")


@SELECTORS.register("central")
class CentralSelector(FixedSelector):
    def __init__(self, k: int = 15, **_: int) -> None:
        super().__init__(k=k, mode="central")


class CheapEvidenceScorer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        metadata_dim: int,
        num_labels: int,
        width: int = 24,
    ) -> None:
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(in_channels, width, 5, stride=4, padding=2),
            nn.GELU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1, groups=width),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, width), nn.GELU(), nn.Linear(width, width)
        )
        self.head = nn.Linear(width, num_labels)

    def forward(self, images: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        batch_size, candidates, channels, height, width = images.shape
        encoded = self.image_encoder(images.reshape(-1, channels, height, width)).flatten(1)
        encoded = encoded + self.metadata_encoder(metadata.reshape(-1, metadata.shape[-1]))
        return self.head(encoded).reshape(batch_size, candidates, -1)


@SELECTORS.register("learned_topk")
class LearnedTopKSelector(nn.Module):
    def __init__(
        self,
        in_channels: int,
        metadata_dim: int,
        num_labels: int,
        k: int = 15,
        width: int = 24,
    ) -> None:
        super().__init__()
        self.k = int(k)
        self.scorer = CheapEvidenceScorer(in_channels, metadata_dim, num_labels, width=width)

    def forward(
        self, images: torch.Tensor, metadata: torch.Tensor, mask: torch.Tensor
    ) -> SelectionOutput:
        scores = self.scorer(images, metadata)
        utility = scores.max(dim=-1).values.masked_fill(~mask, -torch.inf)
        effective_k = min(self.k, images.shape[1])
        indices = utility.topk(effective_k, dim=1).indices
        selected_mask = mask.gather(1, indices)
        return SelectionOutput(indices=indices, mask=selected_mask, scores=scores)


@SELECTORS.register("recall_safe_topk")
class RecallSafeTopKSelector(LearnedTopKSelector):
    """Per-label proposals followed by utility fill, adapted from BCRS."""

    def forward(
        self, images: torch.Tensor, metadata: torch.Tensor, mask: torch.Tensor
    ) -> SelectionOutput:
        scores = self.scorer(images, metadata)
        batch_size, candidates, _ = scores.shape
        effective_k = min(self.k, candidates)
        masked_scores = scores.masked_fill(~mask.unsqueeze(-1), -torch.inf)
        utility = masked_scores.max(dim=-1).values
        proposals = masked_scores.argmax(dim=1)
        proposal_flags = torch.zeros(batch_size, candidates, device=images.device)
        proposal_flags.scatter_(1, proposals, 1.0)
        finite_utility = torch.where(torch.isfinite(utility), utility, torch.zeros_like(utility))
        scale = finite_utility.abs().amax(dim=1, keepdim=True).detach() + 1.0
        priority = utility + proposal_flags * scale * 4.0
        indices = priority.topk(effective_k, dim=1).indices
        selected_mask = mask.gather(1, indices)
        return SelectionOutput(indices=indices, mask=selected_mask, scores=scores)
