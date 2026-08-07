from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as functional
from torch import nn


def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    positive_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    valid = torch.isfinite(targets)
    if not valid.any():
        return logits.sum() * 0.0
    safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
    loss = functional.binary_cross_entropy_with_logits(
        logits, safe_targets, reduction="none", pos_weight=positive_weights
    )
    return loss[valid].mean()


def batch_pairwise_auc_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    losses = []
    for label_index in range(logits.shape[1]):
        valid = torch.isfinite(targets[:, label_index])
        label_targets = targets[valid, label_index]
        label_logits = logits[valid, label_index]
        positive = label_logits[label_targets > 0.5]
        negative = label_logits[label_targets <= 0.5]
        if positive.numel() and negative.numel():
            differences = positive[:, None] - negative[None, :]
            losses.append(functional.softplus(-differences).mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def recall_safe_coverage_loss(
    selector_scores: torch.Tensor,
    evidence_targets: torch.Tensor,
    labels: torch.Tensor,
    candidate_mask: torch.Tensor,
    k: int,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Soft Top-K coverage objective using teacher window evidence q_ij."""
    valid_evidence = torch.isfinite(evidence_targets)
    evidence = torch.where(valid_evidence, evidence_targets.clamp(0, 1), torch.zeros_like(evidence_targets))
    masked_scores = selector_scores.masked_fill(~candidate_mask.unsqueeze(-1), -torch.inf)
    utility = masked_scores.max(dim=-1).values
    sorted_utility = utility.sort(dim=1, descending=True).values
    valid_counts = candidate_mask.sum(dim=1).clamp_min(1)
    rank = torch.minimum(valid_counts, torch.full_like(valid_counts, int(k))) - 1
    threshold = sorted_utility.gather(1, rank[:, None]).detach()
    gates = torch.sigmoid((utility - threshold) / max(float(temperature), 1e-4))
    gates = torch.where(candidate_mask, gates, torch.zeros_like(gates))
    retained = (gates.unsqueeze(-1) * evidence).clamp(0.0, 1.0 - 1e-6)
    coverage = 1.0 - torch.prod(1.0 - retained, dim=1)
    positive = (labels > 0.5) & torch.isfinite(labels) & valid_evidence.any(dim=1)
    if not positive.any():
        return selector_scores.sum() * 0.0
    return -torch.log(coverage.clamp_min(1e-6))[positive].mean()


class CompositeLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 1.0,
        ranking_weight: float = 0.0,
        coverage_weight: float = 0.0,
        selector_k: int = 15,
        coverage_temperature: float = 0.2,
        positive_weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.ranking_weight = float(ranking_weight)
        self.coverage_weight = float(coverage_weight)
        self.selector_k = int(selector_k)
        self.coverage_temperature = float(coverage_temperature)
        weights = None if positive_weights is None else torch.tensor(positive_weights, dtype=torch.float32)
        self.register_buffer("positive_weights", weights)

    def forward(
        self, output: Mapping[str, torch.Tensor], batch: Mapping[str, Any]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = output["logits"]
        labels = batch["labels"]
        components: dict[str, torch.Tensor] = {}
        components["bce"] = masked_bce_with_logits(logits, labels, self.positive_weights)
        components["ranking"] = batch_pairwise_auc_loss(logits, labels)
        coverage = logits.sum() * 0.0
        if (
            self.coverage_weight > 0
            and "selector_scores" in output
            and "window_targets" in batch
        ):
            coverage = recall_safe_coverage_loss(
                output["selector_scores"],
                batch["window_targets"],
                labels,
                batch["mask"],
                k=self.selector_k,
                temperature=self.coverage_temperature,
            )
        components["coverage"] = coverage
        total = (
            self.bce_weight * components["bce"]
            + self.ranking_weight * components["ranking"]
            + self.coverage_weight * components["coverage"]
        )
        components["total"] = total
        return total, components


def build_loss(config: Mapping[str, Any]) -> CompositeLoss:
    return CompositeLoss(**dict(config))
