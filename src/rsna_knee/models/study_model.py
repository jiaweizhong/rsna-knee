from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from rsna_knee.constants import LABEL_COLUMNS

from .aggregators import AGGREGATORS
from .backbones import BACKBONES, FeatureBackbone
from .selectors import SELECTORS


def _batched_gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    view_shape = [indices.shape[0], indices.shape[1]] + [1] * (values.ndim - 2)
    expand_shape = list(indices.shape) + list(values.shape[2:])
    gather_indices = indices.view(*view_shape).expand(*expand_shape)
    return values.gather(1, gather_indices)


class StudyModel(nn.Module):
    def __init__(
        self,
        backbone: FeatureBackbone,
        selector: nn.Module,
        aggregator: nn.Module,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.selector = selector
        self.aggregator = aggregator

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        images = batch["images"]
        metadata = batch["metadata"]
        mask = batch["mask"]
        selection = self.selector(images, metadata, mask)
        selected_images = _batched_gather(images, selection.indices)
        selected_metadata = _batched_gather(metadata, selection.indices)
        batch_size, windows, channels, height, width = selected_images.shape
        if getattr(self.backbone, "spatial_dims", 2) == 3:
            backbone_input = selected_images.reshape(batch_size * windows, 1, channels, height, width)
        else:
            backbone_input = selected_images.reshape(batch_size * windows, channels, height, width)
        features = self.backbone(backbone_input).reshape(batch_size, windows, -1)
        logits = self.aggregator(features, selection.mask, selected_metadata)
        output = {
            "logits": logits,
            "selector_indices": selection.indices,
            "selector_mask": selection.mask,
        }
        if selection.scores is not None:
            output["selector_scores"] = selection.scores
        return output


def build_model(config: Mapping[str, Any]) -> StudyModel:
    num_labels = int(config.get("num_labels", len(LABEL_COLUMNS)))
    in_channels = int(config.get("in_channels", 3))
    metadata_dim = int(config.get("metadata_dim", 5))
    backbone = BACKBONES.build(config["backbone"], in_channels=in_channels)
    if not isinstance(backbone, FeatureBackbone):
        raise TypeError("Backbone registry entry must derive from FeatureBackbone")
    selector = SELECTORS.build(
        config["selector"],
        in_channels=in_channels,
        metadata_dim=metadata_dim,
        num_labels=num_labels,
    )
    aggregator = AGGREGATORS.build(
        config["aggregator"],
        feature_dim=backbone.out_dim,
        metadata_dim=metadata_dim,
        num_labels=num_labels,
    )
    return StudyModel(backbone=backbone, selector=selector, aggregator=aggregator)

