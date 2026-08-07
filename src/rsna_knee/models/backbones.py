from __future__ import annotations

import importlib
from typing import Any

import torch
from torch import nn

from rsna_knee.registry import Registry

BACKBONES: Registry[nn.Module] = Registry("backbone")


def _pool_output(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        for key in ("pooler_output", "last_hidden_state", "features", "x"):
            if key in output and output[key] is not None:
                output = output[key]
                break
    if hasattr(output, "last_hidden_state"):
        output = output.last_hidden_state
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Backbone returned unsupported type: {type(output).__name__}")
    if output.ndim == 2:
        return output
    if output.ndim == 3:
        return output[:, 0] if output.shape[1] > 1 else output[:, 0]
    if output.ndim in (4, 5):
        return output.flatten(2).mean(dim=-1)
    raise ValueError(f"Backbone returned unsupported tensor shape: {tuple(output.shape)}")


class FeatureBackbone(nn.Module):
    out_dim: int
    spatial_dims: int = 2


@BACKBONES.register("tiny_cnn")
class TinyConvBackbone(FeatureBackbone):
    def __init__(self, in_channels: int = 3, out_dim: int = 128, width: int = 32) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, width, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(width),
            nn.GELU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.GELU(),
            nn.Conv2d(width * 2, out_dim, 3, stride=2, padding=1, bias=False),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images).flatten(1)


@BACKBONES.register("timm")
class TimmBackbone(FeatureBackbone):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        in_channels: int = 3,
        checkpoint_path: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as error:
            raise ImportError("Install the 'train' extra to use timm backbones") from error
        kwargs = dict(model_kwargs or {})
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
            global_pool="avg",
            **kwargs,
        )
        self.out_dim = int(getattr(self.model, "num_features"))
        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            state = state.get("model", state)
            self.model.load_state_dict(state, strict=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return _pool_output(self.model(images))


@BACKBONES.register("huggingface")
class HuggingFaceBackbone(FeatureBackbone):
    def __init__(
        self,
        model_name_or_path: str,
        local_files_only: bool = True,
        trust_remote_code: bool = False,
        out_dim: int | None = None,
        in_channels: int | None = None,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as error:
            raise ImportError("Install the 'train' extra to use Hugging Face backbones") from error
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        hidden = getattr(self.model.config, "hidden_size", None)
        self.out_dim = int(out_dim or hidden)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return _pool_output(self.model(pixel_values=images))


def _import_object(path: str) -> Any:
    module_name, attribute = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), attribute)


@BACKBONES.register("external")
class ExternalBackbone(FeatureBackbone):
    """Adapter for source repositories supplied by the user.

    The factory receives ``factory_kwargs`` and must return ``torch.nn.Module``.
    Its output may be [B,D], [B,T,D], [B,D,H,W], or [B,D,Z,H,W].
    """

    def __init__(
        self,
        factory: str,
        out_dim: int,
        spatial_dims: int = 2,
        checkpoint_path: str | None = None,
        checkpoint_key: str | None = None,
        strict: bool = False,
        factory_kwargs: dict[str, Any] | None = None,
        in_channels: int | None = None,
    ) -> None:
        super().__init__()
        self.model = _import_object(factory)(**(factory_kwargs or {}))
        if not isinstance(self.model, nn.Module):
            raise TypeError(f"External factory {factory!r} did not return nn.Module")
        self.out_dim = int(out_dim)
        self.spatial_dims = int(spatial_dims)
        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if checkpoint_key:
                state = state[checkpoint_key]
            elif isinstance(state, dict):
                state = state.get("model", state.get("state_dict", state))
            self.model.load_state_dict(state, strict=strict)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return _pool_output(self.model(images))
