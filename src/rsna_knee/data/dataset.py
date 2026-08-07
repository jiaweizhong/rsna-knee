from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset

from rsna_knee.constants import LABEL_COLUMNS, PLANE_TO_ID


def _decode_dicom(path: Path) -> np.ndarray:
    import pydicom

    dataset = pydicom.dcmread(path, force=True)
    array = np.asarray(dataset.pixel_array, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected one 2D MRI slice at {path}, got shape={array.shape}")
    slope = float(getattr(dataset, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0) or 0.0)
    return array * slope + intercept


def _normalize(array: np.ndarray, method: str) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array, dtype=np.float32)
    if method == "per_slice_robust":
        lower, upper = np.percentile(finite, [1, 99])
        scale = max(float(upper - lower), 1e-6)
        return np.clip((array - lower) / scale, 0.0, 1.0)
    if method == "zscore":
        mean = float(finite.mean())
        std = max(float(finite.std()), 1e-6)
        return np.clip((array - mean) / std, -5.0, 5.0)
    if method == "none":
        return array
    raise KeyError(f"Unknown normalization method: {method}")


def _resize(array: np.ndarray, size: tuple[int, int]) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(array)).float()[None, None]
    resized = functional.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
    return resized[0, 0]


def _uniform_subsample(items: list[Any], count: int) -> list[Any]:
    if count <= 0 or len(items) <= count:
        return items
    indices = np.linspace(0, len(items) - 1, count).round().astype(int)
    return [items[index] for index in sorted(set(indices.tolist()))]


class StudyDataset(Dataset[dict[str, Any]]):
    """Random-access JSONL dataset; only offsets and one study are held in memory."""

    def __init__(
        self,
        manifest_path: str | Path,
        dicom_root: str | Path,
        image_size: tuple[int, int] = (224, 224),
        normalization: str = "per_slice_robust",
        max_candidates: int = 0,
        on_decode_error: str = "raise",
        input_mean: list[float] | None = None,
        input_std: list[float] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.dicom_root = Path(dicom_root)
        self.image_size = tuple(int(value) for value in image_size)
        self.normalization = normalization
        self.max_candidates = int(max_candidates)
        self.on_decode_error = on_decode_error
        self.input_mean = input_mean
        self.input_std = input_std
        self.offsets: list[int] = []
        with self.manifest_path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)

    def __len__(self) -> int:
        return len(self.offsets)

    def _read_record(self, index: int) -> dict[str, Any]:
        with self.manifest_path.open("rb") as handle:
            handle.seek(self.offsets[index])
            return json.loads(handle.readline())

    def _load_window(self, paths: list[str], cache: dict[str, torch.Tensor]) -> torch.Tensor:
        slices = []
        for relative in paths:
            if relative in cache:
                slices.append(cache[relative])
                continue
            try:
                array = _normalize(_decode_dicom(self.dicom_root / relative), self.normalization)
                decoded = _resize(array, self.image_size)
            except Exception:
                if self.on_decode_error != "zeros":
                    raise
                decoded = torch.zeros(self.image_size, dtype=torch.float32)
            cache[relative] = decoded
            slices.append(decoded)
        window = torch.stack(slices, dim=0)
        if self.input_mean is not None or self.input_std is not None:
            mean = torch.tensor(self.input_mean or [0.0] * len(paths), dtype=window.dtype)[:, None, None]
            std = torch.tensor(self.input_std or [1.0] * len(paths), dtype=window.dtype)[:, None, None]
            if mean.shape[0] != window.shape[0] or std.shape[0] != window.shape[0]:
                raise ValueError("input_mean/input_std length must equal 2.5D window channels")
            window = (window - mean) / std.clamp_min(1e-6)
        return window

    @staticmethod
    def _metadata(window: dict[str, Any]) -> torch.Tensor:
        plane_id = PLANE_TO_ID.get(str(window.get("plane", "Unknown")), 0)
        plane = [float(plane_id == index) for index in (1, 2, 3)]
        fluid = window.get("fluid_sensitive")
        try:
            fluid_value = float(fluid)
            if not math.isfinite(fluid_value):
                fluid_value = -1.0
        except (TypeError, ValueError):
            fluid_value = -1.0
        position = float(window.get("relative_position", 0.5))
        return torch.tensor([position, fluid_value, *plane], dtype=torch.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self._read_record(index)
        windows = _uniform_subsample(record.get("windows", []), self.max_candidates)
        if not windows:
            raise ValueError(f"Study {record.get('study_id')} has no valid windows")
        slice_cache: dict[str, torch.Tensor] = {}
        images = torch.stack([self._load_window(window["paths"], slice_cache) for window in windows])
        metadata = torch.stack([self._metadata(window) for window in windows])
        raw_labels = record.get("labels")
        if raw_labels is None:
            labels = torch.full((len(LABEL_COLUMNS),), float("nan"), dtype=torch.float32)
        else:
            labels = torch.tensor(
                [float("nan") if value is None else float(value) for value in raw_labels],
                dtype=torch.float32,
            )
        item: dict[str, Any] = {
            "study_id": str(record["study_id"]),
            "images": images,
            "metadata": metadata,
            "labels": labels,
        }
        targets = [window.get("evidence_targets") for window in windows]
        if any(target is not None for target in targets):
            item["window_targets"] = torch.tensor(
                [
                    [float("nan")] * len(LABEL_COLUMNS) if target is None else target
                    for target in targets
                ],
                dtype=torch.float32,
            )
        return item


def collate_studies(items: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(items)
    max_windows = max(item["images"].shape[0] for item in items)
    channels, height, width = items[0]["images"].shape[1:]
    metadata_dim = items[0]["metadata"].shape[-1]
    images = torch.zeros(batch_size, max_windows, channels, height, width)
    metadata = torch.zeros(batch_size, max_windows, metadata_dim)
    mask = torch.zeros(batch_size, max_windows, dtype=torch.bool)
    labels = torch.stack([item["labels"] for item in items])
    has_targets = any("window_targets" in item for item in items)
    window_targets = torch.full(
        (batch_size, max_windows, len(LABEL_COLUMNS)), float("nan"), dtype=torch.float32
    )
    for index, item in enumerate(items):
        count = item["images"].shape[0]
        images[index, :count] = item["images"]
        metadata[index, :count] = item["metadata"]
        mask[index, :count] = True
        if "window_targets" in item:
            window_targets[index, :count] = item["window_targets"]
    batch: dict[str, Any] = {
        "study_id": [item["study_id"] for item in items],
        "images": images,
        "metadata": metadata,
        "mask": mask,
        "labels": labels,
    }
    if has_targets:
        batch["window_targets"] = window_targets
    return batch
