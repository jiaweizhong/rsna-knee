from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .common import stable_fraction


def _finite_float(value: Any, default: float) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def read_pixel_record(
    absolute_path: str,
    relative_path: str,
    deep: bool,
    hash_pixels: bool,
    force: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    record: dict[str, Any] = {
        "relative_path": Path(relative_path).as_posix(),
        "deep_statistics": bool(deep),
    }
    try:
        import pydicom

        dataset = pydicom.dcmread(absolute_path, force=force)
        decode_started = time.perf_counter()
        raw = np.asarray(dataset.pixel_array)
        decode_seconds = time.perf_counter() - decode_started
        slope = _finite_float(getattr(dataset, "RescaleSlope", 1.0), 1.0)
        intercept = _finite_float(getattr(dataset, "RescaleIntercept", 0.0), 0.0)
        pixels = raw.astype(np.float32, copy=False) * slope + intercept
        flat = pixels.reshape(-1)
        finite = flat[np.isfinite(flat)]
        if finite.size == 0:
            raise ValueError("Decoded pixel array contains no finite values")
        record.update(
            {
                "status": "ok",
                "error_type": None,
                "error_message": None,
                "shape": list(raw.shape),
                "dtype": str(raw.dtype),
                "pixel_count": int(raw.size),
                "decode_seconds": decode_seconds,
                "pixel_min": float(finite.min()),
                "pixel_max": float(finite.max()),
                "pixel_mean": float(finite.mean()),
                "pixel_std": float(finite.std()),
                "zero_fraction": float(np.mean(finite == 0)),
                "finite_fraction": float(finite.size / max(1, flat.size)),
            }
        )
        if hash_pixels:
            record["pixel_sha256"] = hashlib.sha256(raw.tobytes(order="C")).hexdigest()
        if deep:
            max_values = 65536
            step = max(1, finite.size // max_values)
            sample = finite[::step][:max_values]
            quantiles = np.percentile(sample, [0.5, 1, 5, 25, 50, 75, 95, 99, 99.5])
            for name, value in zip(
                ["p005", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "p995"],
                quantiles,
            ):
                record[name] = float(value)
            median = float(np.median(sample))
            record["mad"] = float(np.median(np.abs(sample - median)))
            histogram, _ = np.histogram(sample, bins=128)
            probabilities = histogram[histogram > 0] / histogram.sum()
            record["histogram_entropy"] = float(-np.sum(probabilities * np.log2(probabilities)))
    except Exception as error:  # codec and corrupt data failures are audit output
        record.update(
            {
                "status": "error",
                "error_type": type(error).__name__,
                "error_message": str(error)[:1000],
            }
        )
    record["total_seconds"] = time.perf_counter() - started
    return record


def read_pixel_record_sampled(
    absolute_path: str,
    relative_path: str,
    deep_sample_rate: float,
    hash_pixels: bool,
    force: bool = True,
) -> dict[str, Any]:
    return read_pixel_record(
        absolute_path,
        relative_path,
        deep=stable_fraction(relative_path) < deep_sample_rate,
        hash_pixels=hash_pixels,
        force=force,
    )
