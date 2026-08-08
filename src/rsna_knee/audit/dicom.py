from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from .common import hash_identifier


HEADER_KEYWORDS = [
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "PatientID",
    "PatientSex",
    "PatientAge",
    "InstanceNumber",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "Rows",
    "Columns",
    "PixelSpacing",
    "SliceThickness",
    "SpacingBetweenSlices",
    "BitsAllocated",
    "BitsStored",
    "HighBit",
    "PixelRepresentation",
    "PhotometricInterpretation",
    "RescaleSlope",
    "RescaleIntercept",
    "WindowCenter",
    "WindowWidth",
    "SeriesDescription",
    "ProtocolName",
    "SequenceName",
    "ScanningSequence",
    "SequenceVariant",
    "ScanOptions",
    "MRAcquisitionType",
    "RepetitionTime",
    "EchoTime",
    "EchoTrainLength",
    "FlipAngle",
    "MagneticFieldStrength",
    "Manufacturer",
    "ManufacturerModelName",
    "SoftwareVersions",
    "ImageLaterality",
    "Laterality",
]

# DICOM tags with VM 1-n: decode to a bare scalar on single-valued files and a list
# on multi-valued ones, which Arrow can't write as one parquet column. Normalized to
# list-or-None below so every record has a consistent type regardless of file VM.
MULTI_VALUE_KEYWORDS = {
    "ScanningSequence",
    "SequenceVariant",
    "ScanOptions",
    "WindowCenter",
    "WindowWidth",
    "SoftwareVersions",
}


def _safe_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, (list, tuple)) or value.__class__.__name__ == "MultiValue":
        return [_safe_value(item) for item in value]
    try:
        converted = float(value)
        return converted if math.isfinite(converted) else None
    except (TypeError, ValueError):
        return str(value)


def _float_vector(value: Any, expected: int) -> list[float] | None:
    safe = _safe_value(value)
    if not isinstance(safe, list) or len(safe) != expected:
        return None
    try:
        result = [float(item) for item in safe]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def derive_geometry(
    orientation_value: Any, position_value: Any
) -> tuple[str, list[float] | None, float | None]:
    orientation = _float_vector(orientation_value, 6)
    position = _float_vector(position_value, 3)
    if orientation is None:
        return "Unknown", None, None
    row = orientation[:3]
    col = orientation[3:]
    normal = [
        row[1] * col[2] - row[2] * col[1],
        row[2] * col[0] - row[0] * col[2],
        row[0] * col[1] - row[1] * col[0],
    ]
    norm = math.sqrt(sum(value * value for value in normal))
    if norm <= 1e-8:
        return "Unknown", None, None
    normal = [value / norm for value in normal]
    axis = max(range(3), key=lambda index: abs(normal[index]))
    plane = ("Sagittal", "Coronal", "Axial")[axis]
    scalar = None if position is None else sum(n * p for n, p in zip(normal, position))
    return plane, normal, scalar


def read_header_record(
    absolute_path: str,
    relative_path: str,
    patient_salt: str,
    force: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    path = Path(absolute_path)
    parts = Path(relative_path).parts
    base: dict[str, Any] = {
        "relative_path": Path(relative_path).as_posix(),
        "path_study_uid": parts[0] if len(parts) >= 1 else None,
        "path_series_uid": parts[1] if len(parts) >= 2 else None,
    }
    try:
        import pydicom

        dataset = pydicom.dcmread(
            path,
            stop_before_pixels=True,
            force=force,
            specific_tags=HEADER_KEYWORDS,
        )
        values = {keyword: _safe_value(getattr(dataset, keyword, None)) for keyword in HEADER_KEYWORDS}
        for keyword in MULTI_VALUE_KEYWORDS:
            if values.get(keyword) is not None and not isinstance(values[keyword], list):
                values[keyword] = [values[keyword]]
        plane, normal, position_scalar = derive_geometry(
            values.get("ImageOrientationPatient"), values.get("ImagePositionPatient")
        )
        patient_hash = hash_identifier(
            None if values.get("PatientID") is None else str(values["PatientID"]), patient_salt
        )
        values.pop("PatientID", None)
        transfer_syntax = None
        if getattr(dataset, "file_meta", None) is not None:
            transfer_syntax = _safe_value(getattr(dataset.file_meta, "TransferSyntaxUID", None))
        base.update(values)
        base.update(
            {
                "status": "ok",
                "error_type": None,
                "error_message": None,
                "patient_hash": patient_hash,
                "derived_plane": plane,
                "slice_normal": normal,
                "position_scalar": position_scalar,
                "TransferSyntaxUID": transfer_syntax,
                "file_size_bytes": path.stat().st_size,
            }
        )
    except Exception as error:  # audit must record every malformed file
        base.update(
            {
                "status": "error",
                "error_type": type(error).__name__,
                "error_message": str(error)[:1000],
                "file_size_bytes": path.stat().st_size if path.exists() else None,
            }
        )
    base["header_seconds"] = time.perf_counter() - started
    return base

