from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .common import atomic_text_writer


def _load_parquet_dir(directory: Path, columns: list[str]) -> pd.DataFrame:
    parts = sorted(directory.glob("*.parquet"))
    if not parts:
        return pd.DataFrame(columns=columns)
    frames = [pd.read_parquet(part, columns=columns) for part in parts]
    return pd.concat(frames, ignore_index=True)


def _find_duplicate_groups(
    frame: pd.DataFrame, key_column: str, path_column: str = "relative_path"
) -> list[dict[str, Any]]:
    valid = frame.dropna(subset=[key_column])
    duplicated = valid[valid[key_column].duplicated(keep=False)]
    rows = []
    for key, group in duplicated.groupby(key_column):
        rows.append(
            {
                "level": key_column,
                "key": str(key),
                "count": int(len(group)),
                "paths": ";".join(group[path_column].astype(str).tolist()),
            }
        )
    return rows


def _series_signatures(headers: pd.DataFrame, pixel_hashes: pd.Series) -> pd.DataFrame:
    """One row per series: a signature built from the pixel hashes of its first,
    middle, and last slice in geometric order. Two series with an identical
    signature are almost certainly the same acquisition duplicated under a
    different Study/SeriesInstanceUID. This is an exact-match proxy, not a
    perceptual/near-duplicate check (see decision log for that gap)."""
    frame = headers.copy()
    frame["pixel_sha256"] = frame["relative_path"].map(pixel_hashes)
    frame = frame.dropna(subset=["pixel_sha256"])
    if frame.empty:
        return pd.DataFrame(columns=["StudyInstanceUID", "SeriesInstanceUID", "slice_count", "signature"])
    frame = frame.sort_values(
        ["StudyInstanceUID", "SeriesInstanceUID", "position_scalar", "InstanceNumber"],
        na_position="last",
    )
    rows = []
    for (study, series), group in frame.groupby(["StudyInstanceUID", "SeriesInstanceUID"], sort=False):
        hashes = group["pixel_sha256"].tolist()
        picks = sorted({0, len(hashes) // 2, len(hashes) - 1})
        rows.append(
            {
                "StudyInstanceUID": study,
                "SeriesInstanceUID": series,
                "slice_count": len(hashes),
                "signature": "|".join(hashes[index] for index in picks),
            }
        )
    return pd.DataFrame(rows)


def compute_duplicates(audit_root: str | Path) -> dict[str, Any]:
    root = Path(audit_root)
    tables_dir = root / "tables"
    issues_dir = root / "issues"
    tables_dir.mkdir(parents=True, exist_ok=True)
    issues_dir.mkdir(parents=True, exist_ok=True)

    headers = _load_parquet_dir(
        tables_dir / "dicom_inventory_parts",
        columns=[
            "StudyInstanceUID",
            "SeriesInstanceUID",
            "SOPInstanceUID",
            "relative_path",
            "position_scalar",
            "InstanceNumber",
            "status",
        ],
    )
    headers = headers[headers.get("status") == "ok"] if not headers.empty else headers

    pixels = _load_parquet_dir(
        tables_dir / "pixel_inventory_parts",
        columns=["relative_path", "pixel_sha256", "status"],
    )
    pixels = (
        pixels[pixels.get("status") == "ok"].dropna(subset=["pixel_sha256"])
        if not pixels.empty
        else pixels
    )
    pixel_hashes = pixels.set_index("relative_path")["pixel_sha256"] if not pixels.empty else pd.Series(dtype=object)

    duplicate_rows: list[dict[str, Any]] = []
    duplicate_rows += _find_duplicate_groups(headers, "SOPInstanceUID") if not headers.empty else []
    pixel_frame = pixels.rename(columns={"pixel_sha256": "pixel_sha256"})
    duplicate_rows += _find_duplicate_groups(pixel_frame, "pixel_sha256") if not pixel_frame.empty else []

    study_edges: set[tuple[str, str]] = set()
    path_to_study = (
        headers.dropna(subset=["relative_path"]).set_index("relative_path")["StudyInstanceUID"]
        if not headers.empty
        else pd.Series(dtype=object)
    )

    def _add_cross_study_edges(paths: list[str]) -> None:
        studies = sorted({str(path_to_study[path]) for path in paths if path in path_to_study.index})
        for other in studies[1:]:
            study_edges.add((studies[0], other))

    for row in duplicate_rows:
        _add_cross_study_edges(row["paths"].split(";"))

    signatures = _series_signatures(headers, pixel_hashes) if not headers.empty else pd.DataFrame()
    signature_group_count = 0
    if not signatures.empty:
        duplicated_signatures = signatures[signatures["signature"].duplicated(keep=False)]
        for signature, group in duplicated_signatures.groupby("signature"):
            signature_group_count += 1
            duplicate_rows.append(
                {
                    "level": "series_signature",
                    "key": signature,
                    "count": int(len(group)),
                    "paths": ";".join(group["SeriesInstanceUID"].astype(str).tolist()),
                }
            )
            studies = sorted(set(group["StudyInstanceUID"].astype(str)))
            for other in studies[1:]:
                study_edges.add((studies[0], other))

    duplicates_frame = pd.DataFrame(
        duplicate_rows, columns=["level", "key", "count", "paths"]
    )
    duplicates_frame.to_parquet(tables_dir / "suspected_duplicates.parquet", index=False)

    with atomic_text_writer(issues_dir / "suspected_duplicate_studies.csv") as handle:
        handle.write("study_a,study_b\n")
        for study_a, study_b in sorted(study_edges):
            handle.write(f"{study_a},{study_b}\n")

    sop_groups = int((duplicates_frame["level"] == "SOPInstanceUID").sum()) if not duplicates_frame.empty else 0
    pixel_groups = int((duplicates_frame["level"] == "pixel_sha256").sum()) if not duplicates_frame.empty else 0
    return {
        "sop_uid_duplicate_groups": sop_groups,
        "pixel_duplicate_groups": pixel_groups,
        "series_signature_duplicate_groups": signature_group_count,
        "cross_study_duplicate_edges": len(study_edges),
        "note": "exact-match duplicate detection only (UID reuse, identical pixel bytes, identical "
        "first/mid/last-slice series signature); perceptual/near-duplicate (pHash-after-normalization) "
        "detection is not implemented, see decision log.",
    }
