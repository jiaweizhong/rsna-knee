from pathlib import Path

import pandas as pd

from rsna_knee.audit.duplicates import compute_duplicates


def _write_parquet(directory: Path, rows: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(directory / "part-00000.parquet", index=False)


def test_compute_duplicates_finds_sop_pixel_and_series_level_matches(tmp_path: Path) -> None:
    audit_root = tmp_path / "audit"
    tables_dir = audit_root / "tables"

    headers = [
        {
            "StudyInstanceUID": "S1", "SeriesInstanceUID": "SER1", "SOPInstanceUID": "sop1",
            "relative_path": "S1/SER1/a.dcm", "position_scalar": 0.0, "InstanceNumber": 1, "status": "ok",
        },
        {
            "StudyInstanceUID": "S1", "SeriesInstanceUID": "SER1", "SOPInstanceUID": "sop2",
            "relative_path": "S1/SER1/b.dcm", "position_scalar": 10.0, "InstanceNumber": 2, "status": "ok",
        },
        {
            "StudyInstanceUID": "S1", "SeriesInstanceUID": "SER1", "SOPInstanceUID": "sop3",
            "relative_path": "S1/SER1/c.dcm", "position_scalar": 20.0, "InstanceNumber": 3, "status": "ok",
        },
        # SER2 is a byte-for-byte duplicate of SER1 under a different study/series UID.
        {
            "StudyInstanceUID": "S2", "SeriesInstanceUID": "SER2", "SOPInstanceUID": "sop4",
            "relative_path": "S2/SER2/a.dcm", "position_scalar": 0.0, "InstanceNumber": 1, "status": "ok",
        },
        {
            "StudyInstanceUID": "S2", "SeriesInstanceUID": "SER2", "SOPInstanceUID": "sop5",
            "relative_path": "S2/SER2/b.dcm", "position_scalar": 10.0, "InstanceNumber": 2, "status": "ok",
        },
        {
            "StudyInstanceUID": "S2", "SeriesInstanceUID": "SER2", "SOPInstanceUID": "sop6",
            "relative_path": "S2/SER2/c.dcm", "position_scalar": 20.0, "InstanceNumber": 3, "status": "ok",
        },
        # S3 reuses sop1 by accident (corrupted export), but has unrelated pixel content.
        {
            "StudyInstanceUID": "S3", "SeriesInstanceUID": "SER3", "SOPInstanceUID": "sop1",
            "relative_path": "S3/SER3/x.dcm", "position_scalar": 0.0, "InstanceNumber": 1, "status": "ok",
        },
    ]
    pixels = [
        {"relative_path": "S1/SER1/a.dcm", "pixel_sha256": "h1", "status": "ok"},
        {"relative_path": "S1/SER1/b.dcm", "pixel_sha256": "h2", "status": "ok"},
        {"relative_path": "S1/SER1/c.dcm", "pixel_sha256": "h3", "status": "ok"},
        {"relative_path": "S2/SER2/a.dcm", "pixel_sha256": "h1", "status": "ok"},
        {"relative_path": "S2/SER2/b.dcm", "pixel_sha256": "h2", "status": "ok"},
        {"relative_path": "S2/SER2/c.dcm", "pixel_sha256": "h3", "status": "ok"},
        {"relative_path": "S3/SER3/x.dcm", "pixel_sha256": "h7", "status": "ok"},
    ]
    _write_parquet(tables_dir / "dicom_inventory_parts", headers)
    _write_parquet(tables_dir / "pixel_inventory_parts", pixels)

    summary = compute_duplicates(audit_root)

    assert summary["sop_uid_duplicate_groups"] == 1  # sop1 reused
    assert summary["pixel_duplicate_groups"] == 3  # h1, h2, h3 each duplicated
    assert summary["series_signature_duplicate_groups"] == 1  # SER1 == SER2
    assert summary["cross_study_duplicate_edges"] == 2  # (S1,S2) from pixels/series, (S1,S3) from SOP reuse

    edges = (audit_root / "issues" / "suspected_duplicate_studies.csv").read_text(encoding="utf-8")
    assert "S1,S2" in edges
    assert "S1,S3" in edges

    duplicates_frame = pd.read_parquet(tables_dir / "suspected_duplicates.parquet")
    assert set(duplicates_frame["level"]) == {"SOPInstanceUID", "pixel_sha256", "series_signature"}
