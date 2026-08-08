import csv
from pathlib import Path

from rsna_knee.audit.coverage import compute_coverage
from rsna_knee.audit.index import build_file_index
from rsna_knee.constants import LABEL_COLUMNS


def _write_train_csv(path: Path, gold_uids: set[str], all_uids: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["StudyInstanceUID", "Report", *LABEL_COLUMNS])
        for uid in all_uids:
            if uid in gold_uids:
                writer.writerow([uid, "report text", *(["0"] * len(LABEL_COLUMNS))])
            else:
                writer.writerow([uid, "report text", *([""] * len(LABEL_COLUMNS))])


def _write_train_series_csv(path: Path, rows: list[tuple[str, str, str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive"])
        writer.writerows(rows)


def test_coverage_flags_missing_gold_study(tmp_path: Path) -> None:
    dicom_root = tmp_path / "dicoms"
    # S1 and S3 are present on disk; S2 (gold) is not extracted yet.
    for study, series in [("S1", "SER1"), ("S3", "SER3")]:
        path = dicom_root / study / series / "a.dcm"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not a real dicom, coverage only checks paths")

    audit_root = tmp_path / "audit"
    build_file_index(dicom_root, audit_root, num_shards=2)

    train_csv = tmp_path / "train.csv"
    _write_train_csv(train_csv, gold_uids={"S1", "S2"}, all_uids=["S1", "S2", "S3"])

    train_series_csv = tmp_path / "train_series.csv"
    _write_train_series_csv(
        train_series_csv,
        rows=[
            ("S1", "SER1", "Sagittal", "1"),
            ("S2", "SER2", "Coronal", "0"),
            ("S3", "SER3", "Axial", "1"),
        ],
    )

    result = compute_coverage(audit_root, train_csv=train_csv, train_series_csv=train_series_csv)

    assert result["studies_in_train_csv"] == 3
    assert result["gold_studies_in_train_csv"] == 2
    assert result["studies_on_disk"] == 2
    assert result["gold_studies_on_disk"] == 1
    assert result["missing_gold_study_count"] == 1
    assert result["gold_coverage_fraction"] == 0.5
    assert result["series_bucket_full_corpus"]["Coronal/fluid=0"] == 1
    assert "Coronal/fluid=0" not in result["series_bucket_on_disk"]

    missing = (audit_root / "issues" / "missing_gold_studies.csv").read_text(encoding="utf-8")
    assert "S2" in missing
    assert "S1" not in missing.split("\n")[1:]
