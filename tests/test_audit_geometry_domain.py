import csv
import json
from pathlib import Path

from rsna_knee.audit.summarize import summarize_audit
from rsna_knee.constants import LABEL_COLUMNS


def _header_record(study, series, path, plane, position, instance, manufacturer):
    return {
        "relative_path": f"{study}/{series}/{path}",
        "status": "ok",
        "StudyInstanceUID": study,
        "SeriesInstanceUID": series,
        "derived_plane": plane,
        "position_scalar": position,
        "InstanceNumber": instance,
        "Manufacturer": manufacturer,
        "file_size_bytes": 10,
    }


def _write_headers(headers_dir: Path) -> None:
    records = [
        # Clean series: geometry order matches InstanceNumber order, plane agrees.
        _header_record("S1", "SER-clean", "a.dcm", "Sagittal", 0.0, 1, "SIEMENS"),
        _header_record("S1", "SER-clean", "b.dcm", "Sagittal", 10.0, 2, "SIEMENS"),
        _header_record("S1", "SER-clean", "c.dcm", "Sagittal", 20.0, 3, "SIEMENS"),
        # Bad order: InstanceNumber does not track geometric position.
        _header_record("S2", "SER-badorder", "a.dcm", "Sagittal", 0.0, 1, "Philips Medical Systems"),
        _header_record("S2", "SER-badorder", "b.dcm", "Sagittal", 10.0, 3, "Philips Medical Systems"),
        _header_record("S2", "SER-badorder", "c.dcm", "Sagittal", 20.0, 2, "Philips Medical Systems"),
        # Plane conflict (geometry says Axial, dataset metadata says Sagittal) plus a
        # duplicate position.
        _header_record("S2", "SER-planeconflict", "a.dcm", "Axial", 0.0, 1, "Philips Medical Systems"),
        _header_record("S2", "SER-planeconflict", "b.dcm", "Axial", 0.0, 2, "Philips Medical Systems"),
        _header_record("S2", "SER-planeconflict", "c.dcm", "Axial", 20.0, 3, "Philips Medical Systems"),
    ]
    headers_dir.mkdir(parents=True)
    (headers_dir / "part-00000.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def _write_train_csv(path: Path) -> None:
    gold = {"S1": {"ACL": 1}, "S2": {"MCL": 1}}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["StudyInstanceUID", "Report", *LABEL_COLUMNS])
        for uid, overrides in gold.items():
            writer.writerow([uid, "report", *[overrides.get(c, 0) for c in LABEL_COLUMNS]])


def _write_train_series_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["StudyInstanceUID", "SeriesInstanceUID", "Anatomical_Plane", "Fluid_Sensitive"])
        writer.writerow(["S1", "SER-clean", "Sagittal", "1"])
        writer.writerow(["S2", "SER-badorder", "Sagittal", "1"])
        writer.writerow(["S2", "SER-planeconflict", "Sagittal", "1"])  # dataset says Sagittal, geometry says Axial


def test_geometry_and_domain_checks(tmp_path: Path) -> None:
    audit_root = tmp_path / "audit"
    _write_headers(audit_root / "headers")
    train_csv = tmp_path / "train.csv"
    _write_train_csv(train_csv)
    train_series_csv = tmp_path / "train_series.csv"
    _write_train_series_csv(train_series_csv)

    summary = summarize_audit(audit_root, train_csv=train_csv, train_series_csv=train_series_csv)

    geometry = summary["geometry_issues"]
    assert geometry["plane_conflict_series"] == 1
    assert geometry["duplicate_position_series"] == 1
    assert geometry["instance_geometry_spearman"]["count"] == 3

    failures = (audit_root / "issues" / "geometry_failures.csv").read_text(encoding="utf-8")
    assert "SER-badorder" in failures
    assert "instance_number_disagrees_with_geometry" in failures
    assert "SER-planeconflict" in failures
    assert "plane_conflict" in failures
    assert "duplicate_position" in failures
    assert "SER-clean" not in failures  # the well-behaved series must not be flagged

    domain = summary["labels"]["domain_breakdown"]
    assert domain is not None
    families = {row["domain"] for row in domain["rows"]}
    assert families == {"Siemens", "Philips"}
    acl_siemens = next(r for r in domain["rows"] if r["label"] == "ACL" and r["domain"] == "Siemens")
    assert acl_siemens == {"label": "ACL", "domain": "Siemens", "n": 1, "positive": 1, "prevalence": 1.0}
