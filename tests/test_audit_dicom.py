import functools
import json
from pathlib import Path

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")

from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from rsna_knee.audit.dicom import read_header_record
from rsna_knee.audit.cli import _run_partitioned_stage
from rsna_knee.audit.index import build_file_index
from rsna_knee.audit.pixels import read_pixel_record
from rsna_knee.audit.summarize import summarize_audit


def _write_dicom(path: Path) -> None:
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.ImplementationClassUID = generate_uid()
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.is_little_endian = True
    dataset.is_implicit_VR = False
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.SOPClassUID = MRImageStorage
    dataset.PatientID = "secret-patient"
    dataset.Modality = "MR"
    dataset.Rows = 8
    dataset.Columns = 8
    dataset.InstanceNumber = 1
    dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    dataset.ImagePositionPatient = [0, 0, 2.5]
    dataset.PixelSpacing = [0.5, 0.5]
    dataset.SliceThickness = 3.0
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.PixelData = np.arange(64, dtype=np.uint16).reshape(8, 8).tobytes()
    dataset.save_as(path, enforce_file_format=True)


def test_multi_value_keyword_normalized_to_list(tmp_path: Path) -> None:
    path = tmp_path / "study" / "series" / "slice.dcm"
    path.parent.mkdir(parents=True)
    _write_dicom(path)
    import pydicom

    dataset = pydicom.dcmread(path, force=True)
    dataset.ScanningSequence = "SE"  # single-valued on this file; VM 1-n in general
    dataset.save_as(path, enforce_file_format=True)

    header = read_header_record(str(path), "study/series/slice.dcm", patient_salt="salt")
    assert header["status"] == "ok"
    assert header["ScanningSequence"] == ["SE"]


def test_header_and_pixel_audit(tmp_path: Path) -> None:
    path = tmp_path / "study" / "series" / "slice.dcm"
    path.parent.mkdir(parents=True)
    _write_dicom(path)
    header = read_header_record(str(path), "study/series/slice.dcm", patient_salt="salt")
    assert header["status"] == "ok"
    assert header["derived_plane"] == "Axial"
    assert header["position_scalar"] == pytest.approx(2.5)
    assert "PatientID" not in header
    assert header["patient_hash"]
    pixels = read_pixel_record(str(path), "study/series/slice.dcm", deep=True, hash_pixels=True)
    assert pixels["status"] == "ok"
    assert pixels["pixel_min"] == 0
    assert pixels["pixel_max"] == 63
    assert pixels["pixel_sha256"]


def test_small_audit_pipeline(tmp_path: Path) -> None:
    dicom_root = tmp_path / "dicoms"
    for index in range(3):
        path = dicom_root / "study" / f"series-{index}" / "slice.dcm"
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_dicom(path)
    audit_root = tmp_path / "audit"
    state = build_file_index(dicom_root, audit_root, num_shards=2)
    assert state["total_files"] == 3
    headers = audit_root / "headers"
    pixels = audit_root / "pixels"
    headers.mkdir()
    pixels.mkdir()
    for shard in range(2):
        source = audit_root / "index" / "shard_paths" / f"part-{shard:05d}.txt"
        relative_paths = [line for line in source.read_text(encoding="utf-8").splitlines() if line]
        header_records = [
            read_header_record(str(dicom_root / relative), relative, patient_salt="salt")
            for relative in relative_paths
        ]
        pixel_records = [
            read_pixel_record(str(dicom_root / relative), relative, deep=True, hash_pixels=False)
            for relative in relative_paths
        ]
        (headers / f"part-{shard:05d}.jsonl").write_text(
            "".join(__import__("json").dumps(record) + "\n" for record in header_records),
            encoding="utf-8",
        )
        (pixels / f"part-{shard:05d}.jsonl").write_text(
            "".join(__import__("json").dumps(record) + "\n" for record in pixel_records),
            encoding="utf-8",
        )
    summary = summarize_audit(audit_root)
    assert summary["status_counts"] == {"ok": 3}
    assert summary["pixel_status_counts"] == {"ok": 3}
    assert (audit_root / "tables" / "series_inventory.parquet").exists()
    assert (audit_root / "audit_report.md").exists()


def test_summarize_survives_mixed_scalar_and_list_columns(tmp_path: Path) -> None:
    # Reproduces the real failure: ScanningSequence decodes to a bare string on some
    # files and a list on others, which pyarrow refuses to write as one column.
    audit_root = tmp_path / "audit"
    headers_dir = audit_root / "headers"
    headers_dir.mkdir(parents=True)
    records = [
        {
            "relative_path": "study/series-0/a.dcm",
            "status": "ok",
            "StudyInstanceUID": "study",
            "SeriesInstanceUID": "series-0",
            "ScanningSequence": "SE",
            "file_size_bytes": 10,
        },
        {
            "relative_path": "study/series-1/b.dcm",
            "status": "ok",
            "StudyInstanceUID": "study",
            "SeriesInstanceUID": "series-1",
            "ScanningSequence": ["SE", "IR"],
            "file_size_bytes": 20,
        },
    ]
    (headers_dir / "part-00000.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    summary = summarize_audit(audit_root)
    assert summary["status_counts"] == {"ok": 2}
    assert (audit_root / "tables" / "dicom_inventory_parts" / "part-00000.parquet").exists()


def test_partitioned_header_stage_with_processes(tmp_path: Path) -> None:
    dicom_root = tmp_path / "dicoms"
    for index in range(2):
        path = dicom_root / "study" / f"series-{index}" / "slice.dcm"
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_dicom(path)
    audit_root = tmp_path / "audit"
    build_file_index(dicom_root, audit_root, num_shards=2)

    def factory(_):
        return functools.partial(read_header_record, patient_salt="salt", force=True)

    _run_partitioned_stage(
        audit_root,
        stage="headers",
        shards="all",
        workers=2,
        worker_factory=factory,
        force=False,
    )
    records = []
    for path in sorted((audit_root / "headers").glob("part-*.jsonl")):
        records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    assert len(records) == 2
    assert all(record["status"] == "ok" for record in records)
