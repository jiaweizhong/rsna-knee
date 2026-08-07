import json
from pathlib import Path

import pandas as pd

from rsna_knee.data.manifest import build_study_manifest
from rsna_knee.data.split import split_manifest


def test_build_study_manifest_orders_slices_and_builds_windows(tmp_path: Path) -> None:
    audit_root = tmp_path / "audit"
    headers = audit_root / "headers"
    headers.mkdir(parents=True)
    records = []
    for index, position in enumerate([2.0, 0.0, 1.0]):
        records.append(
            {
                "status": "ok",
                "StudyInstanceUID": "study-1",
                "SeriesInstanceUID": "series-1",
                "SOPInstanceUID": f"sop-{index}",
                "relative_path": f"study-1/series-1/slice-{index}.dcm",
                "position_scalar": position,
                "InstanceNumber": index,
                "derived_plane": "Sagittal",
                "patient_hash": "patient-1",
            }
        )
    (headers / "part-00000.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    labels_csv = tmp_path / "train.csv"
    pd.DataFrame([{"StudyInstanceUID": "study-1", "ACL": 1}]).to_csv(labels_csv, index=False)
    series_csv = tmp_path / "train_series.csv"
    pd.DataFrame(
        [
            {
                "StudyInstanceUID": "study-1",
                "SeriesInstanceUID": "series-1",
                "Fluid_Sensitive": 1,
                "Anatomical_Plane": "Sagittal",
            }
        ]
    ).to_csv(series_csv, index=False)
    output = tmp_path / "manifest.jsonl"
    summary = build_study_manifest(
        audit_root,
        output,
        labels_csv=labels_csv,
        series_csv=series_csv,
        max_windows_per_series=3,
        neighbor_offsets=[-1, 0, 1],
    )
    record = json.loads(output.read_text(encoding="utf-8"))
    assert summary == {"studies": 1, "series": 1, "windows": 3}
    assert record["patient_hash"] == "patient-1"
    assert record["windows"][1]["paths"] == [
        "study-1/series-1/slice-1.dcm",
        "study-1/series-1/slice-2.dcm",
        "study-1/series-1/slice-0.dcm",
    ]
    assert record["labels"][0] == 1.0


def test_patient_grouped_split_has_no_patient_overlap(tmp_path: Path) -> None:
    manifest = tmp_path / "all.jsonl"
    records = []
    for index in range(10):
        labels = [0.0] * 12
        labels[index % 3] = 1.0
        records.append(
            {
                "study_id": f"study-{index}",
                "patient_hash": f"patient-{index // 2}",
                "labels": labels,
                "windows": [{"paths": ["x", "x", "x"]}],
            }
        )
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    output = tmp_path / "folds"
    split_manifest(manifest, output, folds=3, seed=1)
    assignment = {}
    for fold in range(3):
        valid = [
            json.loads(line)
            for line in (output / f"fold-{fold}" / "valid.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        for record in valid:
            patient = record["patient_hash"]
            assert patient not in assignment or assignment[patient] == fold
            assignment[patient] = fold
