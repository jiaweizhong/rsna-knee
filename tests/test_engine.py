import json
from pathlib import Path

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")

from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from rsna_knee.engine import train


def _write_slice(path: Path, offset: int) -> None:
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.ImplementationClassUID = generate_uid()
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = MRImageStorage
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.Rows = 16
    dataset.Columns = 16
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    pixels = np.arange(256, dtype=np.uint16).reshape(16, 16) + offset
    dataset.PixelData = pixels.tobytes()
    dataset.save_as(path, enforce_file_format=True)


def test_one_epoch_training_smoke(tmp_path: Path) -> None:
    dicom_root = tmp_path / "dicoms"
    records = []
    for study_index in range(4):
        paths = []
        for slice_index in range(3):
            relative = f"study-{study_index}/series/slice-{slice_index}.dcm"
            path = dicom_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_slice(path, study_index * 10 + slice_index)
            paths.append(relative)
        value = float(study_index % 2)
        records.append(
            {
                "study_id": f"study-{study_index}",
                "patient_hash": f"patient-{study_index}",
                "labels": [value if label % 2 == 0 else 1.0 - value for label in range(12)],
                "windows": [
                    {
                        "paths": paths,
                        "series_id": "series",
                        "plane": "Sagittal",
                        "fluid_sensitive": 1,
                        "relative_position": 0.5,
                    }
                ],
            }
        )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    data_spec = {
        "manifest_path": str(manifest),
        "dicom_root": str(dicom_root),
        "image_size": [32, 32],
        "normalization": "per_slice_robust",
        "max_candidates": 3,
        "on_decode_error": "raise",
    }
    config = {
        "seed": 1,
        "output_dir": str(tmp_path / "runs"),
        "run_name": "smoke",
        "epochs": 1,
        "precision": "fp32",
        "compile": False,
        "gradient_accumulation": 1,
        "data": {"train": data_spec, "valid": data_spec},
        "loader": {
            "train": {"batch_size": 2, "num_workers": 0, "pin_memory": False},
            "valid": {"batch_size": 2, "num_workers": 0, "pin_memory": False},
        },
        "model": {
            "num_labels": 12,
            "in_channels": 3,
            "metadata_dim": 5,
            "backbone": {"name": "tiny_cnn", "params": {"out_dim": 16, "width": 4}},
            "selector": {"name": "uniform", "params": {"k": 1}},
            "aggregator": {"name": "mean_max", "params": {"hidden_dim": 16}},
        },
        "loss": {"bce_weight": 1.0},
        "optimizer": {"lr": 0.001, "weight_decay": 0.0},
        "scheduler": {"warmup_steps": 0},
    }
    metrics = train(config)
    assert metrics["macro_auc"] is not None
    assert (tmp_path / "runs" / "smoke" / "best.pt").exists()
    assert (tmp_path / "runs" / "smoke" / "config.resolved.yaml").exists()

