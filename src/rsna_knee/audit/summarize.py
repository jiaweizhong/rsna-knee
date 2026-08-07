from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rsna_knee.constants import LABEL_COLUMNS

from .common import atomic_text_writer, iter_jsonl, list_part_files


class NumericSummary:
    def __init__(self, seed: int = 2026, reservoir_size: int = 100_000) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.reservoir_size = reservoir_size
        self.reservoir: list[float] = []
        self.random = random.Random(seed)

    def add(self, value: Any) -> None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(number):
            return
        self.count += 1
        delta = number - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (number - self.mean)
        self.minimum = min(self.minimum, number)
        self.maximum = max(self.maximum, number)
        if len(self.reservoir) < self.reservoir_size:
            self.reservoir.append(number)
        else:
            index = self.random.randrange(self.count)
            if index < self.reservoir_size:
                self.reservoir[index] = number

    def result(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {"count": 0, "min": None, "mean": None, "std": None, "max": None}
        quantiles = np.percentile(self.reservoir, [1, 5, 25, 50, 75, 95, 99])
        result: dict[str, float | int | None] = {
            "count": self.count,
            "min": self.minimum,
            "mean": self.mean,
            "std": math.sqrt(self.m2 / max(1, self.count - 1)),
            "max": self.maximum,
        }
        result.update(
            {name: float(value) for name, value in zip(["p01", "p05", "p25", "p50", "p75", "p95", "p99"], quantiles)}
        )
        return result


@dataclass
class SeriesAggregate:
    study_uid: str
    series_uid: str
    patient_hash: str | None = None
    dicom_count: int = 0
    total_bytes: int = 0
    error_count: int = 0
    positions: list[float] = field(default_factory=list)
    instance_numbers: list[float] = field(default_factory=list)
    planes: Counter[str] = field(default_factory=Counter)
    rows: Counter[str] = field(default_factory=Counter)
    columns: Counter[str] = field(default_factory=Counter)
    transfer_syntaxes: Counter[str] = field(default_factory=Counter)

    def add(self, record: dict[str, Any]) -> None:
        self.dicom_count += 1
        self.total_bytes += int(record.get("file_size_bytes") or 0)
        self.error_count += int(record.get("status") != "ok")
        self.patient_hash = self.patient_hash or record.get("patient_hash")
        for key, target in [("Rows", self.rows), ("Columns", self.columns)]:
            if record.get(key) is not None:
                target[str(record[key])] += 1
        if record.get("derived_plane"):
            self.planes[str(record["derived_plane"])] += 1
        if record.get("TransferSyntaxUID"):
            self.transfer_syntaxes[str(record["TransferSyntaxUID"])] += 1
        for key, target in [
            ("position_scalar", self.positions),
            ("InstanceNumber", self.instance_numbers),
        ]:
            try:
                value = float(record[key])
                if math.isfinite(value):
                    target.append(value)
            except (KeyError, TypeError, ValueError):
                pass

    def result(self, series_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        sorted_positions = np.sort(np.asarray(self.positions, dtype=np.float64))
        differences = np.diff(sorted_positions)
        abs_differences = np.abs(differences)
        spacing_median = float(np.median(abs_differences)) if abs_differences.size else None
        spacing_cv = None
        if abs_differences.size and float(abs_differences.mean()) > 1e-8:
            spacing_cv = float(abs_differences.std() / abs_differences.mean())
        result = {
            "StudyInstanceUID": self.study_uid,
            "SeriesInstanceUID": self.series_uid,
            "patient_hash": self.patient_hash,
            "dicom_count": self.dicom_count,
            "total_bytes": self.total_bytes,
            "header_error_count": self.error_count,
            "position_count": len(self.positions),
            "duplicate_position_count": int(len(sorted_positions) - len(np.unique(sorted_positions))),
            "spacing_median": spacing_median,
            "spacing_cv": spacing_cv,
            "derived_plane_mode": self.planes.most_common(1)[0][0] if self.planes else "Unknown",
            "rows_unique": len(self.rows),
            "columns_unique": len(self.columns),
            "transfer_syntax_count": len(self.transfer_syntaxes),
        }
        if series_metadata:
            result.update(series_metadata)
        return result


def _load_series_metadata(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    frame = pd.read_csv(path)
    if "SeriesInstanceUID" not in frame.columns:
        raise KeyError("train_series.csv requires SeriesInstanceUID")
    return {
        str(row["SeriesInstanceUID"]): {
            key: row[key]
            for key in ["Fluid_Sensitive", "Anatomical_Plane"]
            if key in frame.columns
        }
        for row in frame.to_dict(orient="records")
    }


def _write_label_audit(train_csv: str | Path, tables_dir: Path) -> dict[str, Any]:
    frame = pd.read_csv(train_csv)
    present = [column for column in LABEL_COLUMNS if column in frame.columns]
    if not present:
        raise KeyError("No expected label columns found in train.csv")
    rows = []
    for column in present:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        rows.append(
            {
                "label": column,
                "positive": int((numeric == 1).sum()),
                "negative": int((numeric == 0).sum()),
                "missing": int(numeric.isna().sum()),
                "prevalence_observed": float(numeric.mean()) if numeric.notna().any() else None,
            }
        )
    pd.DataFrame(rows).to_parquet(tables_dir / "label_inventory.parquet", index=False)
    correlations = frame[present].apply(pd.to_numeric, errors="coerce").corr()
    correlations.to_csv(tables_dir / "label_correlation.csv")
    return {"rows": len(frame), "label_columns": present, "labels": rows}


def summarize_audit(
    audit_root: str | Path,
    train_csv: str | Path | None = None,
    train_series_csv: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(audit_root)
    header_parts = list_part_files(root / "headers")
    if not header_parts:
        raise FileNotFoundError(f"No header parts found under {root / 'headers'}")
    tables_dir = root / "tables"
    issues_dir = root / "issues"
    parquet_parts = tables_dir / "dicom_inventory_parts"
    tables_dir.mkdir(parents=True, exist_ok=True)
    issues_dir.mkdir(parents=True, exist_ok=True)
    parquet_parts.mkdir(parents=True, exist_ok=True)

    status_counts: Counter[str] = Counter()
    error_types: Counter[str] = Counter()
    plane_counts: Counter[str] = Counter()
    transfer_counts: Counter[str] = Counter()
    header_time = NumericSummary(seed=2026)
    file_size = NumericSummary(seed=2027)
    series: dict[tuple[str, str], SeriesAggregate] = {}
    study_series: defaultdict[str, set[str]] = defaultdict(set)
    study_dicoms: Counter[str] = Counter()
    study_bytes: Counter[str] = Counter()
    series_metadata = _load_series_metadata(train_series_csv)

    failures_path = issues_dir / "header_failures.csv"
    with atomic_text_writer(failures_path) as failure_handle:
        failure_writer = csv.DictWriter(
            failure_handle,
            fieldnames=["relative_path", "error_type", "error_message"],
        )
        failure_writer.writeheader()
        for part in header_parts:
            part_records: list[dict[str, Any]] = []
            for record in iter_jsonl(part):
                part_records.append(record)
                status = str(record.get("status", "unknown"))
                status_counts[status] += 1
                header_time.add(record.get("header_seconds"))
                file_size.add(record.get("file_size_bytes"))
                if status != "ok":
                    error_type = str(record.get("error_type", "Unknown"))
                    error_types[error_type] += 1
                    failure_writer.writerow(
                        {
                            "relative_path": record.get("relative_path"),
                            "error_type": error_type,
                            "error_message": record.get("error_message"),
                        }
                    )
                plane_counts[str(record.get("derived_plane", "Unknown"))] += 1
                if record.get("TransferSyntaxUID"):
                    transfer_counts[str(record["TransferSyntaxUID"])] += 1
                study_uid = str(record.get("StudyInstanceUID") or record.get("path_study_uid") or "")
                series_uid = str(record.get("SeriesInstanceUID") or record.get("path_series_uid") or "")
                if not study_uid or not series_uid:
                    continue
                key = (study_uid, series_uid)
                if key not in series:
                    series[key] = SeriesAggregate(study_uid=study_uid, series_uid=series_uid)
                series[key].add(record)
                study_series[study_uid].add(series_uid)
                study_dicoms[study_uid] += 1
                study_bytes[study_uid] += int(record.get("file_size_bytes") or 0)
            frame = pd.DataFrame.from_records(part_records)
            frame.to_parquet(parquet_parts / f"{part.stem}.parquet", index=False)

    series_rows = [
        aggregate.result(series_metadata.get(aggregate.series_uid))
        for aggregate in series.values()
    ]
    series_frame = pd.DataFrame.from_records(series_rows)
    series_frame.to_parquet(tables_dir / "series_inventory.parquet", index=False)
    study_rows = [
        {
            "StudyInstanceUID": study_uid,
            "series_count": len(series_ids),
            "dicom_count": study_dicoms[study_uid],
            "total_bytes": study_bytes[study_uid],
        }
        for study_uid, series_ids in study_series.items()
    ]
    pd.DataFrame.from_records(study_rows).to_parquet(
        tables_dir / "study_inventory.parquet", index=False
    )

    pixel_parts = list_part_files(root / "pixels")
    pixel_parquet_parts = tables_dir / "pixel_inventory_parts"
    pixel_parquet_parts.mkdir(parents=True, exist_ok=True)
    pixel_status: Counter[str] = Counter()
    pixel_errors: Counter[str] = Counter()
    decode_time = NumericSummary(seed=2028)
    with atomic_text_writer(issues_dir / "decode_failures.csv") as pixel_failure_handle:
        pixel_failure_writer = csv.DictWriter(
            pixel_failure_handle,
            fieldnames=["relative_path", "error_type", "error_message"],
        )
        pixel_failure_writer.writeheader()
        for part in pixel_parts:
            records = list(iter_jsonl(part))
            for record in records:
                pixel_status[str(record.get("status", "unknown"))] += 1
                decode_time.add(record.get("decode_seconds"))
                if record.get("status") != "ok":
                    error_type = str(record.get("error_type", "Unknown"))
                    pixel_errors[error_type] += 1
                    pixel_failure_writer.writerow(
                        {
                            "relative_path": record.get("relative_path"),
                            "error_type": error_type,
                            "error_message": record.get("error_message"),
                        }
                    )
            if records:
                pd.DataFrame.from_records(records).to_parquet(
                    pixel_parquet_parts / f"{part.stem}.parquet",
                    index=False,
                )

    label_summary = _write_label_audit(train_csv, tables_dir) if train_csv else None
    summary: dict[str, Any] = {
        "header_parts": len(header_parts),
        "status_counts": dict(status_counts),
        "error_types": dict(error_types),
        "plane_counts": dict(plane_counts),
        "transfer_syntax_counts": dict(transfer_counts),
        "header_seconds": header_time.result(),
        "file_size_bytes": file_size.result(),
        "studies": len(study_rows),
        "series": len(series_rows),
        "pixel_parts": len(pixel_parts),
        "pixel_status_counts": dict(pixel_status),
        "pixel_error_types": dict(pixel_errors),
        "decode_seconds": decode_time.result(),
        "labels": label_summary,
    }
    (root / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(root / "audit_report.md", summary)
    return summary


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    status = summary["status_counts"]
    pixel_status = summary["pixel_status_counts"]
    lines = [
        "# RSNA Knee MRI Data Audit Report",
        "",
        "> Auto-generated. Review all Critical issues before training.",
        "",
        "## Inventory",
        "",
        f"- Studies: {summary['studies']:,}",
        f"- Series: {summary['series']:,}",
        f"- Header records: {sum(status.values()):,}",
        f"- Header errors: {sum(v for k, v in status.items() if k != 'ok'):,}",
        f"- Pixel records: {sum(pixel_status.values()):,}",
        f"- Pixel errors: {sum(v for k, v in pixel_status.items() if k != 'ok'):,}",
        "",
        "## Geometry and protocol",
        "",
        f"- Derived planes: `{json.dumps(summary['plane_counts'], ensure_ascii=False)}`",
        f"- Transfer syntaxes: `{json.dumps(summary['transfer_syntax_counts'], ensure_ascii=False)}`",
        "",
        "## Performance",
        "",
        f"- Header seconds/file: `{json.dumps(summary['header_seconds'])}`",
        f"- Pixel decode seconds/file: `{json.dumps(summary['decode_seconds'])}`",
        "",
        "## Critical checks",
        "",
        "- [ ] Pixel decode errors reviewed and handled.",
        "- [ ] UID/path mismatches reviewed.",
        "- [ ] Geometry ordering montage reviewed.",
        "- [ ] Patient/duplicate-safe folds generated.",
        "- [ ] Normalization and series-slot policy frozen.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
