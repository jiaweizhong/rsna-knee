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


def _coerce_mixed_list_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """DICOM tags with VM 1-n (e.g. ScanningSequence, WindowCenter) decode to a bare
    scalar on single-valued files and a list on multi-valued ones. Arrow can't write a
    column that mixes the two, so force any column that does into list-or-None."""
    for column in frame.columns:
        series = frame[column]
        has_list = series.map(lambda value: isinstance(value, list)).any()
        has_scalar = series.map(lambda value: value is not None and not isinstance(value, list)).any()
        if has_list and has_scalar:
            frame[column] = series.map(lambda value: value if value is None or isinstance(value, list) else [value])
    return frame


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


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    rank_x = pd.Series(x).rank().to_numpy()
    rank_y = pd.Series(y).rank().to_numpy()
    if np.std(rank_x) == 0 or np.std(rank_y) == 0:
        return None
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


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
    position_instance_pairs: list[tuple[float, float]] = field(default_factory=list)
    planes: Counter[str] = field(default_factory=Counter)
    manufacturers: Counter[str] = field(default_factory=Counter)
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
        if record.get("Manufacturer"):
            self.manufacturers[str(record["Manufacturer"])] += 1
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
        try:
            position_value = float(record["position_scalar"])
            instance_value = float(record["InstanceNumber"])
            if math.isfinite(position_value) and math.isfinite(instance_value):
                self.position_instance_pairs.append((position_value, instance_value))
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
        derived_plane_mode = self.planes.most_common(1)[0][0] if self.planes else "Unknown"
        metadata_plane = (series_metadata or {}).get("Anatomical_Plane")
        plane_conflict = bool(metadata_plane) and str(metadata_plane) != derived_plane_mode
        paired_positions = [pair[0] for pair in self.position_instance_pairs]
        paired_instances = [pair[1] for pair in self.position_instance_pairs]
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
            "derived_plane_mode": derived_plane_mode,
            "manufacturer_mode": self.manufacturers.most_common(1)[0][0] if self.manufacturers else None,
            "plane_conflict": plane_conflict,
            "instance_geometry_spearman": _spearman(paired_positions, paired_instances),
            "instance_geometry_pair_count": len(self.position_instance_pairs),
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


def _bootstrap_ci(values: np.ndarray, num_resamples: int = 2000, seed: int = 2026) -> tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(num_resamples, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


# DICOM Manufacturer is free text and varies by software era (e.g. "SIEMENS" vs
# "Siemens Healthineers" are the same vendor). Raw values stay untouched everywhere
# else; this grouping exists only to keep the domain-correlation buckets meaningful.
_MANUFACTURER_FAMILIES = [
    ("siemens", "Siemens"),
    ("philips", "Philips"),
    ("ge medical", "GE"),
    ("gehc", "GE"),
    ("toshiba", "Canon/Toshiba"),
    ("canon", "Canon/Toshiba"),
    ("fujifilm", "Fujifilm/Hitachi"),
    ("hitachi", "Fujifilm/Hitachi"),
]


def _manufacturer_family(raw: str | None) -> str:
    if not raw:
        return "Unknown"
    lowered = raw.lower()
    for needle, family in _MANUFACTURER_FAMILIES:
        if needle in lowered:
            return family
    return raw


def _write_label_audit(
    train_csv: str | Path,
    tables_dir: Path,
    study_domain: dict[str, str] | None = None,
) -> dict[str, Any]:
    frame = pd.read_csv(train_csv)
    present = [column for column in LABEL_COLUMNS if column in frame.columns]
    if not present:
        raise KeyError("No expected label columns found in train.csv")

    numeric_frame = frame[present].apply(pd.to_numeric, errors="coerce")
    is_gold = numeric_frame.notna().all(axis=1)
    gold_frame = numeric_frame.loc[is_gold]

    rows = []
    for column in present:
        numeric = numeric_frame[column]
        gold_values = gold_frame[column].to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap_ci(gold_values)
        rows.append(
            {
                "label": column,
                "positive": int((numeric == 1).sum()),
                "negative": int((numeric == 0).sum()),
                "missing": int(numeric.isna().sum()),
                "prevalence_observed": float(numeric.mean()) if numeric.notna().any() else None,
                "prevalence_ci_low": ci_low if gold_values.size else None,
                "prevalence_ci_high": ci_high if gold_values.size else None,
            }
        )
    pd.DataFrame(rows).to_parquet(tables_dir / "label_inventory.parquet", index=False)

    correlations = numeric_frame.corr()
    correlations.to_csv(tables_dir / "label_correlation.csv")

    # Co-occurrence and Jaccard are computed on the gold-complete rows only: with
    # non-gold rows entirely NaN, mixing them in would silently zero out every pair.
    cooccurrence = pd.DataFrame(0, index=present, columns=present, dtype=int)
    jaccard = pd.DataFrame(0.0, index=present, columns=present, dtype=float)
    positive_masks = {column: (gold_frame[column] == 1) for column in present}
    for a in present:
        for b in present:
            both = int((positive_masks[a] & positive_masks[b]).sum())
            union = int((positive_masks[a] | positive_masks[b]).sum())
            cooccurrence.loc[a, b] = both
            jaccard.loc[a, b] = (both / union) if union else 0.0
    cooccurrence.to_csv(tables_dir / "label_cooccurrence.csv")
    jaccard.to_csv(tables_dir / "label_jaccard.csv")

    # Shortcut-risk check (Image-Audit-Plan 10.2): does label prevalence track scanner
    # vendor. Gold n=58 split across domain buckets is too thin for a reliable odds
    # ratio; this is a flag to revisit on the larger weak-label corpus, not a verdict.
    domain_breakdown: dict[str, Any] | None = None
    if study_domain and "StudyInstanceUID" in frame.columns:
        gold_uids = frame.loc[is_gold, "StudyInstanceUID"].astype(str)
        families = gold_uids.map(lambda uid: _manufacturer_family(study_domain.get(uid))).to_numpy()
        domain_gold_frame = gold_frame.copy()
        domain_gold_frame["_domain_family"] = families
        domain_rows = []
        for label in present:
            for family, group in domain_gold_frame.groupby("_domain_family"):
                n = len(group)
                positive = int((group[label] == 1).sum())
                domain_rows.append(
                    {
                        "label": label,
                        "domain": family,
                        "n": n,
                        "positive": positive,
                        "prevalence": (positive / n) if n else None,
                    }
                )
        pd.DataFrame(domain_rows).to_csv(tables_dir / "label_domain_correlation.csv", index=False)
        domain_breakdown = {
            "caveat": "gold n=58 spread across domain buckets is too small for a reliable "
            "odds ratio; use only as a shortcut-risk flag, recompute on the weak-label corpus.",
            "rows": domain_rows,
        }

    positive_count_per_study = gold_frame.sum(axis=1)
    positive_count_distribution = {
        int(count): int(freq)
        for count, freq in positive_count_per_study.value_counts().sort_index().items()
    }

    return {
        "rows": len(frame),
        "label_columns": present,
        "labels": rows,
        "gold_study_count": int(is_gold.sum()),
        "positive_label_count_distribution": positive_count_distribution,
        "domain_breakdown": domain_breakdown,
    }


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
    manufacturer_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    header_time = NumericSummary(seed=2026)
    file_size = NumericSummary(seed=2027)
    field_strength = NumericSummary(seed=2029)
    series: dict[tuple[str, str], SeriesAggregate] = {}
    study_series: defaultdict[str, set[str]] = defaultdict(set)
    study_dicoms: Counter[str] = Counter()
    study_bytes: Counter[str] = Counter()
    study_manufacturers: defaultdict[str, Counter[str]] = defaultdict(Counter)
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
                if record.get("Manufacturer"):
                    manufacturer_counts[str(record["Manufacturer"])] += 1
                if record.get("ManufacturerModelName"):
                    model_counts[str(record["ManufacturerModelName"])] += 1
                field_strength.add(record.get("MagneticFieldStrength"))
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
                if record.get("Manufacturer"):
                    study_manufacturers[study_uid][str(record["Manufacturer"])] += 1
            frame = _coerce_mixed_list_columns(pd.DataFrame.from_records(part_records))
            frame.to_parquet(parquet_parts / f"{part.stem}.parquet", index=False)

    series_rows = [
        aggregate.result(series_metadata.get(aggregate.series_uid))
        for aggregate in series.values()
    ]
    series_frame = pd.DataFrame.from_records(series_rows)
    series_frame.to_parquet(tables_dir / "series_inventory.parquet", index=False)

    # Geometry stats needed for the 2D/2.5D/3D call (Image-Audit-Plan 13.1): how
    # consistent is inter-slice spacing, and how many slices does a series actually have.
    slice_count_summary = NumericSummary(seed=2030)
    spacing_median_summary = NumericSummary(seed=2031)
    spacing_cv_summary = NumericSummary(seed=2032)
    for row in series_rows:
        slice_count_summary.add(row.get("dicom_count"))
        spacing_median_summary.add(row.get("spacing_median"))
        spacing_cv_summary.add(row.get("spacing_cv"))

    # A2 ordering checks (Image-Audit-Plan 7.2): does InstanceNumber agree with the
    # geometry-derived order, and does the dataset's own Anatomical_Plane label agree
    # with the plane we derived from ImageOrientationPatient. Flagged rows, not deleted.
    instance_geometry_spearman_summary = NumericSummary(seed=2033)
    plane_conflict_count = 0
    duplicate_position_series_count = 0
    with atomic_text_writer(issues_dir / "geometry_failures.csv") as geometry_handle:
        geometry_writer = csv.DictWriter(
            geometry_handle,
            fieldnames=[
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "issue",
                "derived_plane_mode",
                "metadata_plane",
                "instance_geometry_spearman",
                "duplicate_position_count",
            ],
        )
        geometry_writer.writeheader()
        for row in series_rows:
            spearman = row.get("instance_geometry_spearman")
            instance_geometry_spearman_summary.add(spearman)
            issues = []
            if row.get("plane_conflict"):
                plane_conflict_count += 1
                issues.append("plane_conflict")
            if (row.get("duplicate_position_count") or 0) > 0:
                duplicate_position_series_count += 1
                issues.append("duplicate_position")
            if spearman is not None and abs(spearman) < 0.9 and row.get("instance_geometry_pair_count", 0) >= 3:
                issues.append("instance_number_disagrees_with_geometry")
            if issues:
                geometry_writer.writerow(
                    {
                        "StudyInstanceUID": row.get("StudyInstanceUID"),
                        "SeriesInstanceUID": row.get("SeriesInstanceUID"),
                        "issue": ";".join(issues),
                        "derived_plane_mode": row.get("derived_plane_mode"),
                        "metadata_plane": row.get("Anatomical_Plane"),
                        "instance_geometry_spearman": spearman,
                        "duplicate_position_count": row.get("duplicate_position_count"),
                    }
                )

    study_manufacturer_mode: dict[str, str] = {
        study_uid: counter.most_common(1)[0][0]
        for study_uid, counter in study_manufacturers.items()
        if counter
    }

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
                _coerce_mixed_list_columns(pd.DataFrame.from_records(records)).to_parquet(
                    pixel_parquet_parts / f"{part.stem}.parquet",
                    index=False,
                )

    label_summary = (
        _write_label_audit(train_csv, tables_dir, study_domain=study_manufacturer_mode)
        if train_csv
        else None
    )
    summary: dict[str, Any] = {
        "header_parts": len(header_parts),
        "status_counts": dict(status_counts),
        "error_types": dict(error_types),
        "plane_counts": dict(plane_counts),
        "transfer_syntax_counts": dict(transfer_counts),
        "manufacturer_counts": dict(manufacturer_counts),
        "manufacturer_model_counts": dict(model_counts),
        "magnetic_field_strength_tesla": field_strength.result(),
        "header_seconds": header_time.result(),
        "file_size_bytes": file_size.result(),
        "studies": len(study_rows),
        "series": len(series_rows),
        "series_slice_count": slice_count_summary.result(),
        "series_spacing_median_mm": spacing_median_summary.result(),
        "series_spacing_cv": spacing_cv_summary.result(),
        "geometry_issues": {
            "plane_conflict_series": plane_conflict_count,
            "duplicate_position_series": duplicate_position_series_count,
            "instance_geometry_spearman": instance_geometry_spearman_summary.result(),
        },
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


def _label_report_lines(label_summary: dict[str, Any] | None) -> list[str]:
    if not label_summary:
        return []
    lines = [
        "## Labels",
        "",
        f"- Gold-labeled studies (all 12 labels present): "
        f"{label_summary['gold_study_count']:,} / {label_summary['rows']:,}",
        f"- Positive-label-count distribution (gold studies): "
        f"`{json.dumps(label_summary['positive_label_count_distribution'])}`",
        "",
        "| label | positive | negative | missing | prevalence | 95% CI |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in label_summary["labels"]:
        prevalence = (
            f"{row['prevalence_observed']:.3f}" if row["prevalence_observed"] is not None else "n/a"
        )
        ci = (
            f"[{row['prevalence_ci_low']:.3f}, {row['prevalence_ci_high']:.3f}]"
            if row.get("prevalence_ci_low") is not None
            else "n/a"
        )
        lines.append(
            f"| {row['label']} | {row['positive']} | {row['negative']} | {row['missing']} "
            f"| {prevalence} | {ci} |"
        )
    lines.append("")
    domain = label_summary.get("domain_breakdown")
    if domain:
        lines += [
            "### Label vs. scanner-vendor (shortcut-risk check)",
            "",
            f"> {domain['caveat']}",
            "",
            "| label | domain | n | positive | prevalence |",
            "|---|---|---:|---:|---:|",
        ]
        for row in domain["rows"]:
            prevalence = f"{row['prevalence']:.3f}" if row["prevalence"] is not None else "n/a"
            lines.append(
                f"| {row['label']} | {row['domain']} | {row['n']} | {row['positive']} | {prevalence} |"
            )
        lines.append("")
    return lines


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
        f"- Manufacturers: `{json.dumps(summary['manufacturer_counts'], ensure_ascii=False)}`",
        f"- Manufacturer models: `{json.dumps(summary['manufacturer_model_counts'], ensure_ascii=False)}`",
        f"- Magnetic field strength (Tesla): `{json.dumps(summary['magnetic_field_strength_tesla'])}`",
        f"- Series slice count: `{json.dumps(summary['series_slice_count'])}`",
        f"- Series spacing median (mm): `{json.dumps(summary['series_spacing_median_mm'])}`",
        f"- Series spacing CV: `{json.dumps(summary['series_spacing_cv'])}`",
        f"- Geometry issues: `{json.dumps(summary['geometry_issues'])}`"
        " (see issues/geometry_failures.csv for the flagged series)",
        "",
        "## Performance",
        "",
        f"- Header seconds/file: `{json.dumps(summary['header_seconds'])}`",
        f"- Pixel decode seconds/file: `{json.dumps(summary['decode_seconds'])}`",
        "",
        *_label_report_lines(summary.get("labels")),
        "## Critical checks",
        "",
        "- [ ] Pixel decode errors reviewed and handled.",
        "- [ ] UID/path mismatches reviewed.",
        "- [ ] Geometry ordering montage reviewed.",
        "- [ ] Patient/duplicate-safe folds generated.",
        "- [ ] Normalization and series-slot policy frozen.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
