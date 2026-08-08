from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from rsna_knee.constants import LABEL_COLUMNS

from .common import atomic_text_writer, iter_jsonl


def _studies_and_series_on_disk(audit_root: Path) -> tuple[set[str], set[tuple[str, str]]]:
    inventory_path = audit_root / "index" / "files.jsonl"
    if not inventory_path.exists():
        raise FileNotFoundError(f"Run the index stage first: {inventory_path}")
    studies: set[str] = set()
    series: set[tuple[str, str]] = set()
    for record in iter_jsonl(inventory_path):
        parts = Path(record["relative_path"]).parts
        if len(parts) >= 1:
            studies.add(parts[0])
        if len(parts) >= 2:
            series.add((parts[0], parts[1]))
    return studies, series


def _column_or_unknown(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column].fillna("Unknown").astype(str)
    return pd.Series(["Unknown"] * len(frame), index=frame.index)


def _series_bucket_counts(frame: pd.DataFrame) -> dict[str, int]:
    plane = _column_or_unknown(frame, "Anatomical_Plane")
    fluid = _column_or_unknown(frame, "Fluid_Sensitive")
    return dict(Counter(f"{p}/fluid={f}" for p, f in zip(plane, fluid)))


def compute_coverage(
    audit_root: str | Path,
    train_csv: str | Path,
    train_series_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Compare studies present on disk (from the index stage) against train.csv,
    prioritizing gold-labeled studies. Useful when disk space forces a partial
    extraction of the training corpus and gold coverage must be verified first."""
    root = Path(audit_root)
    studies_on_disk, _series_on_disk = _studies_and_series_on_disk(root)

    train = pd.read_csv(train_csv)
    if "StudyInstanceUID" not in train.columns:
        raise KeyError("train.csv requires StudyInstanceUID")
    train = train.astype({"StudyInstanceUID": str})
    present_labels = [column for column in LABEL_COLUMNS if column in train.columns]
    if not present_labels:
        raise KeyError("No expected label columns found in train.csv")

    is_gold = train[present_labels].notna().all(axis=1)
    gold_uids = set(train.loc[is_gold, "StudyInstanceUID"])
    all_uids = set(train["StudyInstanceUID"])

    gold_on_disk = gold_uids & studies_on_disk
    missing_gold = sorted(gold_uids - studies_on_disk)

    result: dict[str, Any] = {
        "studies_in_train_csv": len(all_uids),
        "gold_studies_in_train_csv": len(gold_uids),
        "studies_on_disk": len(studies_on_disk & all_uids),
        "studies_on_disk_unrecognized": len(studies_on_disk - all_uids),
        "gold_studies_on_disk": len(gold_on_disk),
        "gold_coverage_fraction": (len(gold_on_disk) / len(gold_uids)) if gold_uids else None,
        "overall_coverage_fraction": (
            len(studies_on_disk & all_uids) / len(all_uids) if all_uids else None
        ),
        "missing_gold_study_count": len(missing_gold),
    }

    tables_dir = root / "tables"
    issues_dir = root / "issues"
    tables_dir.mkdir(parents=True, exist_ok=True)
    issues_dir.mkdir(parents=True, exist_ok=True)

    with atomic_text_writer(issues_dir / "missing_gold_studies.csv") as handle:
        handle.write("StudyInstanceUID\n")
        for uid in missing_gold:
            handle.write(f"{uid}\n")

    if train_series_csv:
        series_frame = pd.read_csv(train_series_csv)
        required = {"StudyInstanceUID", "SeriesInstanceUID"}
        if not required.issubset(series_frame.columns):
            raise KeyError("train_series.csv requires StudyInstanceUID and SeriesInstanceUID")
        series_frame = series_frame.astype({"StudyInstanceUID": str, "SeriesInstanceUID": str})
        present_mask = series_frame["StudyInstanceUID"].isin(studies_on_disk)

        result["series_bucket_full_corpus"] = _series_bucket_counts(series_frame)
        result["series_bucket_on_disk"] = _series_bucket_counts(series_frame.loc[present_mask])

    (root / "coverage_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
