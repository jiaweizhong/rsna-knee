from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from rsna_knee.audit.common import atomic_text_writer, iter_jsonl, json_dumps, list_part_files
from rsna_knee.constants import LABEL_COLUMNS


def _number(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _create_slice_db(audit_root: Path, database_path: Path, rebuild: bool) -> sqlite3.Connection:
    if rebuild and database_path.exists():
        database_path.unlink()
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS slices (
            study_uid TEXT NOT NULL,
            series_uid TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            position_scalar REAL,
            instance_number REAL,
            derived_plane TEXT,
            patient_hash TEXT,
            PRIMARY KEY (relative_path)
        )
        """
    )
    count = connection.execute("SELECT COUNT(*) FROM slices").fetchone()[0]
    if count:
        return connection
    parts = list_part_files(audit_root / "headers")
    if not parts:
        raise FileNotFoundError(f"No audit header parts under {audit_root / 'headers'}")
    insert_sql = "INSERT OR REPLACE INTO slices VALUES (?, ?, ?, ?, ?, ?, ?)"
    batch: list[tuple[Any, ...]] = []
    for part in parts:
        for record in iter_jsonl(part):
            if record.get("status") != "ok":
                continue
            study_uid = record.get("StudyInstanceUID") or record.get("path_study_uid")
            series_uid = record.get("SeriesInstanceUID") or record.get("path_series_uid")
            relative_path = record.get("relative_path")
            if not study_uid or not series_uid or not relative_path:
                continue
            batch.append(
                (
                    str(study_uid),
                    str(series_uid),
                    str(relative_path),
                    _number(record.get("position_scalar")),
                    _number(record.get("InstanceNumber")),
                    str(record.get("derived_plane") or "Unknown"),
                    record.get("patient_hash"),
                )
            )
            if len(batch) >= 10_000:
                connection.executemany(insert_sql, batch)
                connection.commit()
                batch.clear()
    if batch:
        connection.executemany(insert_sql, batch)
        connection.commit()
    connection.execute("CREATE INDEX IF NOT EXISTS idx_slices_order ON slices(study_uid, series_uid)")
    connection.commit()
    return connection


def _load_series_csv(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    frame = pd.read_csv(path)
    if "SeriesInstanceUID" not in frame.columns:
        raise KeyError("series CSV requires SeriesInstanceUID")
    output = {}
    for row in frame.to_dict(orient="records"):
        uid = str(row["SeriesInstanceUID"])
        output[uid] = {
            "plane": str(row.get("Anatomical_Plane") or "Unknown"),
            "fluid_sensitive": _number(row.get("Fluid_Sensitive")),
        }
    return output


def _load_labels(path: str | Path | None) -> dict[str, list[float | None]]:
    if not path:
        return {}
    frame = pd.read_csv(path)
    if "StudyInstanceUID" not in frame.columns:
        raise KeyError("label CSV requires StudyInstanceUID")
    labels: dict[str, list[float | None]] = {}
    for row in frame.to_dict(orient="records"):
        values = []
        for column in LABEL_COLUMNS:
            value = _number(row.get(column))
            values.append(value)
        labels[str(row["StudyInstanceUID"])] = values
    return labels


def _uniform_indices(length: int, count: int) -> list[int]:
    if length <= 0:
        return []
    if count <= 0 or length <= count:
        return list(range(length))
    return sorted(set(int(round(value)) for value in np.linspace(0, length - 1, count)))


def _series_windows(
    rows: list[tuple[Any, ...]],
    max_windows: int,
    neighbor_offsets: list[int],
    series_metadata: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    centers = _uniform_indices(len(rows), max_windows)
    series_uid = str(rows[0][1])
    derived_plane = str(rows[len(rows) // 2][5] or "Unknown")
    metadata = series_metadata or {}
    plane = str(metadata.get("plane") or derived_plane)
    fluid = metadata.get("fluid_sensitive")
    windows = []
    for center in centers:
        paths = [
            str(rows[min(max(center + offset, 0), len(rows) - 1)][2])
            for offset in neighbor_offsets
        ]
        windows.append(
            {
                "paths": paths,
                "series_id": series_uid,
                "plane": plane,
                "fluid_sensitive": fluid,
                "relative_position": center / max(1, len(rows) - 1),
                "center_index": center,
                "series_length": len(rows),
            }
        )
    return windows


def _iter_ordered_series(connection: sqlite3.Connection) -> Iterator[tuple[str, str, list[tuple[Any, ...]]]]:
    cursor = connection.execute(
        """
        SELECT study_uid, series_uid, relative_path, position_scalar,
               instance_number, derived_plane, patient_hash
        FROM slices
        ORDER BY study_uid, series_uid,
                 CASE WHEN position_scalar IS NULL THEN 1 ELSE 0 END,
                 position_scalar,
                 instance_number,
                 relative_path
        """
    )
    current_key: tuple[str, str] | None = None
    rows: list[tuple[Any, ...]] = []
    for row in cursor:
        key = (str(row[0]), str(row[1]))
        if current_key is not None and key != current_key:
            yield current_key[0], current_key[1], rows
            rows = []
        current_key = key
        rows.append(row)
    if current_key is not None:
        yield current_key[0], current_key[1], rows


def build_study_manifest(
    audit_root: str | Path,
    output_path: str | Path,
    labels_csv: str | Path | None = None,
    series_csv: str | Path | None = None,
    max_windows_per_series: int = 25,
    neighbor_offsets: list[int] | None = None,
    rebuild_db: bool = False,
    keep_db: bool = False,
) -> dict[str, int]:
    audit = Path(audit_root)
    output = Path(output_path)
    database_path = output.with_suffix(".sqlite")
    connection = _create_slice_db(audit, database_path, rebuild=rebuild_db)
    labels = _load_labels(labels_csv)
    metadata = _load_series_csv(series_csv)
    neighbor_offsets = neighbor_offsets or [-1, 0, 1]

    study_count = 0
    series_count = 0
    window_count = 0
    current_study: str | None = None
    current_patient: str | None = None
    current_windows: list[dict[str, Any]] = []

    with atomic_text_writer(output) as handle:
        for study_uid, series_uid, rows in _iter_ordered_series(connection):
            if current_study is not None and study_uid != current_study:
                handle.write(
                    json_dumps(
                        {
                            "study_id": current_study,
                            "patient_hash": current_patient,
                            "labels": labels.get(current_study),
                            "windows": current_windows,
                        }
                    )
                    + "\n"
                )
                study_count += 1
                current_windows = []
            current_study = study_uid
            current_patient = current_patient if current_windows else rows[0][6]
            windows = _series_windows(
                rows,
                max_windows=max_windows_per_series,
                neighbor_offsets=neighbor_offsets,
                series_metadata=metadata.get(series_uid),
            )
            current_windows.extend(windows)
            series_count += 1
            window_count += len(windows)
        if current_study is not None:
            handle.write(
                json_dumps(
                    {
                        "study_id": current_study,
                        "patient_hash": current_patient,
                        "labels": labels.get(current_study),
                        "windows": current_windows,
                    }
                )
                + "\n"
            )
            study_count += 1
    connection.close()
    if not keep_db and database_path.exists():
        database_path.unlink()
    return {"studies": study_count, "series": series_count, "windows": window_count}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build study-level 2.5D window manifest")
    parser.add_argument("--audit-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--labels-csv")
    parser.add_argument("--series-csv")
    parser.add_argument("--max-windows-per-series", type=int, default=25)
    parser.add_argument("--neighbor-offsets", default="-1,0,1")
    parser.add_argument("--rebuild-db", action="store_true")
    parser.add_argument("--keep-db", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    offsets = [int(value) for value in args.neighbor_offsets.split(",")]
    summary = build_study_manifest(
        args.audit_root,
        args.output,
        labels_csv=args.labels_csv,
        series_csv=args.series_csv,
        max_windows_per_series=args.max_windows_per_series,
        neighbor_offsets=offsets,
        rebuild_db=args.rebuild_db,
        keep_db=args.keep_db,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

