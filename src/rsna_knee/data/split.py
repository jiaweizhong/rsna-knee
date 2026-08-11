from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from rsna_knee.audit.common import atomic_text_writer, json_dumps
from rsna_knee.constants import LABEL_COLUMNS


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _label_vector(record: dict[str, Any]) -> np.ndarray:
    raw = record.get("labels")
    if raw is None:
        return np.zeros(len(LABEL_COLUMNS), dtype=np.float64)
    return np.asarray([0.0 if value is None else float(value) for value in raw], dtype=np.float64)


def _union_find_groups(
    records: list[dict[str, Any]], duplicate_edges: list[tuple[str, str]] | None
) -> dict[str, str]:
    """Map each study_id to a canonical group id: studies sharing a patient_hash are
    one group, and any pair connected by a duplicate-study edge (Image-Audit-Plan
    10.4, exact-duplicate detection) gets merged into that same group too, even
    across different patient_hash values."""
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_a] = root_b

    study_ids: set[str] = set()
    for record in records:
        study_id = str(record["study_id"])
        study_ids.add(study_id)
        union(study_id, str(record.get("patient_hash") or study_id))

    for study_a, study_b in duplicate_edges or []:
        if study_a in study_ids and study_b in study_ids:
            union(study_a, study_b)

    return {study_id: find(study_id) for study_id in study_ids}


def verify_fold_disjointness(
    records: list[dict[str, Any]],
    assignments: dict[str, int],
    duplicate_edges: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Image-Audit-Plan 10.4: patient hash and duplicate-study sets must not cross
    fold boundaries. Returns a pass/fail report; does not raise, so callers decide
    how to react (this is meant to be checked before training starts, not silently
    trusted)."""
    group_of_study = _union_find_groups(records, duplicate_edges)
    patient_hash_by_study = {str(r["study_id"]): r.get("patient_hash") for r in records}

    fold_of_group: dict[str, int] = {}
    group_conflicts: set[str] = set()
    fold_of_patient_hash: dict[str, int] = {}
    patient_conflicts: set[str] = set()
    for study_id, fold in assignments.items():
        group = group_of_study.get(study_id, study_id)
        if group in fold_of_group and fold_of_group[group] != fold:
            group_conflicts.add(group)
        fold_of_group[group] = fold

        patient_hash = patient_hash_by_study.get(study_id)
        if not patient_hash:
            continue
        patient_hash = str(patient_hash)
        if patient_hash in fold_of_patient_hash and fold_of_patient_hash[patient_hash] != fold:
            patient_conflicts.add(patient_hash)
        fold_of_patient_hash[patient_hash] = fold

    study_ids = [str(r["study_id"]) for r in records]
    return {
        "patient_hash_disjoint": len(patient_conflicts) == 0,
        "patient_hash_conflicts": sorted(patient_conflicts),
        "duplicate_group_disjoint": len(group_conflicts) == 0,
        "duplicate_group_conflicts": sorted(group_conflicts),
        "study_uid_unique": len(study_ids) == len(set(study_ids)),
    }


def assign_grouped_multilabel_folds(
    records: list[dict[str, Any]],
    folds: int = 5,
    seed: int = 2026,
    duplicate_edges: list[tuple[str, str]] | None = None,
) -> dict[str, int]:
    group_of_study = _union_find_groups(records, duplicate_edges)
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        group = group_of_study[str(record["study_id"])]
        groups[group].append(record)
    total_labels = sum((_label_vector(record) for record in records), np.zeros(len(LABEL_COLUMNS)))
    target_labels = total_labels / folds
    target_size = len(records) / folds
    rarity = 1.0 / np.maximum(total_labels, 1.0)
    random_generator = random.Random(seed)
    ordered = []
    for group, items in groups.items():
        labels = np.maximum.reduce([_label_vector(item) for item in items])
        priority = float(np.sum(labels * rarity)) + 0.01 * len(items)
        ordered.append((group, items, labels, priority, random_generator.random()))
    ordered.sort(key=lambda item: (-item[3], item[4]))

    fold_labels = np.zeros((folds, len(LABEL_COLUMNS)), dtype=np.float64)
    fold_sizes = np.zeros(folds, dtype=np.float64)
    assignments: dict[str, int] = {}
    for _, items, labels, _, _ in ordered:
        costs = []
        for fold in range(folds):
            proposed_labels = fold_labels[fold] + labels
            label_cost = np.mean(((proposed_labels - target_labels) / np.maximum(target_labels, 1.0)) ** 2)
            size_cost = ((fold_sizes[fold] + len(items) - target_size) / max(target_size, 1.0)) ** 2
            costs.append(float(label_cost + 0.25 * size_cost))
        chosen = min(range(folds), key=lambda fold: (costs[fold], fold_sizes[fold], fold))
        fold_labels[chosen] += labels
        fold_sizes[chosen] += len(items)
        for record in items:
            assignments[str(record["study_id"])] = chosen
    return assignments


def _load_duplicate_edges(path: str | Path | None) -> list[tuple[str, str]]:
    if not path or not Path(path).exists():
        return []
    edges = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            edges.append((row["study_a"], row["study_b"]))
    return edges


def split_manifest(
    manifest_path: str | Path,
    output_dir: str | Path,
    folds: int = 5,
    seed: int = 2026,
    duplicate_edges_path: str | Path | None = None,
) -> dict[str, Any]:
    records = _read_manifest(Path(manifest_path))
    duplicate_edges = _load_duplicate_edges(duplicate_edges_path)
    assignments = assign_grouped_multilabel_folds(
        records, folds=folds, seed=seed, duplicate_edges=duplicate_edges
    )
    disjointness = verify_fold_disjointness(records, assignments, duplicate_edges=duplicate_edges)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with atomic_text_writer(output / "assignments.csv") as handle:
        writer = csv.DictWriter(handle, fieldnames=["StudyInstanceUID", "fold"])
        writer.writeheader()
        for study_id in sorted(assignments):
            writer.writerow({"StudyInstanceUID": study_id, "fold": assignments[study_id]})
    diagnostics = []
    for fold in range(folds):
        fold_dir = output / f"fold-{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        validation = [record for record in records if assignments[str(record["study_id"])] == fold]
        training = [record for record in records if assignments[str(record["study_id"])] != fold]
        for name, items in [("train.jsonl", training), ("valid.jsonl", validation)]:
            with atomic_text_writer(fold_dir / name) as handle:
                for record in items:
                    handle.write(json_dumps(record) + "\n")
        positives = (
            np.sum([_label_vector(record) for record in validation], axis=0)
            if validation
            else np.zeros(len(LABEL_COLUMNS))
        )
        diagnostics.append(
            {
                "fold": fold,
                "studies": len(validation),
                "positives": dict(zip(LABEL_COLUMNS, positives.astype(int).tolist())),
            }
        )
    summary = {
        "studies": len(records),
        "folds": folds,
        "duplicate_edges_used": len(duplicate_edges),
        "disjointness": disjointness,
        "diagnostics": diagnostics,
    }
    (output / "diagnostics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Patient-grouped multilabel manifest split")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--duplicate-edges",
        help="issues/suspected_duplicate_studies.csv from `rsna-knee-audit duplicates` (optional but recommended)",
    )
    args = parser.parse_args()
    summary = split_manifest(
        args.manifest, args.output_dir, args.folds, args.seed, duplicate_edges_path=args.duplicate_edges
    )
    print(json.dumps(summary, indent=2))
    if not (summary["disjointness"]["patient_hash_disjoint"] and summary["disjointness"]["duplicate_group_disjoint"]):
        raise SystemExit(
            "Fold leakage detected: patient_hash or duplicate-group boundaries were crossed. "
            "See diagnostics.json's disjointness section before using these folds."
        )


if __name__ == "__main__":
    main()

