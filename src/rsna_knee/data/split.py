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


def assign_grouped_multilabel_folds(
    records: list[dict[str, Any]], folds: int = 5, seed: int = 2026
) -> dict[str, int]:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        group = str(record.get("patient_hash") or record["study_id"])
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


def split_manifest(
    manifest_path: str | Path,
    output_dir: str | Path,
    folds: int = 5,
    seed: int = 2026,
) -> dict[str, Any]:
    records = _read_manifest(Path(manifest_path))
    assignments = assign_grouped_multilabel_folds(records, folds=folds, seed=seed)
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
        positives = np.sum([_label_vector(record) for record in validation], axis=0)
        diagnostics.append(
            {
                "fold": fold,
                "studies": len(validation),
                "positives": dict(zip(LABEL_COLUMNS, positives.astype(int).tolist())),
            }
        )
    summary = {"studies": len(records), "folds": folds, "diagnostics": diagnostics}
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
    args = parser.parse_args()
    print(json.dumps(split_manifest(args.manifest, args.output_dir, args.folds, args.seed), indent=2))


if __name__ == "__main__":
    main()

