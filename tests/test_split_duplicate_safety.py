import csv
import json
from pathlib import Path

from rsna_knee.data.split import (
    assign_grouped_multilabel_folds,
    split_manifest,
    verify_fold_disjointness,
)


def _record(study_id: str, patient_hash: str, positive_label: int) -> dict:
    labels = [0.0] * 12
    labels[positive_label] = 1.0
    return {
        "study_id": study_id,
        "patient_hash": patient_hash,
        "labels": labels,
        "windows": [{"paths": ["x", "x", "x"]}],
    }


def _unlabeled_record(study_id: str, patient_hash: str) -> dict:
    return {
        "study_id": study_id,
        "patient_hash": patient_hash,
        "labels": [None] * 12,
        "windows": [{"paths": ["x", "x", "x"]}],
    }


def test_sparse_gold_labels_do_not_strand_a_fold_empty() -> None:
    # Reproduces the real failure at roughly real scale: 58 gold-labeled studies
    # among ~2,700 total (the actual RSNA train split). A single label/size cost
    # trapped whichever fold fell behind on labels early so it never received any
    # further studies at all, labeled or not.
    records = [_record(f"gold-{i}", f"gold-patient-{i}", i % 12) for i in range(58)]
    records += [_unlabeled_record(f"plain-{i}", f"plain-patient-{i}") for i in range(2656)]
    assignments = assign_grouped_multilabel_folds(records, folds=5, seed=7)
    counts = [0] * 5
    gold_counts = [0] * 5
    for study_id, fold in assignments.items():
        counts[fold] += 1
        if study_id.startswith("gold-"):
            gold_counts[fold] += 1
    assert sum(counts) == len(records)
    assert min(counts) > 0, f"a fold ended up empty: {counts}"
    assert max(counts) - min(counts) <= 10, f"folds are unevenly sized: {counts}"
    # This is the part that actually broke: overall size balanced out fine while
    # every gold study piled into 3 of the 5 folds, leaving 2 folds with zero gold
    # signal to validate against.
    assert min(gold_counts) > 0, f"a fold got zero gold-labeled studies: {gold_counts}"
    assert max(gold_counts) - min(gold_counts) <= 6, f"gold studies are unevenly spread: {gold_counts}"


def test_duplicate_edge_forces_same_fold_across_different_patients() -> None:
    records = [_record(f"study-{i}", f"patient-{i}", i % 3) for i in range(10)]
    # study-0 and study-9 are flagged as an exact-duplicate pair despite having
    # unrelated patient_hash values; they must land in the same fold anyway.
    duplicate_edges = [("study-0", "study-9")]
    assignments = assign_grouped_multilabel_folds(records, folds=3, seed=1, duplicate_edges=duplicate_edges)
    assert assignments["study-0"] == assignments["study-9"]

    report = verify_fold_disjointness(records, assignments, duplicate_edges=duplicate_edges)
    assert report == {
        "patient_hash_disjoint": True,
        "patient_hash_conflicts": [],
        "duplicate_group_disjoint": True,
        "duplicate_group_conflicts": [],
        "study_uid_unique": True,
    }


def test_verify_fold_disjointness_flags_a_real_leak() -> None:
    records = [_record("study-a", "patient-1", 0), _record("study-b", "patient-1", 1)]
    # Deliberately split the same patient across two folds to confirm the checker
    # actually catches it rather than always reporting clean.
    leaking_assignments = {"study-a": 0, "study-b": 1}
    report = verify_fold_disjointness(records, leaking_assignments)
    assert report["patient_hash_disjoint"] is False
    assert "patient-1" in report["patient_hash_conflicts"]


def test_split_manifest_end_to_end_with_duplicate_edges(tmp_path: Path) -> None:
    manifest = tmp_path / "all.jsonl"
    records = [_record(f"study-{i}", f"patient-{i}", i % 3) for i in range(10)]
    manifest.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

    edges_path = tmp_path / "suspected_duplicate_studies.csv"
    with edges_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["study_a", "study_b"])
        writer.writerow(["study-0", "study-9"])

    output = tmp_path / "folds"
    summary = split_manifest(manifest, output, folds=3, seed=1, duplicate_edges_path=edges_path)

    assert summary["duplicate_edges_used"] == 1
    assert summary["disjointness"]["patient_hash_disjoint"] is True
    assert summary["disjointness"]["duplicate_group_disjoint"] is True

    assignments = {
        row["StudyInstanceUID"]: int(row["fold"])
        for row in csv.DictReader((output / "assignments.csv").open(encoding="utf-8"))
    }
    assert assignments["study-0"] == assignments["study-9"]
