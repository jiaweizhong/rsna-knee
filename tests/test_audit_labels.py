import csv
from pathlib import Path

from rsna_knee.audit.summarize import _write_label_audit
from rsna_knee.constants import LABEL_COLUMNS


def _write_train_csv(path: Path) -> None:
    # G1: ACL+MCL positive: G2: ACL only: G3: none positive. N1: ungraded (all labels blank).
    rows = {
        "G1": {"ACL": 1, "MCL": 1},
        "G2": {"ACL": 1, "MCL": 0},
        "G3": {"ACL": 0, "MCL": 0},
        "N1": {},
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["StudyInstanceUID", "Report", *LABEL_COLUMNS])
        for uid, overrides in rows.items():
            values = [overrides.get(column, 0 if overrides else "") for column in LABEL_COLUMNS]
            writer.writerow([uid, "report text", *values])


def test_label_audit_bootstrap_and_cooccurrence(tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    _write_train_csv(train_csv)
    tables_dir = tmp_path / "tables"
    tables_dir.mkdir()

    summary = _write_label_audit(train_csv, tables_dir)

    assert summary["gold_study_count"] == 3
    assert summary["positive_label_count_distribution"] == {0: 1, 1: 1, 2: 1}

    by_label = {row["label"]: row for row in summary["labels"]}
    acl = by_label["ACL"]
    assert acl["positive"] == 2
    assert acl["negative"] == 1
    assert acl["missing"] == 1
    assert acl["prevalence_observed"] == 2 / 3
    assert 0.0 <= acl["prevalence_ci_low"] <= acl["prevalence_observed"] <= acl["prevalence_ci_high"] <= 1.0

    cooccurrence = (tables_dir / "label_cooccurrence.csv").read_text(encoding="utf-8")
    jaccard = (tables_dir / "label_jaccard.csv").read_text(encoding="utf-8")
    assert (tables_dir / "label_correlation.csv").exists()
    assert (tables_dir / "label_inventory.parquet").exists()

    import pandas as pd

    cooccurrence_frame = pd.read_csv(tables_dir / "label_cooccurrence.csv", index_col=0)
    jaccard_frame = pd.read_csv(tables_dir / "label_jaccard.csv", index_col=0)
    assert cooccurrence_frame.loc["ACL", "MCL"] == 1  # only G1 has both positive
    assert cooccurrence_frame.loc["ACL", "ACL"] == 2  # diagonal equals positive count
    assert jaccard_frame.loc["ACL", "ACL"] == 1.0
    assert jaccard_frame.loc["ACL", "MCL"] == 1 / 2  # intersection {G1} / union {G1, G2}
