from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from rsna_knee.constants import LABEL_COLUMNS


def multilabel_auc(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    per_label: dict[str, float | None] = {}
    valid_scores = []
    for index, label in enumerate(LABEL_COLUMNS):
        valid = np.isfinite(targets[:, index]) & np.isfinite(probabilities[:, index])
        label_targets = targets[valid, index]
        if valid.sum() == 0 or np.unique(label_targets).size < 2:
            per_label[label] = None
            continue
        score = float(roc_auc_score(label_targets, probabilities[valid, index]))
        per_label[label] = score
        valid_scores.append(score)
    return {
        "macro_auc": float(np.mean(valid_scores)) if valid_scores else None,
        "valid_labels": len(valid_scores),
        "per_label_auc": per_label,
    }

