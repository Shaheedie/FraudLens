from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def metrics_at_threshold(y_true, scores, threshold: float) -> dict:
    pred = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def choose_threshold(
    y_true, scores, fp_cost: float = 1.0, fn_cost: float = 20.0
) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in np.linspace(0.01, 0.99, 197):
        m = metrics_at_threshold(y_true, scores, float(threshold))
        m["expected_cost"] = m["fp"] * fp_cost + m["fn"] * fn_cost
        rows.append(m)
    frame = pd.DataFrame(rows)
    best = frame.sort_values(["expected_cost", "f1"], ascending=[True, False]).iloc[0]
    return float(best["threshold"]), frame
