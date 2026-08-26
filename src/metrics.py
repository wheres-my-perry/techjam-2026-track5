"""Metrics for AIGC detection. Labels: 1 = AI-generated (positive), 0 = real."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auroc(y_true, y_score) -> float:
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def balanced_accuracy(y_true, y_score, threshold: float) -> float:
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_score) >= threshold
    tpr = y_pred[y_true].mean() if y_true.any() else float("nan")
    tnr = (~y_pred[~y_true]).mean() if (~y_true).any() else float("nan")
    return float((tpr + tnr) / 2.0)


def fpr_at_tpr(y_true, y_score, target_tpr: float = 0.95) -> float:
    """FPR at the lowest threshold achieving >= target_tpr. NaN if unattainable."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ok = tpr >= target_tpr
    if not ok.any():
        return float("nan")
    return float(fpr[ok].min())


def pick_threshold(y_true, y_score) -> float:
    """Threshold maximizing balanced accuracy (Youden's J) on *clean validation* data.

    Picked ONCE on clean val, then frozen for every transformed condition.
    """
    y_true = np.asarray(y_true)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    return float(thresholds[int(np.argmax(j))])


def condition_report(y_true, y_score, threshold: float) -> dict:
    return {
        "auroc": auroc(y_true, y_score),
        "balanced_acc@frozen_thr": balanced_accuracy(y_true, y_score, threshold),
        "fpr@tpr95": fpr_at_tpr(y_true, y_score, 0.95),
        "n": int(len(np.asarray(y_true))),
    }
