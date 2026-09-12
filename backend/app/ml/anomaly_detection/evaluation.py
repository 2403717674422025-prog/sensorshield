"""
ml/anomaly_detection/evaluation.py
=====================================
Evaluation harness for Phase 4 anomaly detectors.

Metrics computed
----------------
Per detector × per fault type:
  - Precision, Recall, F1-score    (at fitted threshold)
  - AUROC                          (threshold-free, area under ROC curve)
  - Average Precision (PR-AUC)     (better for imbalanced data)
  - Detection Delay (seconds)      (median samples from fault start to first TP)

Global summary table formatted for the Phase 16 report.

All evaluation is done on fault-injected test windows where:
  y = 1 for windows that overlap with a fault region (from fault_labels)
  y = 0 for clean windows
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .detectors import BaseDetector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------

@dataclass
class PerFaultMetrics:
    """Metrics for one detector evaluated on one fault type."""
    fault_type:        str
    n_anomaly:         int
    n_normal:          int
    precision:         float
    recall:            float
    f1:                float
    auroc:             float
    avg_precision:     float
    detection_delay:   Optional[float]  # median windows from fault start to first TP


@dataclass
class DetectionResult:
    """Full evaluation result for one detector."""
    detector_name:    str
    overall_f1:       float
    overall_auroc:    float
    overall_avg_prec: float
    per_fault:        List[PerFaultMetrics] = field(default_factory=list)
    fit_time_s:       float = 0.0
    score_time_s:     float = 0.0

    def summary_dict(self) -> dict:
        return {
            "Detector":        self.detector_name,
            "F1":              f"{self.overall_f1:.3f}",
            "AUROC":           f"{self.overall_auroc:.3f}",
            "Avg Precision":   f"{self.overall_avg_prec:.3f}",
            "Fit time (s)":    f"{self.fit_time_s:.1f}",
            "Score time (s)":  f"{self.score_time_s:.2f}",
        }


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate_detector(
    detector:       BaseDetector,
    X_normal_train: np.ndarray,
    X_test:         np.ndarray,
    y_test:         np.ndarray,
    fault_types:    Optional[np.ndarray] = None,
    window_step_s:  int = 60,
) -> DetectionResult:
    """
    Fit detector on normal data, evaluate on fault-injected test set.

    Parameters
    ----------
    detector        : unfitted BaseDetector instance
    X_normal_train  : (N_normal, W, S) — clean windows for training
    X_test          : (N_test, W, S)   — mixed clean + corrupted test windows
    y_test          : (N_test,)        — 1 if window overlaps fault, 0 otherwise
    fault_types     : (N_test,) str    — fault type label per window (or None)
    window_step_s   : int              — time between consecutive windows (seconds)

    Returns
    -------
    DetectionResult
    """
    import time

    # ---- Fit ----
    t0 = time.perf_counter()
    detector.fit(X_normal_train)
    fit_time = time.perf_counter() - t0

    # ---- Score ----
    t0 = time.perf_counter()
    scores = detector.score(X_test)
    score_time = time.perf_counter() - t0

    preds = detector.predict(X_test)
    y     = y_test.astype(int)

    # ---- Overall metrics ----
    has_both = len(np.unique(y)) == 2
    overall_auroc    = roc_auc_score(y, scores)    if has_both else 0.5
    overall_avg_prec = average_precision_score(y, scores) if has_both else 0.0
    overall_f1       = f1_score(y, preds, zero_division=0)

    # ---- Per-fault-type metrics ----
    per_fault: List[PerFaultMetrics] = []

    if fault_types is not None:
        unique_faults = sorted(set(ft for ft in fault_types if ft and ft != "NORMAL"))
        for ft in unique_faults:
            mask = (fault_types == ft) | (y == 0)
            y_ft      = y[mask]
            scores_ft = scores[mask]
            preds_ft  = preds[mask]

            n_anom   = int(y_ft.sum())
            n_normal = int((y_ft == 0).sum())

            if n_anom == 0:
                continue

            has_both_ft = len(np.unique(y_ft)) == 2
            f1_ft       = f1_score(y_ft, preds_ft, zero_division=0)
            auroc_ft    = roc_auc_score(y_ft, scores_ft) if has_both_ft else 0.5
            ap_ft       = average_precision_score(y_ft, scores_ft) if has_both_ft else 0.0
            prec_ft     = precision_score(y_ft, preds_ft, zero_division=0)
            rec_ft      = recall_score(y_ft, preds_ft, zero_division=0)

            # Detection delay: position of first TP among anomaly windows
            anom_indices = np.where(fault_types == ft)[0]
            tp_indices   = anom_indices[preds[anom_indices] == 1]
            if len(tp_indices) > 0 and len(anom_indices) > 0:
                delay_windows = int(tp_indices[0] - anom_indices[0])
                delay_s       = float(delay_windows * window_step_s)
            else:
                delay_s = None

            per_fault.append(PerFaultMetrics(
                fault_type      = ft,
                n_anomaly       = n_anom,
                n_normal        = n_normal,
                precision       = prec_ft,
                recall          = rec_ft,
                f1              = f1_ft,
                auroc           = auroc_ft,
                avg_precision   = ap_ft,
                detection_delay = delay_s,
            ))

    return DetectionResult(
        detector_name    = detector.name,
        overall_f1       = overall_f1,
        overall_auroc    = overall_auroc,
        overall_avg_prec = overall_avg_prec,
        per_fault        = per_fault,
        fit_time_s       = fit_time,
        score_time_s     = score_time,
    )


# ---------------------------------------------------------------------------
# Table builder — formats all results for reporting
# ---------------------------------------------------------------------------

def build_evaluation_table(results: Sequence[DetectionResult]) -> pd.DataFrame:
    """
    Build a summary DataFrame suitable for display and saving to CSV.

    Two tables are returned concatenated:
      1. Overall metrics per detector
      2. Per-fault-type breakdown (if per_fault data present)

    Returns
    -------
    summary_df : pd.DataFrame — overall metrics
    per_fault_df : pd.DataFrame — per-fault breakdown
    """
    # Overall
    summary_rows = [r.summary_dict() for r in results]
    summary_df   = pd.DataFrame(summary_rows).set_index("Detector")

    # Per-fault
    rows = []
    for r in results:
        for pf in r.per_fault:
            rows.append({
                "Detector":          r.detector_name,
                "Fault Type":        pf.fault_type,
                "N Anomaly":         pf.n_anomaly,
                "Precision":         round(pf.precision, 3),
                "Recall":            round(pf.recall, 3),
                "F1":                round(pf.f1, 3),
                "AUROC":             round(pf.auroc, 3),
                "Avg Precision":     round(pf.avg_precision, 3),
                "Detection Delay(s)": pf.detection_delay,
            })
    per_fault_df = pd.DataFrame(rows) if rows else pd.DataFrame()

    return summary_df, per_fault_df
