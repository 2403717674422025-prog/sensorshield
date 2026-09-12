"""
ml/reliability/metrics.py
==========================
Statistical metric extractors for Sensor Reliability & Health Scoring (Phase 6).

All sub-scores are scaled to [0.0, 100.0]:
  - 100.0: Perfectly healthy / optimal reliability
  - 0.0:   Severely degraded / failed

Sub-scores computed:
  1. Drift Score:       Detects persistent monotonic trend deviations.
  2. Noise Score:       Detects excessive high-frequency signal variance / jitter.
  3. Consistency Score: Detects physical range violations and cross-sensor correlation breakage.
  4. Missing Data Score: Quantifies sample availability, dropout rate, and NaN bursts.
  5. Stuck Check:       Identifies zero/negligible variance frozen sensor states.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Drift Score (0-100)
# ---------------------------------------------------------------------------
def compute_drift_score(
    signal: np.ndarray,
    baseline_std: float = 1.0,
    max_drift_threshold: float = 3.0,
) -> Tuple[float, float]:
    """
    Compute drift score based on linear slope and cumulative displacement.

    Parameters
    ----------
    signal : np.ndarray
        1D sensor window array (length W).
    baseline_std : float
        Nominal standard deviation of the sensor during healthy operation.
    max_drift_threshold : float
        Number of baseline stds of displacement considered 0/100 health.

    Returns
    -------
    (drift_score, estimated_drift_rate)
    """
    valid = signal[~np.isnan(signal)]
    if len(valid) < 2:
        return 0.0, 0.0

    n = len(valid)
    t = np.arange(n)

    # Linear regression slope: delta y / delta t
    t_mean = np.mean(t)
    y_mean = np.mean(valid)
    numerator = np.sum((t - t_mean) * (valid - y_mean))
    denominator = np.sum((t - t_mean) ** 2)

    slope = float(numerator / denominator) if denominator > 1e-12 else 0.0
    total_displacement = abs(slope * n)

    # Normalized drift magnitude relative to healthy baseline variance
    safe_std = max(baseline_std, 1e-6)
    drift_ratio = total_displacement / (max_drift_threshold * safe_std)

    # Sub-linear decay for minor noise fluctuations, steep drop for genuine trend
    if drift_ratio <= 0.25:
        score = 100.0 - (drift_ratio * 40.0)
    else:
        score = float(np.clip(90.0 * np.exp(-1.8 * (drift_ratio - 0.25)), 0.0, 100.0))
    return float(np.clip(score, 0.0, 100.0)), slope


# ---------------------------------------------------------------------------
# 2. Noise Score (0-100)
# ---------------------------------------------------------------------------
def compute_noise_score(
    signal: np.ndarray,
    baseline_diff_std: float = 1.0,
    max_noise_ratio: float = 4.0,
) -> float:
    """
    Compute noise score based on high-frequency first differences:
        delta_x_t = x_t - x_{t-1}

    Parameters
    ----------
    signal : np.ndarray
        1D sensor window array.
    baseline_diff_std : float
        Nominal std of consecutive differences under healthy conditions.
    max_noise_ratio : float
        Noise ratio that degrades score to 0.

    Returns
    -------
    noise_score : float in [0.0, 100.0]
    """
    valid = signal[~np.isnan(signal)]
    if len(valid) < 3:
        return 0.0

    diffs = np.diff(valid)
    current_diff_std = float(np.std(diffs))

    safe_base = max(baseline_diff_std, 1e-6)
    noise_ratio = current_diff_std / safe_base

    if noise_ratio <= 1.0:
        return 100.0

    penalty = (noise_ratio - 1.0) / max(max_noise_ratio - 1.0, 1e-6)
    score = float(np.clip(100.0 * np.exp(-1.8 * penalty), 0.0, 100.0))
    return score


# ---------------------------------------------------------------------------
# 3. Consistency Score (0-100)
# ---------------------------------------------------------------------------
def compute_consistency_score(
    sensor_val: float,
    valid_range: Optional[Tuple[float, float]] = None,
    correlation_error: Optional[float] = None,
) -> float:
    """
    Compute consistency score considering operating bounds and peer correlation.

    Parameters
    ----------
    sensor_val : float
        Current or window-average sensor value.
    valid_range : Optional[Tuple[float, float]]
        (min_valid, max_valid) physical range.
    correlation_error : Optional[float]
        Mean absolute discrepancy from expected cross-sensor correlation (0 to 1).

    Returns
    -------
    consistency_score : float in [0.0, 100.0]
    """
    if np.isnan(sensor_val):
        return 0.0

    range_score = 100.0
    if valid_range is not None:
        min_v, max_v = valid_range
        span = max_v - min_v
        if span > 1e-9:
            if sensor_val < min_v:
                overshoot = (min_v - sensor_val) / span
                range_score = float(np.clip(100.0 - (overshoot * 200.0), 0.0, 100.0))
            elif sensor_val > max_v:
                overshoot = (sensor_val - max_v) / span
                range_score = float(np.clip(100.0 - (overshoot * 200.0), 0.0, 100.0))

    corr_score = 100.0
    if correlation_error is not None:
        corr_score = float(np.clip(100.0 * (1.0 - correlation_error), 0.0, 100.0))

    return 0.6 * range_score + 0.4 * corr_score


# ---------------------------------------------------------------------------
# 4. Missing Data Score (0-100)
# ---------------------------------------------------------------------------
def compute_missing_data_score(signal: np.ndarray) -> Tuple[float, float, int]:
    """
    Compute missing data score and identify burstiness.

    Parameters
    ----------
    signal : np.ndarray
        1D sensor window array.

    Returns
    -------
    (missing_score, missing_rate, max_burst_len)
    """
    n = len(signal)
    if n == 0:
        return 0.0, 1.0, 0

    is_nan = np.isnan(signal)
    n_missing = int(np.sum(is_nan))
    missing_rate = float(n_missing / n)

    # Max contiguous NaN burst length
    max_burst = 0
    curr_burst = 0
    for val in is_nan:
        if val:
            curr_burst += 1
            if curr_burst > max_burst:
                max_burst = curr_burst
        else:
            curr_burst = 0

    # Score degrades with missing rate + burst length penalty
    base_score = 100.0 * (1.0 - missing_rate)
    burst_penalty = (max_burst / n) * 30.0 if n > 0 else 0.0
    score = float(np.clip(base_score - burst_penalty, 0.0, 100.0))

    return score, missing_rate, max_burst


# ---------------------------------------------------------------------------
# 5. Stuck Sensor Check
# ---------------------------------------------------------------------------
def is_stuck_signal(
    signal: np.ndarray,
    baseline_std: float = 1.0,
    min_variance_ratio: float = 0.01,
) -> Tuple[bool, float]:
    """
    Detect if sensor signal is frozen/stuck (near zero variance over full window).

    Parameters
    ----------
    signal : np.ndarray
        1D sensor window array.
    baseline_std : float
        Nominal sensor standard deviation.
    min_variance_ratio : float
        Fraction of baseline std below which signal is considered stuck.

    Returns
    -------
    (is_stuck, window_std)
    """
    valid = signal[~np.isnan(signal)]
    if len(valid) < 5:
        return False, 0.0

    w_std = float(np.std(valid))
    w_range = float(np.ptp(valid))

    safe_std = max(baseline_std, 1e-6)
    variance_ratio = w_std / safe_std

    is_stuck = (variance_ratio < min_variance_ratio) or (w_range < 1e-7)
    return is_stuck, w_std
