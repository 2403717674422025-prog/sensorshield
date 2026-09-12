"""
ml/reliability/engine.py
=========================
Sensor Reliability & Health Scoring Engine (Phase 6).

Evaluates incoming sensor stream windows against statistical baselines
and ML reconstruction models to produce:
  - Composite Health Score (0–100)
  - Sub-scores: Drift, Noise, Consistency, Missing-data, Reconstruction
  - Sensor Health State (HEALTHY, DRIFTING, NOISY, STUCK, MISSING, INTERMITTENT, FAILED)
  - Diagnostic Confidence & Human-Readable Explainability Reason
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from app.models.sensor import SensorStatus
from .metrics import (
    compute_drift_score,
    compute_noise_score,
    compute_consistency_score,
    compute_missing_data_score,
    is_stuck_signal,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluation Output Dataclass
# ---------------------------------------------------------------------------
@dataclass
class SensorHealthEvaluation:
    """Complete evaluation report for a single sensor window."""
    sensor_id: str
    timestamp: datetime
    health_score: float                   # Composite 0–100 (100 = perfect)
    status: SensorStatus                  # Inferred state enum
    status_confidence: float              # 0.0–1.0
    reason: str                           # Explainable diagnostic reason

    # Sub-scores (all 0–100)
    drift_score: float
    noise_score: float
    consistency_score: float
    missing_data_score: float
    anomaly_score: float                  # ML anomaly sub-score
    reconstruction_error: float           # Raw MSE reconstruction loss

    # Diagnostic metadata
    metadata: Dict[str, Union[float, int, str, bool]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "sensor_id": self.sensor_id,
            "timestamp": self.timestamp.isoformat(),
            "health_score": round(self.health_score, 2),
            "status": self.status.value,
            "status_confidence": round(self.status_confidence, 2),
            "reason": self.reason,
            "drift_score": round(self.drift_score, 2),
            "noise_score": round(self.noise_score, 2),
            "consistency_score": round(self.consistency_score, 2),
            "missing_data_score": round(self.missing_data_score, 2),
            "anomaly_score": round(self.anomaly_score, 2),
            "reconstruction_error": round(self.reconstruction_error, 6),
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Sensor Baseline Configuration
# ---------------------------------------------------------------------------
@dataclass
class SensorBaseline:
    """Learned nominal baseline metrics for a single sensor channel."""
    sensor_id: str
    mean: float
    std: float
    diff_std: float                       # Nominal std of 1st differences
    min_valid: Optional[float] = None     # Physical valid lower bound
    max_valid: Optional[float] = None     # Physical valid upper bound
    expected_recon_mse: float = 0.01      # Baseline normal autoencoder MSE


# ---------------------------------------------------------------------------
# Sensor Reliability Engine
# ---------------------------------------------------------------------------
class SensorReliabilityEngine:
    """
    Stateful Reliability Engine for industrial sensor health monitoring.

    Parameters
    ----------
    weights : Optional[Dict[str, float]]
        Weights for composite score: drift, noise, consistency, missing, anomaly.
        Must sum to 1.0.
    """

    DEFAULT_WEIGHTS = {
        "drift": 0.25,
        "noise": 0.20,
        "consistency": 0.20,
        "missing": 0.20,
        "anomaly": 0.15,
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        baselines: Optional[Dict[str, SensorBaseline]] = None,
    ):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        total_w = sum(self.weights.values())
        if abs(total_w - 1.0) > 1e-4:
            # Normalize weights to sum to 1.0
            self.weights = {k: v / total_w for k, v in self.weights.items()}

        self.baselines: Dict[str, SensorBaseline] = baselines or {}

    def fit_baselines(
        self,
        df_normal: np.ndarray,
        sensor_ids: List[str],
        valid_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> "SensorReliabilityEngine":
        """
        Learn nominal baselines from clean normal training windows or DataFrame.

        Parameters
        ----------
        df_normal : np.ndarray
            Shape (N, n_sensors) or (N, W, n_sensors).
        sensor_ids : List[str]
            Ordered list of sensor identifiers.
        valid_ranges : Optional[Dict[str, Tuple[float, float]]]
            Optional physical sensor min/max operating ranges.
        """
        if df_normal.ndim == 3:
            # Flatten (N, W, S) -> (N*W, S)
            flat_normal = df_normal.reshape(-1, df_normal.shape[-1])
        else:
            flat_normal = df_normal

        self.baselines.clear()
        for idx, sid in enumerate(sensor_ids):
            col = flat_normal[:, idx]
            valid = col[~np.isnan(col)]
            mean_val = float(np.mean(valid)) if len(valid) > 0 else 0.0
            std_val = float(np.std(valid)) if len(valid) > 1 else 1.0
            diff_std_val = float(np.std(np.diff(valid))) if len(valid) > 2 else 1.0

            rng = valid_ranges.get(sid) if valid_ranges else None
            min_v = rng[0] if rng else None
            max_v = rng[1] if rng else None

            self.baselines[sid] = SensorBaseline(
                sensor_id=sid,
                mean=mean_val,
                std=max(std_val, 1e-4),
                diff_std=max(diff_std_val, 1e-4),
                min_valid=min_v,
                max_valid=max_v,
                expected_recon_mse=0.01,
            )

        logger.info("SensorReliabilityEngine fitted with baselines for %d sensors", len(self.baselines))
        return self

    def evaluate_sensor(
        self,
        sensor_id: str,
        signal_window: np.ndarray,
        timestamp: Optional[datetime] = None,
        reconstruction_error: float = 0.0,
    ) -> SensorHealthEvaluation:
        """
        Evaluate health of a single sensor window.

        Parameters
        ----------
        sensor_id : str
            Sensor identifier.
        signal_window : np.ndarray
            1D array of recent sensor readings of length W.
        timestamp : Optional[datetime]
            Timestamp of evaluation (defaults to now).
        reconstruction_error : float
            Per-sensor reconstruction MSE from Phase 5 LSTM Autoencoder.

        Returns
        -------
        SensorHealthEvaluation
        """
        ts = timestamp or datetime.now(timezone.utc).replace(tzinfo=None)
        valid_all = signal_window[~np.isnan(signal_window)]
        base = self.baselines.get(
            sensor_id,
            SensorBaseline(
                sensor_id=sensor_id,
                mean=float(np.mean(valid_all)) if len(valid_all) > 0 else 0.0,
                std=float(np.std(valid_all)) if len(valid_all) > 1 else 1.0,
                diff_std=float(np.std(np.diff(valid_all))) if len(valid_all) > 2 else 1.0,
            ),
        )

        # 1. Compute individual sub-scores
        drift_score, slope = compute_drift_score(signal_window, baseline_std=base.std)
        noise_score = compute_noise_score(signal_window, baseline_diff_std=base.diff_std)

        valid_vals = signal_window[~np.isnan(signal_window)]
        mean_val = float(np.mean(valid_vals)) if len(valid_vals) > 0 else np.nan
        vr = (base.min_valid, base.max_valid) if base.min_valid is not None else None
        consistency_score = compute_consistency_score(sensor_val=mean_val, valid_range=vr)

        missing_score, missing_rate, max_burst = compute_missing_data_score(signal_window)
        stuck, w_std = is_stuck_signal(signal_window, baseline_std=base.std)

        # 2. ML Anomaly sub-score from reconstruction error
        # Normalized mapping: recon_error <= base.expected_recon_mse -> 100
        recon_ratio = reconstruction_error / max(base.expected_recon_mse * 5.0, 1e-4)
        anomaly_score = float(np.clip(100.0 * np.exp(-1.5 * recon_ratio), 0.0, 100.0))

        # 3. Composite Health Score (Weighted Sum)
        composite = (
            self.weights["drift"] * drift_score
            + self.weights["noise"] * noise_score
            + self.weights["consistency"] * consistency_score
            + self.weights["missing"] * missing_score
            + self.weights["anomaly"] * anomaly_score
        )

        # If stuck or missing, heavily suppress composite health score
        if stuck and missing_rate < 0.15:
            composite = min(composite, 25.0)
        elif missing_rate >= 0.75:
            composite = min(composite, 15.0)

        health_score = float(np.clip(composite, 0.0, 100.0))

        # 4. Classify Sensor Status & Formulate Diagnostic Reason
        status, confidence, reason = self._classify_status(
            health_score=health_score,
            drift_score=drift_score,
            noise_score=noise_score,
            consistency_score=consistency_score,
            missing_score=missing_score,
            anomaly_score=anomaly_score,
            slope=slope,
            missing_rate=missing_rate,
            max_burst=max_burst,
            stuck=stuck,
            base_std=base.std,
        )

        return SensorHealthEvaluation(
            sensor_id=sensor_id,
            timestamp=ts,
            health_score=health_score,
            status=status,
            status_confidence=confidence,
            reason=reason,
            drift_score=drift_score,
            noise_score=noise_score,
            consistency_score=consistency_score,
            missing_data_score=missing_score,
            anomaly_score=anomaly_score,
            reconstruction_error=reconstruction_error,
            metadata={
                "drift_slope": slope,
                "missing_rate": missing_rate,
                "max_burst_nan": max_burst,
                "is_stuck": stuck,
                "window_std": w_std,
            },
        )

    def _classify_status(
        self,
        health_score: float,
        drift_score: float,
        noise_score: float,
        consistency_score: float,
        missing_score: float,
        anomaly_score: float,
        slope: float,
        missing_rate: float,
        max_burst: int,
        stuck: bool,
        base_std: float,
    ) -> Tuple[SensorStatus, float, str]:
        """Classify health state and generate human-readable reason."""
        # Check Total Dropout first
        if missing_rate >= 0.75:
            conf = min(1.0, 0.5 + (missing_rate * 0.5))
            reason = f"Severe signal loss: {missing_rate:.1%} of readings missing in window"
            return SensorStatus.MISSING, conf, reason

        # Check Intermittent Dropouts
        if missing_rate >= 0.15 or max_burst >= 4:
            conf = 0.85
            reason = f"Intermittent dropout detected ({missing_rate:.1%} NaN, burst length {max_burst})"
            return SensorStatus.INTERMITTENT, conf, reason

        # Check Stuck / Frozen sensor (when not due to dropouts)
        if stuck:
            conf = 0.95
            reason = "Sensor output is frozen/constant with zero variance (stuck reading)"
            return SensorStatus.STUCK, conf, reason

        # Check Severe Failure
        if health_score < 30.0 or consistency_score < 30.0:
            conf = 0.90
            reason = f"Sensor operating outside valid bounds (health: {health_score:.1f}/100)"
            return SensorStatus.FAILED, conf, reason

        # Check Drift
        if drift_score < 60.0 and drift_score <= min(noise_score, consistency_score):
            conf = min(1.0, 0.6 + (1.0 - drift_score / 100.0) * 0.4)
            direction = "positive" if slope > 0 else "negative"
            reason = f"Monotonic signal drift detected ({direction} slope {slope:+.4f}/min, drift score: {drift_score:.1f}/100)"
            return SensorStatus.DRIFTING, conf, reason

        # Check Noise
        if noise_score < 60.0 and noise_score <= min(drift_score, consistency_score):
            conf = min(1.0, 0.6 + (1.0 - noise_score / 100.0) * 0.4)
            reason = f"Excessive high-frequency noise / jitter (noise score: {noise_score:.1f}/100)"
            return SensorStatus.NOISY, conf, reason

        # Healthy
        if health_score >= 75.0:
            conf = min(1.0, health_score / 100.0)
            reason = f"Sensor operating within nominal parameters (health: {health_score:.1f}/100)"
            return SensorStatus.HEALTHY, conf, reason

        # Degrading / Unknown transition
        conf = 0.70
        reason = f"Moderate degradation across metrics (health: {health_score:.1f}/100)"
        return SensorStatus.HEALTHY, conf, reason
