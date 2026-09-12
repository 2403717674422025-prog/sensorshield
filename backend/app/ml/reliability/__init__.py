"""
ml/reliability/__init__.py
===========================
Phase 6 — Sensor Reliability & Health Scoring Engine.

Public surface:
    SensorReliabilityEngine   — primary evaluation engine
    SensorHealthEvaluation    — health report dataclass
    SensorBaseline            — per-sensor learned nominal baseline
    compute_drift_score       — drift subscore calculator
    compute_noise_score       — noise subscore calculator
    compute_consistency_score — consistency subscore calculator
    compute_missing_data_score — missing data subscore calculator
    is_stuck_signal           — frozen sensor detector
"""

from .engine import (
    SensorReliabilityEngine,
    SensorHealthEvaluation,
    SensorBaseline,
)
from .metrics import (
    compute_drift_score,
    compute_noise_score,
    compute_consistency_score,
    compute_missing_data_score,
    is_stuck_signal,
)

__all__ = [
    "SensorReliabilityEngine",
    "SensorHealthEvaluation",
    "SensorBaseline",
    "compute_drift_score",
    "compute_noise_score",
    "compute_consistency_score",
    "compute_missing_data_score",
    "is_stuck_signal",
]
