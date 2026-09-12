"""
ml/anomaly_detection/__init__.py
==================================
Phase 4 — Baseline Anomaly Detection.

Three baselines evaluated on clean vs. fault-injected sensor windows:
  1. RuleBasedDetector   — per-sensor statistical thresholds
  2. IsolationForestDetector — sklearn IsolationForest
  3. OneClassSVMDetector — sklearn OneClassSVM

Public surface:
    BaseDetector        — abstract base class
    RuleBasedDetector
    IsolationForestDetector
    OneClassSVMDetector
    evaluate_detector   — computes F1, AUROC, detection delay, per-fault metrics
    DetectionResult     — dataclass for evaluation output
"""

from .detectors import (
    BaseDetector,
    RuleBasedDetector,
    IsolationForestDetector,
    OneClassSVMDetector,
)
from .evaluation import DetectionResult, evaluate_detector, build_evaluation_table
from .lstm_autoencoder import LSTMAutoencoderDetector, PyTorchLSTMAutoencoder

__all__ = [
    "BaseDetector",
    "RuleBasedDetector",
    "IsolationForestDetector",
    "OneClassSVMDetector",
    "LSTMAutoencoderDetector",
    "PyTorchLSTMAutoencoder",
    "DetectionResult",
    "evaluate_detector",
    "build_evaluation_table",
]
