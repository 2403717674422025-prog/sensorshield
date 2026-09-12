"""
ml/prediction/__init__.py
=========================
Phase 8 — Downstream Predictive Model & Trust Evaluation.

Public surface:
    DownstreamPredictor         — prediction and comparative evaluation engine
    PredictionOutput            — prediction dataclass with uncertainty
    XGBoostFailureClassifier    — gradient boosted failure classifier
    LSTMFailureClassifier       — sequence LSTM classifier with MC-Dropout
"""

from .engine import DownstreamPredictor, PredictionOutput
from .models import (
    XGBoostFailureClassifier,
    LSTMFailureClassifier,
    extract_window_features,
    predictive_entropy,
)

__all__ = [
    "DownstreamPredictor",
    "PredictionOutput",
    "XGBoostFailureClassifier",
    "LSTMFailureClassifier",
    "extract_window_features",
    "predictive_entropy",
]
