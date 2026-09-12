"""
ml/prediction/engine.py
========================
Downstream Predictive Engine (Phase 8).

Wraps downstream failure predictors (XGBoost + LSTM) and evaluates
predictive reliability across:
  1. Clean sensor data (ideal upper bound)
  2. Corrupted sensor data (silent prediction failure / trust degradation)
  3. Reconstructed sensor data (repaired via Phase 7 Signal Reconstructor)
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .models import (
    LSTMFailureClassifier,
    PyTorchLSTMClassifierNet,
    XGBoostFailureClassifier,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output Dataclass
# ---------------------------------------------------------------------------
@dataclass
class PredictionOutput:
    """Downstream machine failure inference output."""
    failure_probability: float
    model_uncertainty: float
    is_failure_predicted: bool
    model_name: str
    timestamp: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "failure_probability": round(self.failure_probability, 4),
            "model_uncertainty": round(self.model_uncertainty, 4),
            "is_failure_predicted": self.is_failure_predicted,
            "model_name": self.model_name,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


# ---------------------------------------------------------------------------
# Downstream Predictor Engine
# ---------------------------------------------------------------------------
class DownstreamPredictor:
    """
    Unified Downstream Failure Prediction & Trust Evaluation Engine.
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device
        self.xgboost_model = XGBoostFailureClassifier()
        self.lstm_model = LSTMFailureClassifier(device=device)
        self._is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        lstm_epochs: int = 15,
        verbose: bool = False,
    ) -> "DownstreamPredictor":
        """
        Fit both XGBoost and LSTM failure classifiers on clean training data.
        """
        logger.info("Fitting XGBoost failure classifier...")
        self.xgboost_model.fit(X_train, y_train)

        logger.info("Fitting LSTM failure classifier...")
        self.lstm_model.fit(X_train, y_train, epochs=lstm_epochs, verbose=verbose)

        self._is_fitted = True
        return self

    def predict_window(
        self,
        window: np.ndarray,
        use_model: str = "lstm",
        n_mc_samples: int = 10,
        threshold: float = 0.5,
        timestamp: Optional[datetime] = None,
    ) -> PredictionOutput:
        """
        Run failure prediction on a single sliding window (W, S) or batch (N, W, S).
        """
        if not self._is_fitted:
            raise RuntimeError("DownstreamPredictor is not fitted.")

        single = window.ndim == 2
        X = np.expand_dims(window, axis=0) if single else window

        if use_model.lower() == "xgboost":
            probs = self.xgboost_model.predict_proba(X)
            # Estimate tabular uncertainty via distance from decision boundary
            uncertainty = np.abs(probs - 0.5)
            uncertainty_std = 0.5 - uncertainty  # Higher near 0.5
            prob = float(probs[0]) if single else probs
            unc = float(uncertainty_std[0]) if single else uncertainty_std
        else:
            probs, uncertainties = self.lstm_model.predict_proba_with_uncertainty(
                X, n_mc_samples=n_mc_samples
            )
            prob = float(probs[0]) if single else probs
            unc = float(uncertainties[0]) if single else uncertainties

        is_fail = bool(prob >= threshold) if single else (probs >= threshold).astype(bool)

        return PredictionOutput(
            failure_probability=prob if single else float(np.mean(prob)),
            model_uncertainty=unc if single else float(np.mean(unc)),
            is_failure_predicted=is_fail if single else bool(np.any(is_fail)),
            model_name=use_model,
            timestamp=timestamp,
        )

    def evaluate(
        self,
        y_true: np.ndarray,
        X: np.ndarray,
        use_model: str = "lstm",
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Evaluate classification metrics on given data.
        """
        if use_model.lower() == "xgboost":
            probs = self.xgboost_model.predict_proba(X)
        else:
            probs = self.lstm_model.predict_proba(X)

        preds = (probs >= threshold).astype(int)
        y = y_true.astype(int)
        labels = np.unique(y)
        if not np.isin(labels, [0, 1]).all():
            raise ValueError(
                f"y_true must use binary labels 0=normal and 1=failure, got {labels.tolist()}"
            )

        has_both = len(np.unique(y)) == 2
        auroc = float(roc_auc_score(y, probs)) if has_both else float("nan")
        brier = float(brier_score_loss(y, probs))
        f1 = float(f1_score(y, preds, zero_division=0))
        prec = float(precision_score(y, preds, zero_division=0))
        rec = float(recall_score(y, preds, zero_division=0))
        acc = float(accuracy_score(y, preds))

        return {
            "F1": round(f1, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "AUROC": round(auroc, 4),
            "Brier Score": round(brier, 4),
            "Accuracy": round(acc, 4),
        }

    def compare_trust_impact(
        self,
        y_true: np.ndarray,
        X_clean: np.ndarray,
        X_corrupted: np.ndarray,
        X_reconstructed: np.ndarray,
        use_model: str = "lstm",
    ) -> Dict[str, Dict[str, float]]:
        """
        Demonstrate the trust problem and signal reconstruction recovery:
          1. Clean Evaluation (ground-truth baseline)
          2. Corrupted Evaluation (silent failure under untrusted sensors)
          3. Reconstructed Evaluation (restored with Signal Reconstructor)
        """
        clean_res = self.evaluate(y_true, X_clean, use_model=use_model)
        corrupted_res = self.evaluate(y_true, X_corrupted, use_model=use_model)
        reconstructed_res = self.evaluate(y_true, X_reconstructed, use_model=use_model)

        return {
            "1. Clean Ground-Truth": clean_res,
            "2. Corrupted (Silent Degradation)": corrupted_res,
            "3. Reconstructed (Trust Restored)": reconstructed_res,
        }

    def save(self, dir_path: str) -> None:
        """Save both models to disk."""
        os.makedirs(dir_path, exist_ok=True)
        xgb_path = os.path.join(dir_path, "xgboost_model.pkl")
        lstm_path = os.path.join(dir_path, "lstm_model.pt")

        with open(xgb_path, "wb") as f:
            pickle.dump({"model": self.xgboost_model.model, "scaler": self.xgboost_model.scaler}, f)

        if self.lstm_model.model is not None:
            torch.save(
                {
                    "state_dict": self.lstm_model.model.state_dict(),
                    "hidden_dim": self.lstm_model.hidden_dim,
                    "num_layers": self.lstm_model.num_layers,
                    "dropout": self.lstm_model.dropout,
                    "lr": self.lstm_model.lr,
                    "n_sensors": self.lstm_model.model.lstm.input_size,
                },
                lstm_path,
            )
        logger.info("Saved DownstreamPredictor models to %s", dir_path)

    @classmethod
    def load(cls, dir_path: str, device: Optional[str] = None) -> "DownstreamPredictor":
        """Load saved DownstreamPredictor."""
        xgb_path = os.path.join(dir_path, "xgboost_model.pkl")
        lstm_path = os.path.join(dir_path, "lstm_model.pt")

        predictor = cls(device=device)

        with open(xgb_path, "rb") as f:
            xgb_dict = pickle.load(f)
            predictor.xgboost_model.model = xgb_dict["model"]
            predictor.xgboost_model.scaler = xgb_dict["scaler"]
            predictor.xgboost_model._is_fitted = True

        lstm_ckpt = torch.load(lstm_path, map_location=device or "cpu")
        predictor.lstm_model.hidden_dim = lstm_ckpt["hidden_dim"]
        predictor.lstm_model.num_layers = lstm_ckpt["num_layers"]
        predictor.lstm_model.dropout = lstm_ckpt.get("dropout", 0.25)
        predictor.lstm_model.lr = lstm_ckpt.get("lr", predictor.lstm_model.lr)
        n_sensors = lstm_ckpt.get("n_sensors", 6)

        # Re-instantiate net
        predictor.lstm_model.model = PyTorchLSTMClassifierNet(
            n_sensors=n_sensors,
            hidden_dim=predictor.lstm_model.hidden_dim,
            num_layers=predictor.lstm_model.num_layers,
            dropout=predictor.lstm_model.dropout,
        ).to(predictor.lstm_model.device)
        predictor.lstm_model.model.load_state_dict(lstm_ckpt["state_dict"])
        predictor.lstm_model._is_fitted = True

        predictor._is_fitted = True
        return predictor
