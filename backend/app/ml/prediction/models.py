"""
ml/prediction/models.py
========================
Downstream Machine Failure Predictive Models (Phase 8).

Includes:
  1. XGBoostFailureClassifier:
     Gradient Boosted Decision Trees on window statistical features.
     Predicts failure probability with scale_pos_weight for class imbalance.

  2. PyTorchLSTMClassifier:
     Recurrent sequence classifier with Monte Carlo Dropout for epistemic uncertainty.
     Trained on temporal windows (batch_size, seq_len, n_sensors).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def predictive_entropy(probabilities: np.ndarray, epsilon: float = 1e-7) -> np.ndarray:
    """Binary predictive entropy for a vector of failure probabilities.

    This is an uncertainty proxy for the XGBoost point-probability model.  It
    is distinct from confidence: it measures the entropy of the Bernoulli
    predictive distribution and is maximal at p=0.5.
    """
    p = np.clip(np.asarray(probabilities, dtype=float), epsilon, 1.0 - epsilon)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


# ---------------------------------------------------------------------------
# Feature Extraction Helper for Tabular / Tree Models
# ---------------------------------------------------------------------------
def extract_window_features(X_windows: np.ndarray) -> np.ndarray:
    """
    Extract comprehensive temporal features from 3D sensor windows.

    X_windows: (N, seq_len, n_sensors)
    returns: (N, n_sensors * 7)
    """
    X_clean = np.nan_to_num(X_windows, nan=0.0, posinf=0.0, neginf=0.0)
    means = np.mean(X_clean, axis=1)
    stds = np.std(X_clean, axis=1)
    mins = np.min(X_clean, axis=1)
    maxs = np.max(X_clean, axis=1)
    ptps = maxs - mins
    diffs = np.std(np.diff(X_clean, axis=1), axis=1)
    trends = X_clean[:, -1, :] - X_clean[:, 0, :]

    features = np.concatenate([means, stds, mins, maxs, ptps, diffs, trends], axis=1)
    return features


# ---------------------------------------------------------------------------
# 1. XGBoost Failure Classifier
# ---------------------------------------------------------------------------
class XGBoostFailureClassifier:
    """
    Downstream failure predictor based on XGBoost.
    """

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.model: Optional[xgb.XGBClassifier] = None
        self._is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "XGBoostFailureClassifier":
        """
        X_train: (N, seq_len, n_sensors)
        y_train: (N,) binary labels (0 = normal, 1 = failure)
        """
        feats = extract_window_features(X_train)
        feats_scaled = self.scaler.fit_transform(feats)

        pos_count = int(np.sum(y_train == 1))
        neg_count = int(np.sum(y_train == 0))
        scale_pos_weight = max(1.0, float(neg_count / max(pos_count, 1)))

        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            eval_metric="logloss",
            n_jobs=-1,
        )
        self.model.fit(feats_scaled, y_train)
        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns failure probabilities of shape (N,).
        """
        if not self._is_fitted or self.model is None:
            raise RuntimeError("XGBoostFailureClassifier is not fitted.")

        feats = extract_window_features(X)
        feats_scaled = self.scaler.transform(feats)
        probs = self.model.predict_proba(feats_scaled)[:, 1]
        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)


# ---------------------------------------------------------------------------
# 2. PyTorch Sequence LSTM Classifier with MC-Dropout
# ---------------------------------------------------------------------------
class PyTorchLSTMClassifierNet(nn.Module):
    def __init__(
        self,
        n_sensors: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_sensors,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, (hn, _) = self.lstm(x)
        # Use last timestep representation
        last_step = out[:, -1, :]
        dropped = self.dropout(last_step)
        logits = self.fc(dropped).squeeze(-1)
        return logits


class LSTMFailureClassifier:
    """
    Recurrent sequence classifier with Monte Carlo Dropout for uncertainty estimation.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.25,
        lr: float = 0.002,
        device: Optional[str] = None,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: Optional[PyTorchLSTMClassifierNet] = None
        self._is_fitted = False
        self.training_history: List[Dict[str, float]] = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 20,
        batch_size: int = 128,
        verbose: bool = False,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        pos_weight_override: Optional[float] = None,
    ) -> "LSTMFailureClassifier":
        N, seq_len, n_sensors = X_train.shape
        self.model = PyTorchLSTMClassifierNet(
            n_sensors=n_sensors,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)
        self.training_history = []

        X_clean = np.nan_to_num(X_train, nan=0.0).astype(np.float32)
        y_clean = y_train.astype(np.float32)

        dataset = TensorDataset(torch.tensor(X_clean), torch.tensor(y_clean))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        pos_count = float(np.sum(y_train == 1))
        neg_count = float(np.sum(y_train == 0))
        class_weight = (
            max(1.0, neg_count / max(pos_count, 1.0))
            if pos_weight_override is None else float(pos_weight_override)
        )
        if class_weight <= 0:
            raise ValueError("pos_weight_override must be positive.")
        pos_weight = torch.tensor([class_weight]).to(self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logits = self.model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(bx)

            epoch_record = {"epoch": float(epoch), "train_loss": total_loss / N}
            if validation_data is not None:
                train_metrics = self.evaluate_deterministic(X_train, y_train, criterion)
                val_metrics = self.evaluate_deterministic(*validation_data, criterion)
                epoch_record.update({
                    "train_accuracy": train_metrics["accuracy"],
                    "train_auroc": train_metrics["auroc"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_auroc": val_metrics["auroc"],
                })
            self.training_history.append(epoch_record)
            if verbose:
                logger.info("LSTM Classifier epoch metrics: %s", epoch_record)

        self._is_fitted = True
        return self

    def evaluate_deterministic(
        self,
        X: np.ndarray,
        y: np.ndarray,
        criterion: Optional[nn.Module] = None,
        batch_size: int = 128,
    ) -> Dict[str, float]:
        """Evaluate logits/probabilities with dropout disabled for diagnostics."""
        if self.model is None:
            raise RuntimeError("LSTMFailureClassifier is not fitted.")
        X_clean = np.nan_to_num(X, nan=0.0).astype(np.float32)
        y_clean = y.astype(np.float32)
        loader = DataLoader(
            TensorDataset(torch.tensor(X_clean), torch.tensor(y_clean)),
            batch_size=batch_size,
            shuffle=False,
        )
        self.model.eval()
        logits_out, targets, total_loss = [], [], 0.0
        with torch.no_grad():
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                logits = self.model(bx)
                if criterion is not None:
                    total_loss += criterion(logits, by).item() * len(bx)
                logits_out.append(logits.cpu().numpy())
                targets.append(by.cpu().numpy())
        logits_np = np.concatenate(logits_out)
        targets_np = np.concatenate(targets).astype(int)
        probabilities = 1.0 / (1.0 + np.exp(-logits_np))
        predictions = (probabilities >= 0.5).astype(int)
        try:
            from sklearn.metrics import accuracy_score, roc_auc_score
            accuracy = float(accuracy_score(targets_np, predictions))
            auroc = float(roc_auc_score(targets_np, probabilities)) if np.unique(targets_np).size == 2 else float("nan")
        except ImportError:  # pragma: no cover - sklearn is a project dependency
            accuracy, auroc = float("nan"), float("nan")
        return {
            "loss": total_loss / len(X_clean) if criterion is not None else float("nan"),
            "accuracy": accuracy,
            "auroc": auroc,
        }

    def predict_proba_with_uncertainty(
        self,
        X: np.ndarray,
        n_mc_samples: int = 15,
        batch_size: int = 128,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Monte Carlo Dropout inference:
        1. Get base prediction with dropout disabled (eval mode)
        2. Run n_mc_samples stochastic forward passes with dropout enabled for uncertainty

        Returns
        -------
        (mean_probabilities, epistemic_uncertainty_std)
        """
        if not self._is_fitted or self.model is None:
            raise RuntimeError("LSTMFailureClassifier is not fitted.")

        X_clean = np.nan_to_num(X, nan=0.0).astype(np.float32)
        dataset = TensorDataset(torch.tensor(X_clean))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Step 1: Get base predictions with dropout disabled (eval mode)
        self.model.eval()
        base_probs = []
        with torch.no_grad():
            for (bx,) in loader:
                bx = bx.to(self.device)
                logits = self.model(bx)
                probs = torch.sigmoid(logits).cpu().numpy()
                base_probs.append(probs)
        mean_probs = np.concatenate(base_probs, axis=0)

        # Step 2: Get MC-Dropout samples for uncertainty estimation
        self.model.train()
        all_mc_probs = []
        with torch.no_grad():
            for _ in range(n_mc_samples):
                pass_probs = []
                for (bx,) in loader:
                    bx = bx.to(self.device)
                    logits = self.model(bx)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    pass_probs.append(probs)
                all_mc_probs.append(np.concatenate(pass_probs, axis=0))

        # Compute uncertainty from stochastic samples
        stacked = np.stack(all_mc_probs, axis=0)
        uncertainty_std = np.std(stacked, axis=0)

        return mean_probs, uncertainty_std

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs, _ = self.predict_proba_with_uncertainty(X, n_mc_samples=5)
        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
