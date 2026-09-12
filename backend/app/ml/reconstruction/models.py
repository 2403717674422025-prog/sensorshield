"""
ml/reconstruction/models.py
============================
Model architectures for Signal Reconstruction & Imputation (Phase 7).

Includes:
  1. SpatialCrossSensorRegressor:
     Fast cross-sensor regression (Ridge & Gradient Boosting).
     Estimates target sensor s_i from peer healthy sensors at time t.

  2. PyTorchLSTMReconstructor:
     Temporal Recurrent Neural Network for multivariate sequence completion.
     Reconstructs target sensor trajectory using past temporal context + peer channels.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Spatial Cross-Sensor Regressor
# ---------------------------------------------------------------------------
class SpatialCrossSensorRegressor:
    """
    Predicts each sensor channel from the remaining peer sensor channels.
    Trains one regressor per sensor: f_i(s_1, ..., s_{i-1}, s_{i+1}, ..., s_S) -> s_i.
    """

    def __init__(self, alpha: float = 1.0, use_ensemble: bool = False):
        self.alpha = alpha
        self.use_ensemble = use_ensemble
        self.models: Dict[str, Union[Ridge, ExtraTreesRegressor]] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.sensor_cols: List[str] = []
        self._is_fitted = False

    def fit(self, X: np.ndarray, sensor_cols: List[str]) -> "SpatialCrossSensorRegressor":
        """
        Fit one model per sensor on 2D array of readings (N, n_sensors).
        """
        self.sensor_cols = list(sensor_cols)
        n_sensors = len(sensor_cols)

        if X.ndim == 3:
            X_2d = X.reshape(-1, n_sensors)
        else:
            X_2d = X

        # Clean NaNs if any
        valid_mask = ~np.isnan(X_2d).any(axis=1)
        X_clean = X_2d[valid_mask]

        if len(X_clean) == 0:
            raise ValueError("No complete clean rows found to fit SpatialCrossSensorRegressor.")

        self.models.clear()
        self.scalers.clear()

        for idx, sid in enumerate(self.sensor_cols):
            # Target is sensor idx, features are all other sensors
            y = X_clean[:, idx]
            peer_indices = [j for j in range(n_sensors) if j != idx]
            X_peers = X_clean[:, peer_indices]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_peers)

            if self.use_ensemble:
                model = ExtraTreesRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            else:
                model = Ridge(alpha=self.alpha)

            model.fit(X_scaled, y)

            self.models[sid] = model
            self.scalers[sid] = scaler

        self._is_fitted = True
        logger.info("Fitted SpatialCrossSensorRegressor for %d sensors", len(self.models))
        return self

    def predict_sensor(self, sensor_id: str, peer_values: Dict[str, float]) -> float:
        """
        Estimate a single sensor's value from current peer sensor readings.
        """
        if not self._is_fitted:
            raise RuntimeError("SpatialCrossSensorRegressor is not fitted.")

        if sensor_id not in self.models:
            raise KeyError(f"Sensor '{sensor_id}' not in fitted models: {self.sensor_cols}")

        peer_keys = [s for s in self.sensor_cols if s != sensor_id]
        feature_vector = np.array([[peer_values.get(s, 0.0) for s in peer_keys]], dtype=float)

        scaler = self.scalers[sensor_id]
        model = self.models[sensor_id]

        feature_vector_scaled = scaler.transform(np.nan_to_num(feature_vector, nan=0.0))
        predicted = float(model.predict(feature_vector_scaled)[0])
        return predicted


# ---------------------------------------------------------------------------
# 2. PyTorch Temporal LSTM Reconstructor Network
# ---------------------------------------------------------------------------
class PyTorchLSTMReconstructor(nn.Module):
    """
    Multi-sensor Sequence-to-Sequence LSTM Reconstructor.
    Given a window with some missing/corrupted values, outputs denoised
    and imputed complete sequence of shape (batch_size, seq_len, n_sensors).
    """

    def __init__(
        self,
        n_sensors: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_sensors = n_sensors
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Bi-directional LSTM captures past & future context in window
        self.lstm = nn.LSTM(
            input_size=n_sensors,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_sensors),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, n_sensors)
        returns: (batch_size, seq_len, n_sensors)
        """
        lstm_out, _ = self.lstm(x)
        reconstruction = self.head(lstm_out)
        return reconstruction
