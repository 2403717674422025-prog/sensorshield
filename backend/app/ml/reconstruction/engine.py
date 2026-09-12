"""
ml/reconstruction/engine.py
============================
Signal Reconstruction Engine (Phase 7).

Provides unified signal imputation and denoising for untrusted or corrupted sensors.
Combines Spatial Cross-Sensor Regression and Temporal LSTM Reconstructors.

Outputs:
  - Reconstructed sensor value(s)
  - Reconstruction Confidence (0.0–1.0)
  - Reconstruction Method label
  - Full window / batch reconstruction
  - MAE, RMSE, R^2 evaluation metrics
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .models import SpatialCrossSensorRegressor, PyTorchLSTMReconstructor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reconstruction Result Dataclass
# ---------------------------------------------------------------------------
@dataclass
class ReconstructionResult:
    """Estimated value, confidence score, and provenance for a single reading."""
    sensor_id: str
    observed_value: Optional[float]
    reconstructed_value: float
    confidence: float                     # 0.0–1.0
    method: str                           # "spatial_regression" | "lstm_temporal" | "fallback"
    timestamp: Optional[datetime] = None
    residual_error_bound: float = 0.0

    def to_dict(self) -> dict:
        return {
            "sensor_id": self.sensor_id,
            "observed_value": self.observed_value,
            "reconstructed_value": round(self.reconstructed_value, 4),
            "confidence": round(self.confidence, 3),
            "method": self.method,
            "residual_error_bound": round(self.residual_error_bound, 4),
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


# ---------------------------------------------------------------------------
# Signal Reconstructor Engine
# ---------------------------------------------------------------------------
class SignalReconstructor:
    """
    Unified Signal Reconstruction Engine.

    Parameters
    ----------
    hidden_dim : int
        Hidden size for the PyTorch LSTM Reconstructor.
    device : Optional[str]
        Compute device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        device: Optional[str] = None,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.sensor_cols: List[str] = []
        self.spatial_regressor = SpatialCrossSensorRegressor(alpha=1.0)
        self.lstm_model: Optional[PyTorchLSTMReconstructor] = None
        self.sensor_means: Dict[str, float] = {}
        self.sensor_stds: Dict[str, float] = {}
        self._is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        sensor_cols: List[str],
        epochs: int = 20,
        batch_size: int = 64,
        lr: float = 0.003,
        verbose: bool = False,
    ) -> "SignalReconstructor":
        """
        Fit both Spatial and LSTM Reconstructors on clean sensor windows.

        Parameters
        ----------
        X_train : np.ndarray
            Shape (N, seq_len, n_sensors) or (N, n_sensors).
        sensor_cols : List[str]
            Sensor channel names in canonical column order.
        """
        self.sensor_cols = list(sensor_cols)
        n_sensors = len(sensor_cols)

        # 1. Fit Spatial Cross-Sensor Regressor
        self.spatial_regressor.fit(X_train, sensor_cols)

        # 2. Store per-sensor normal mean & std
        if X_train.ndim == 3:
            flat_data = X_train.reshape(-1, n_sensors)
        else:
            flat_data = X_train

        for idx, sid in enumerate(self.sensor_cols):
            col = flat_data[:, idx]
            valid = col[~np.isnan(col)]
            self.sensor_means[sid] = float(np.mean(valid)) if len(valid) > 0 else 0.0
            self.sensor_stds[sid] = float(np.std(valid)) if len(valid) > 1 else 1.0

        # 3. Fit PyTorch Sequence LSTM Reconstructor (if 3D windows provided)
        if X_train.ndim == 3:
            N, seq_len, _ = X_train.shape
            self.lstm_model = PyTorchLSTMReconstructor(
                n_sensors=n_sensors,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout,
            ).to(self.device)

            X_clean = np.nan_to_num(X_train, nan=0.0).astype(np.float32)
            dataset = TensorDataset(torch.tensor(X_clean), torch.tensor(X_clean))
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            self.lstm_model.train()
            for epoch in range(1, epochs + 1):
                total_loss = 0.0
                for bx, by in loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    # Simulate random sensor dropouts during training to force cross-reconstruction
                    mask = torch.rand_like(bx) > 0.15
                    masked_input = bx * mask.float()

                    optimizer.zero_grad()
                    pred = self.lstm_model(masked_input)
                    loss = criterion(pred, by)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(bx)

                if verbose and epoch % 5 == 0:
                    logger.info("Reconstructor Epoch %d/%d - Loss: %.6f", epoch, epochs, total_loss / N)

        self._is_fitted = True
        return self

    def reconstruct_reading(
        self,
        sensor_id: str,
        observed_value: Optional[float],
        peer_values: Dict[str, float],
        history_window: Optional[np.ndarray] = None,
        timestamp: Optional[datetime] = None,
    ) -> ReconstructionResult:
        """
        Estimate a single sensor reading from current peer readings and/or past history.

        Parameters
        ----------
        sensor_id : str
            Degraded sensor ID.
        observed_value : Optional[float]
            Raw observed value (or None if missing).
        peer_values : Dict[str, float]
            Readings of healthy peer sensors at the same timestep.
        history_window : Optional[np.ndarray]
            Recent (W, S) multi-sensor history window.
        """
        if not self._is_fitted:
            raise RuntimeError("SignalReconstructor: fit() must be called first.")

        # Method 1: Temporal LSTM Reconstructor if history window available
        if history_window is not None and self.lstm_model is not None:
            recon_win = self.reconstruct_window(history_window)
            sensor_idx = self.sensor_cols.index(sensor_id)
            est_val = float(recon_win[-1, sensor_idx])
            confidence = 0.90
            method = "lstm_temporal"
            return ReconstructionResult(
                sensor_id=sensor_id,
                observed_value=observed_value,
                reconstructed_value=est_val,
                confidence=confidence,
                method=method,
                timestamp=timestamp,
                residual_error_bound=self.sensor_stds.get(sensor_id, 1.0) * 0.2,
            )

        # Method 2: Spatial Cross-Sensor Regression
        try:
            est_val = self.spatial_regressor.predict_sensor(sensor_id, peer_values)
            # Confidence scales with number of valid peers available
            n_peers = len([k for k, v in peer_values.items() if not np.isnan(v) and k != sensor_id])
            total_peers = max(1, len(self.sensor_cols) - 1)
            confidence = float(np.clip(0.5 + 0.45 * (n_peers / total_peers), 0.0, 0.95))
            method = "spatial_regression"
        except Exception as exc:
            logger.warning("Spatial reconstruction failed for %s: %s. Using fallback mean.", sensor_id, exc)
            est_val = self.sensor_means.get(sensor_id, 0.0)
            confidence = 0.30
            method = "fallback"

        return ReconstructionResult(
            sensor_id=sensor_id,
            observed_value=observed_value,
            reconstructed_value=est_val,
            confidence=confidence,
            method=method,
            timestamp=timestamp,
            residual_error_bound=self.sensor_stds.get(sensor_id, 1.0) * 0.3,
        )

    def reconstruct_window(
        self,
        window: np.ndarray,
        degraded_sensors: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Reconstruct and denoise an entire multi-sensor window (seq_len, n_sensors).

        Parameters
        ----------
        window : np.ndarray
            Shape (seq_len, n_sensors) or (N, seq_len, n_sensors).
        degraded_sensors : Optional[List[str]]
            Specific sensor channels to overwrite with reconstructed estimates.
            If None, replaces all NaN or severely degraded channels.
        """
        if not self._is_fitted:
            raise RuntimeError("SignalReconstructor: fit() must be called first.")

        single_window = window.ndim == 2
        if single_window:
            w_arr = np.expand_dims(window, axis=0)
        else:
            w_arr = window

        X_in = np.nan_to_num(w_arr, nan=0.0).astype(np.float32)

        if self.lstm_model is not None:
            self.lstm_model.eval()
            with torch.no_grad():
                tensor_in = torch.tensor(X_in, dtype=torch.float32).to(self.device)
                out = self.lstm_model(tensor_in).cpu().numpy()
        else:
            # Fallback to spatial row-by-row
            out = np.copy(X_in)
            for b in range(out.shape[0]):
                for t in range(out.shape[1]):
                    row_peers = {sid: out[b, t, idx] for idx, sid in enumerate(self.sensor_cols)}
                    for idx, sid in enumerate(self.sensor_cols):
                        out[b, t, idx] = self.spatial_regressor.predict_sensor(sid, row_peers)

        # Merge: If degraded_sensors specified, only replace those; preserve healthy readings
        result = np.copy(w_arr)
        if degraded_sensors is not None:
            for sid in degraded_sensors:
                if sid in self.sensor_cols:
                    s_idx = self.sensor_cols.index(sid)
                    result[:, :, s_idx] = out[:, :, s_idx]
        else:
            # Overwrite anywhere input was NaN
            nan_mask = np.isnan(w_arr)
            result[nan_mask] = out[nan_mask]

        return result[0] if single_window else result

    def evaluate_reconstruction(
        self,
        ground_truth: np.ndarray,
        corrupted_input: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """
        Benchmark reconstruction quality against ground-truth clean sensor signals.

        Returns
        -------
        metrics_per_sensor : Dict[str, Dict[str, float]]
            Contains MAE, RMSE, and R2 per sensor channel and overall.
        """
        reconstructed = self.reconstruct_window(corrupted_input)

        if ground_truth.ndim == 3:
            gt_flat = ground_truth.reshape(-1, len(self.sensor_cols))
            rec_flat = reconstructed.reshape(-1, len(self.sensor_cols))
        else:
            gt_flat = ground_truth
            rec_flat = reconstructed

        results = {}
        for idx, sid in enumerate(self.sensor_cols):
            y_true = gt_flat[:, idx]
            y_pred = rec_flat[:, idx]

            valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
            y_t_val = y_true[valid_mask]
            y_p_val = y_pred[valid_mask]

            if len(y_t_val) == 0:
                continue

            mae = float(mean_absolute_error(y_t_val, y_p_val))
            rmse = float(np.sqrt(mean_squared_error(y_t_val, y_p_val)))
            r2 = float(r2_score(y_t_val, y_p_val)) if np.var(y_t_val) > 1e-6 else 1.0

            results[sid] = {
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
                "R2": round(r2, 4),
            }

        # Overall aggregate
        overall_mae = float(np.mean([m["MAE"] for m in results.values()]))
        overall_rmse = float(np.mean([m["RMSE"] for m in results.values()]))
        results["OVERALL"] = {
            "MAE": round(overall_mae, 4),
            "RMSE": round(overall_rmse, 4),
        }
        return results

    def save(self, filepath_pt: str, filepath_pkl: str) -> None:
        """Save PyTorch LSTM weights and Scikit-learn regressors."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath_pt)), exist_ok=True)
        if self.lstm_model is not None:
            payload = {
                "lstm_state": self.lstm_model.state_dict(),
                "sensor_cols": self.sensor_cols,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "sensor_means": self.sensor_means,
                "sensor_stds": self.sensor_stds,
            }
            torch.save(payload, filepath_pt)

        with open(filepath_pkl, "wb") as f:
            pickle.dump(self.spatial_regressor, f)

        logger.info("Saved SignalReconstructor models to %s and %s", filepath_pt, filepath_pkl)

    @classmethod
    def load(cls, filepath_pt: str, filepath_pkl: str, device: Optional[str] = None) -> "SignalReconstructor":
        """Load trained SignalReconstructor."""
        checkpoint = torch.load(filepath_pt, map_location=device or "cpu")
        reconstructor = cls(
            hidden_dim=checkpoint["hidden_dim"],
            num_layers=checkpoint["num_layers"],
            dropout=checkpoint.get("dropout", 0.1),
            device=device,
        )
        reconstructor.sensor_cols = checkpoint["sensor_cols"]
        reconstructor.sensor_means = checkpoint["sensor_means"]
        reconstructor.sensor_stds = checkpoint["sensor_stds"]

        reconstructor.lstm_model = PyTorchLSTMReconstructor(
            n_sensors=len(reconstructor.sensor_cols),
            hidden_dim=reconstructor.hidden_dim,
            num_layers=reconstructor.num_layers,
            dropout=reconstructor.dropout,
        ).to(reconstructor.device)
        reconstructor.lstm_model.load_state_dict(checkpoint["lstm_state"])

        with open(filepath_pkl, "rb") as f:
            reconstructor.spatial_regressor = pickle.load(f)

        reconstructor._is_fitted = True
        return reconstructor
