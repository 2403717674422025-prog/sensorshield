"""
ml/anomaly_detection/lstm_autoencoder.py
=========================================
Multivariate LSTM Autoencoder for Anomaly Detection (Phase 5).

Architecture
------------
- Encoder:
    Input: (batch_size, seq_len, n_sensors)
    Layer 1: LSTM(n_sensors -> hidden_dim1)
    Layer 2: LSTM(hidden_dim1 -> latent_dim)
- Latent Representation:
    Bottleneck vector: (batch_size, latent_dim)
- Decoder:
    RepeatVector: (batch_size, seq_len, latent_dim)
    Layer 1: LSTM(latent_dim -> hidden_dim1)
    Layer 2: LSTM(hidden_dim1 -> hidden_dim1)
    Output Projection: Linear(hidden_dim1 -> n_sensors)
    Reconstruction: (batch_size, seq_len, n_sensors)

Detection Principle
-------------------
Trained SOLELY on healthy/normal sensor windows using Mean Squared Error (MSE).
Normal sensor patterns have low reconstruction error. Corrupted/anomalous signals
produce large reconstruction errors.

The reconstruction error serves as the anomaly score:
  - Global score per window: mean MSE across all timesteps and sensors
  - Per-sensor score: MSE per sensor channel (enables sensor fault localization)
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .detectors import BaseDetector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PyTorch Neural Network Module
# ---------------------------------------------------------------------------
class PyTorchLSTMAutoencoder(nn.Module):
    """
    Multivariate LSTM Autoencoder network.

    Parameters
    ----------
    n_sensors : int
        Number of sensor channels (features per timestep).
    seq_len : int
        Sequence length / window size (e.g. 30).
    hidden_dim : int
        Dimension of intermediate LSTM layers.
    latent_dim : int
        Dimension of latent bottleneck layer.
    dropout : float
        Dropout probability between recurrent layers.
    """

    def __init__(
        self,
        n_sensors: int = 6,
        seq_len: int = 30,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_sensors = n_sensors
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder
        self.enc_lstm1 = nn.LSTM(
            input_size=n_sensors,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.enc_lstm2 = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=latent_dim,
            batch_first=True,
        )

        # Decoder
        self.dec_lstm1 = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.dec_lstm2 = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_dim, n_sensors)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode sequence to latent representation.
        x: (batch_size, seq_len, n_sensors)
        returns: (batch_size, latent_dim)
        """
        out, _ = self.enc_lstm1(x)
        out = self.dropout(out)
        _, (hn, _) = self.enc_lstm2(out)
        # hn is (1, batch_size, latent_dim)
        latent = hn[-1]
        return latent

    def decode(self, latent: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """
        Decode latent bottleneck back into reconstructed sequence.
        latent: (batch_size, latent_dim)
        returns: (batch_size, seq_len, n_sensors)
        """
        target_len = seq_len or self.seq_len
        # Repeat latent vector across time: (batch_size, seq_len, latent_dim)
        repeated = latent.unsqueeze(1).repeat(1, target_len, 1)

        out, _ = self.dec_lstm1(repeated)
        out = self.dropout(out)
        out, _ = self.dec_lstm2(out)
        reconstruction = self.output_layer(out)
        return reconstruction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: reconstruct input sequence."""
        latent = self.encode(x)
        reconstruction = self.decode(latent, seq_len=x.shape[1])
        return reconstruction


# ---------------------------------------------------------------------------
# High-Level Detector Wrapper
# ---------------------------------------------------------------------------
class LSTMAutoencoderDetector(BaseDetector):
    """
    Anomaly detector based on the LSTM Autoencoder.
    Implements the standard BaseDetector interface with PyTorch backend.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension of encoder/decoder LSTM.
    latent_dim : int
        Latent bottleneck dimension.
    lr : float
        Learning rate for Adam optimizer.
    weight_decay : float
        L2 regularization weight decay.
    percentile : float
        Percentile of normal validation reconstruction error used for threshold.
    device : Optional[str]
        'cuda', 'cpu', or auto-detected if None.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        percentile: float = 99.0,
        device: Optional[str] = None,
    ):
        super().__init__("LSTMAutoencoder")
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.percentile = percentile

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: Optional[PyTorchLSTMAutoencoder] = None
        self.train_history: Dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    def _init_model(self, n_sensors: int, seq_len: int) -> None:
        self.model = PyTorchLSTMAutoencoder(
            n_sensors=n_sensors,
            seq_len=seq_len,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            dropout=self.dropout,
        ).to(self.device)

    def fit(
        self,
        X_normal: np.ndarray,
        val_split: float = 0.15,
        epochs: int = 25,
        batch_size: int = 64,
        patience: int = 5,
        verbose: bool = False,
    ) -> "LSTMAutoencoderDetector":
        """
        Train the autoencoder on clean/normal windows.

        Parameters
        ----------
        X_normal : np.ndarray
            Clean training windows of shape (N, seq_len, n_sensors).
        val_split : float
            Fraction of data held out for validation and threshold calibration.
        epochs : int
            Maximum number of training epochs.
        batch_size : int
            Batch size.
        patience : int
            Early stopping patience.
        verbose : bool
            Whether to log epoch progress.
        """
        X_clean = np.nan_to_num(X_normal, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        N, seq_len, n_sensors = X_clean.shape

        self._init_model(n_sensors, seq_len)

        # Train/Validation split
        n_val = int(N * val_split)
        n_train = N - n_val

        X_tr = torch.tensor(X_clean[:n_train], dtype=torch.float32)
        X_va = torch.tensor(X_clean[n_train:], dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(X_tr), batch_size=batch_size, shuffle=True, drop_last=len(X_tr) > batch_size
        )
        val_loader = DataLoader(TensorDataset(X_va), batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        self.train_history = {"train_loss": [], "val_loss": []}

        self.model.train()
        for epoch in range(1, epochs + 1):
            total_train_loss = 0.0
            for (batch_x,) in train_loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                recon = self.model(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_train_loss += loss.item() * len(batch_x)

            avg_train_loss = total_train_loss / max(n_train, 1)

            # Validation
            self.model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for (batch_x,) in val_loader:
                    batch_x = batch_x.to(self.device)
                    recon = self.model(batch_x)
                    loss = criterion(recon, batch_x)
                    total_val_loss += loss.item() * len(batch_x)

            avg_val_loss = total_val_loss / max(n_val, 1) if n_val > 0 else avg_train_loss

            self.train_history["train_loss"].append(avg_train_loss)
            self.train_history["val_loss"].append(avg_val_loss)

            if verbose and epoch % 5 == 0:
                logger.info(
                    "Epoch %d/%d - Train Loss: %.6f - Val Loss: %.6f",
                    epoch,
                    epochs,
                    avg_train_loss,
                    avg_val_loss,
                )

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        logger.info("Early stopping triggered at epoch %d", epoch)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self._is_fitted = True

        # Calibrate threshold on normal validation (or training) set
        calib_data = X_clean[n_train:] if n_val > 0 else X_clean
        normal_scores = self.score(calib_data)
        self._set_threshold_from_normal(normal_scores, percentile=self.percentile)

        return self

    def reconstruct(self, X: np.ndarray, batch_size: int = 128) -> np.ndarray:
        """
        Reconstruct input windows.

        Parameters
        ----------
        X : np.ndarray
            Windows array of shape (N, seq_len, n_sensors).

        Returns
        -------
        reconstructed : np.ndarray
            Reconstructed array of shape (N, seq_len, n_sensors).
        """
        if not self._is_fitted or self.model is None:
            raise RuntimeError("LSTMAutoencoderDetector: fit() must be called before reconstruct()")

        self.model.eval()
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        dataset = TensorDataset(torch.tensor(X_clean, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        recons = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                out = self.model(batch_x)
                recons.append(out.cpu().numpy())

        return np.concatenate(recons, axis=0)

    def score(self, X: np.ndarray, batch_size: int = 128) -> np.ndarray:
        """
        Compute global anomaly score per window (mean squared reconstruction error).

        Returns
        -------
        scores : np.ndarray of shape (N,)
        """
        if not self._is_fitted:
            raise RuntimeError("LSTMAutoencoderDetector: fit() must be called before score()")

        recons = self.reconstruct(X, batch_size=batch_size)
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # MSE across time and sensors: (N, W, S) -> (N,)
        squared_errors = (X_clean - recons) ** 2
        # Penalize NaN windows heavily if original had missing data
        nan_penalty = np.mean(np.isnan(X), axis=(1, 2)) * 10.0
        scores = np.mean(squared_errors, axis=(1, 2)) + nan_penalty
        return scores

    def score_per_sensor(self, X: np.ndarray, batch_size: int = 128) -> np.ndarray:
        """
        Compute reconstruction error per sensor channel (for fault diagnosis).

        Returns
        -------
        sensor_scores : np.ndarray of shape (N, n_sensors)
        """
        if not self._is_fitted:
            raise RuntimeError("LSTMAutoencoderDetector: fit() must be called before score_per_sensor()")

        recons = self.reconstruct(X, batch_size=batch_size)
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # MSE across time dimension: (N, W, S) -> (N, S)
        squared_errors = (X_clean - recons) ** 2
        nan_penalty = np.mean(np.isnan(X), axis=1) * 10.0
        sensor_scores = np.mean(squared_errors, axis=1) + nan_penalty
        return sensor_scores

    def save(self, filepath: str) -> None:
        """Save detector weights, threshold, and configuration to disk."""
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Cannot save unfitted detector.")

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            "model_state": self.model.state_dict(),
            "n_sensors": self.model.n_sensors,
            "seq_len": self.model.seq_len,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "dropout": self.dropout,
            "threshold": self._threshold,
            "percentile": self.percentile,
            "train_history": self.train_history,
        }
        torch.save(payload, filepath)
        logger.info("Saved LSTMAutoencoder to %s", filepath)

    @classmethod
    def load(cls, filepath: str, device: Optional[str] = None) -> "LSTMAutoencoderDetector":
        """Load saved detector checkpoint."""
        checkpoint = torch.load(filepath, map_location=device or "cpu")
        detector = cls(
            hidden_dim=checkpoint["hidden_dim"],
            latent_dim=checkpoint["latent_dim"],
            dropout=checkpoint.get("dropout", 0.1),
            percentile=checkpoint["percentile"],
            device=device,
        )
        detector._init_model(
            n_sensors=checkpoint["n_sensors"],
            seq_len=checkpoint["seq_len"],
        )
        detector.model.load_state_dict(checkpoint["model_state"])
        detector._threshold = checkpoint["threshold"]
        detector.train_history = checkpoint.get("train_history", {})
        detector._is_fitted = True
        return detector
