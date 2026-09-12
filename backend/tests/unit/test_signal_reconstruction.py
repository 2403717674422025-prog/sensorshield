"""
tests/unit/test_signal_reconstruction.py
=========================================
Unit tests for Phase 7 — Signal Reconstruction Engine.

Coverage:
  - SpatialCrossSensorRegressor: fit, predict_sensor, validation
  - PyTorchLSTMReconstructor: forward pass shapes
  - SignalReconstructor:
      - fit on multivariate sensor sequences
      - reconstruct_reading (provenance, confidence, method)
      - reconstruct_window (imputation of NaN / degraded channels)
      - evaluate_reconstruction (MAE, RMSE, R^2)
      - Model persistence (save / load checkpoint roundtrip)
"""

import os
import tempfile
import numpy as np
import pytest
import torch

from app.ml.reconstruction import (
    SignalReconstructor,
    ReconstructionResult,
    SpatialCrossSensorRegressor,
    PyTorchLSTMReconstructor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def correlated_sensor_windows():
    """Generate correlated 3-sensor dataset (N=80, W=25, S=3)."""
    rng = np.random.default_rng(42)
    t = np.linspace(0, 4 * np.pi, 25)
    windows = []
    for _ in range(80):
        phase = rng.uniform(0, np.pi)
        s0 = np.sin(t + phase) * 10.0 + 50.0
        # s1 and s2 are physically coupled to s0
        s1 = 0.8 * s0 + rng.normal(0, 0.2, s0.shape) + 5.0
        s2 = -0.5 * s0 + rng.normal(0, 0.2, s0.shape) + 70.0
        win = np.stack([s0, s1, s2], axis=-1)
        windows.append(win)
    return np.array(windows, dtype=np.float32)


# ===========================================================================
# 1. Spatial Model Tests
# ===========================================================================
class TestSpatialRegressor:
    def test_fit_and_predict(self, correlated_sensor_windows):
        reg = SpatialCrossSensorRegressor(alpha=1.0)
        sensor_cols = ["S0", "S1", "S2"]
        reg.fit(correlated_sensor_windows, sensor_cols)

        assert reg._is_fitted is True

        # Test predicting S0 given S1 and S2
        sample_row = correlated_sensor_windows[0, 0, :]
        true_s0 = sample_row[0]
        peer_vals = {"S1": sample_row[1], "S2": sample_row[2]}

        pred_s0 = reg.predict_sensor("S0", peer_vals)
        # Check prediction is close to ground truth
        assert abs(pred_s0 - true_s0) < 2.0

    def test_predict_unfitted_raises(self):
        reg = SpatialCrossSensorRegressor()
        with pytest.raises(RuntimeError, match="not fitted"):
            reg.predict_sensor("S0", {"S1": 50.0})


# ===========================================================================
# 2. PyTorch LSTM Reconstructor Architecture Tests
# ===========================================================================
class TestLSTMReconstructorNet:
    def test_forward_output_shape(self):
        model = PyTorchLSTMReconstructor(n_sensors=4, hidden_dim=32, num_layers=2)
        x = torch.randn(8, 20, 4)
        out = model(x)
        assert out.shape == (8, 20, 4)


# ===========================================================================
# 3. SignalReconstructor Engine Tests
# ===========================================================================
class TestSignalReconstructor:
    @pytest.fixture
    def fitted_reconstructor(self, correlated_sensor_windows):
        rec = SignalReconstructor(hidden_dim=32, num_layers=1, device="cpu")
        rec.fit(correlated_sensor_windows, sensor_cols=["S0", "S1", "S2"], epochs=6, batch_size=32)
        return rec

    def test_fit_success(self, fitted_reconstructor):
        assert fitted_reconstructor._is_fitted is True
        assert fitted_reconstructor.lstm_model is not None
        assert "S0" in fitted_reconstructor.sensor_means

    def test_reconstruct_reading_spatial(self, fitted_reconstructor, correlated_sensor_windows):
        sample_row = correlated_sensor_windows[0, 0, :]
        peer_vals = {"S1": sample_row[1], "S2": sample_row[2]}

        res = fitted_reconstructor.reconstruct_reading(
            sensor_id="S0",
            observed_value=None,
            peer_values=peer_vals,
        )

        assert isinstance(res, ReconstructionResult)
        assert res.sensor_id == "S0"
        assert res.method == "spatial_regression"
        assert res.confidence > 0.8
        assert abs(res.reconstructed_value - sample_row[0]) < 2.5

    def test_reconstruct_reading_temporal(self, fitted_reconstructor, correlated_sensor_windows):
        history = np.copy(correlated_sensor_windows[0])
        # Mask target sensor S0 at the latest timestep
        history[-1, 0] = np.nan

        peer_vals = {"S1": history[-1, 1], "S2": history[-1, 2]}
        res = fitted_reconstructor.reconstruct_reading(
            sensor_id="S0",
            observed_value=np.nan,
            peer_values=peer_vals,
            history_window=history,
        )

        assert res.method == "lstm_temporal"
        assert res.confidence == 0.90
        assert not np.isnan(res.reconstructed_value)

    def test_reconstruct_window_imputation(self, fitted_reconstructor, correlated_sensor_windows):
        corrupted_window = np.copy(correlated_sensor_windows[5])
        # Inject NaNs into S0
        corrupted_window[10:18, 0] = np.nan

        repaired = fitted_reconstructor.reconstruct_window(corrupted_window)
        assert repaired.shape == corrupted_window.shape
        assert not np.isnan(repaired).any()

    def test_evaluate_reconstruction_metrics(self, fitted_reconstructor, correlated_sensor_windows):
        gt = correlated_sensor_windows[:20]
        corrupted = np.copy(gt)
        corrupted[:, 5:15, 0] = np.nan  # Dropout on S0

        metrics = fitted_reconstructor.evaluate_reconstruction(gt, corrupted)
        assert "S0" in metrics
        assert "OVERALL" in metrics
        assert metrics["OVERALL"]["MAE"] >= 0.0
        assert metrics["OVERALL"]["RMSE"] >= 0.0

    def test_save_and_load_roundtrip(self, fitted_reconstructor, correlated_sensor_windows):
        with tempfile.TemporaryDirectory() as tmpdir:
            pt_path = os.path.join(tmpdir, "recon.pt")
            pkl_path = os.path.join(tmpdir, "spatial.pkl")

            fitted_reconstructor.save(pt_path, pkl_path)
            assert os.path.exists(pt_path)
            assert os.path.exists(pkl_path)

            loaded = SignalReconstructor.load(pt_path, pkl_path, device="cpu")
            assert loaded._is_fitted is True

            # Check inference on loaded model
            sample_row = correlated_sensor_windows[0, 0, :]
            res = loaded.reconstruct_reading(
                sensor_id="S0",
                observed_value=None,
                peer_values={"S1": sample_row[1], "S2": sample_row[2]},
            )
            assert res.confidence > 0.8
            assert abs(res.reconstructed_value - sample_row[0]) < 2.5
