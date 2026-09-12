"""
tests/unit/test_lstm_autoencoder.py
====================================
Unit tests for Phase 5 — Multivariate LSTM Autoencoder Anomaly Detection.

Coverage:
  - PyTorchLSTMAutoencoder forward pass, encode/decode shapes
  - LSTMAutoencoderDetector fit, predict, score, score_per_sensor, reconstruct
  - Calibration of threshold on validation data
  - Persistence (save and load model checkpoint)
  - Anomaly scoring discrimination (corrupted signals have higher reconstruction error)
  - Integration with evaluate_detector
"""

import os
import tempfile
import numpy as np
import pytest
import torch

from app.ml.anomaly_detection.lstm_autoencoder import (
    PyTorchLSTMAutoencoder,
    LSTMAutoencoderDetector,
)
from app.ml.anomaly_detection.evaluation import evaluate_detector, DetectionResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_clean_windows():
    """Deterministic sine-wave sensor windows (N=100, W=20, S=4)."""
    rng = np.random.default_rng(42)
    t = np.linspace(0, 4 * np.pi, 20)
    windows = []
    for _ in range(100):
        phase = rng.uniform(0, np.pi)
        base = np.stack([
            np.sin(t + phase),
            np.cos(t + phase),
            np.sin(2 * t + phase) * 0.5,
            np.cos(0.5 * t + phase) * 0.8,
        ], axis=-1)
        noise = rng.normal(0, 0.05, base.shape)
        windows.append(base + noise)
    return np.array(windows, dtype=np.float32)


@pytest.fixture
def sample_anomaly_windows():
    """Anomalous sensor windows with severe bias and noise (N=30, W=20, S=4)."""
    rng = np.random.default_rng(99)
    windows = []
    for _ in range(30):
        anom = rng.normal(5.0, 2.0, (20, 4))
        windows.append(anom)
    return np.array(windows, dtype=np.float32)


# ===========================================================================
# 1. PyTorch Network Architecture Tests
# ===========================================================================
class TestPyTorchLSTMAutoencoder:
    def test_forward_output_shape(self):
        model = PyTorchLSTMAutoencoder(
            n_sensors=6, seq_len=30, hidden_dim=32, latent_dim=8
        )
        x = torch.randn(16, 30, 6)
        out = model(x)
        assert out.shape == (16, 30, 6)

    def test_encode_decode_shapes(self):
        model = PyTorchLSTMAutoencoder(
            n_sensors=4, seq_len=20, hidden_dim=32, latent_dim=8
        )
        x = torch.randn(8, 20, 4)
        latent = model.encode(x)
        assert latent.shape == (8, 8)

        recon = model.decode(latent, seq_len=20)
        assert recon.shape == (8, 20, 4)


# ===========================================================================
# 2. Detector Interface & Training Tests
# ===========================================================================
class TestLSTMAutoencoderDetector:
    def test_predict_before_fit_raises(self, sample_clean_windows):
        detector = LSTMAutoencoderDetector(hidden_dim=16, latent_dim=4)
        with pytest.raises(RuntimeError, match="fit"):
            detector.predict(sample_clean_windows)

    def test_score_before_fit_raises(self, sample_clean_windows):
        detector = LSTMAutoencoderDetector(hidden_dim=16, latent_dim=4)
        with pytest.raises(RuntimeError, match="fit"):
            detector.score(sample_clean_windows)

    def test_fit_and_score_shapes(self, sample_clean_windows):
        detector = LSTMAutoencoderDetector(
            hidden_dim=16, latent_dim=4, lr=0.01, device="cpu"
        )
        detector.fit(sample_clean_windows, epochs=5, batch_size=32)

        assert detector._is_fitted is True
        assert len(detector.train_history["train_loss"]) == 5

        scores = detector.score(sample_clean_windows)
        assert scores.shape == (len(sample_clean_windows),)
        assert not np.isnan(scores).any()

    def test_predict_binary_outputs(self, sample_clean_windows):
        detector = LSTMAutoencoderDetector(
            hidden_dim=16, latent_dim=4, lr=0.01, device="cpu"
        )
        detector.fit(sample_clean_windows, epochs=5, batch_size=32)

        preds = detector.predict(sample_clean_windows)
        assert preds.shape == (len(sample_clean_windows),)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_score_per_sensor(self, sample_clean_windows):
        detector = LSTMAutoencoderDetector(
            hidden_dim=16, latent_dim=4, lr=0.01, device="cpu"
        )
        detector.fit(sample_clean_windows, epochs=5, batch_size=32)

        sensor_scores = detector.score_per_sensor(sample_clean_windows)
        assert sensor_scores.shape == (len(sample_clean_windows), 4)
        assert (sensor_scores >= 0.0).all()

    def test_reconstruct_shape(self, sample_clean_windows):
        detector = LSTMAutoencoderDetector(
            hidden_dim=16, latent_dim=4, lr=0.01, device="cpu"
        )
        detector.fit(sample_clean_windows, epochs=5, batch_size=32)

        recons = detector.reconstruct(sample_clean_windows)
        assert recons.shape == sample_clean_windows.shape

    def test_anomalies_score_higher_than_clean(
        self, sample_clean_windows, sample_anomaly_windows
    ):
        detector = LSTMAutoencoderDetector(
            hidden_dim=32, latent_dim=8, lr=0.01, device="cpu"
        )
        detector.fit(sample_clean_windows, epochs=12, batch_size=32)

        clean_scores = detector.score(sample_clean_windows)
        anom_scores = detector.score(sample_anomaly_windows)

        assert np.mean(anom_scores) > np.mean(clean_scores) * 2.0


# ===========================================================================
# 3. Persistence Tests (Save / Load)
# ===========================================================================
class TestPersistence:
    def test_save_and_load_roundtrip(self, sample_clean_windows):
        detector = LSTMAutoencoderDetector(
            hidden_dim=16, latent_dim=4, lr=0.01, device="cpu"
        )
        detector.fit(sample_clean_windows, epochs=5, batch_size=32)

        orig_scores = detector.score(sample_clean_windows)

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "lstm_ae.pt")
            detector.save(ckpt_path)
            assert os.path.exists(ckpt_path)

            loaded_detector = LSTMAutoencoderDetector.load(ckpt_path, device="cpu")
            assert loaded_detector._is_fitted is True

            loaded_scores = loaded_detector.score(sample_clean_windows)
            np.testing.assert_allclose(orig_scores, loaded_scores, rtol=1e-5)


# ===========================================================================
# 4. Evaluation Harness Integration Tests
# ===========================================================================
class TestEvaluationIntegration:
    def test_evaluate_detector_with_lstm_ae(
        self, sample_clean_windows, sample_anomaly_windows
    ):
        detector = LSTMAutoencoderDetector(
            hidden_dim=16, latent_dim=4, lr=0.01, device="cpu"
        )

        n_clean = len(sample_clean_windows)
        n_anom = len(sample_anomaly_windows)

        X_test = np.concatenate([sample_clean_windows, sample_anomaly_windows], axis=0)
        y_test = np.concatenate([np.zeros(n_clean), np.ones(n_anom)]).astype(int)
        ft_test = np.array(["NORMAL"] * n_clean + ["BIAS"] * n_anom, dtype=object)

        result = evaluate_detector(
            detector=detector,
            X_normal_train=sample_clean_windows,
            X_test=X_test,
            y_test=y_test,
            fault_types=ft_test,
        )

        assert isinstance(result, DetectionResult)
        assert result.detector_name == "LSTMAutoencoder"
        assert result.overall_auroc > 0.8
        assert len(result.per_fault) == 1
        assert result.per_fault[0].fault_type == "BIAS"
