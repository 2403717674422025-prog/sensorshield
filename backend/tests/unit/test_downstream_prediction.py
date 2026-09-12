"""
tests/unit/test_downstream_prediction.py
=========================================
Unit tests for Phase 8 — Downstream Predictive Model & Trust Degradation.

Coverage:
  - extract_window_features: feature generation & shape
  - XGBoostFailureClassifier: fit, predict_proba, predict
  - LSTMFailureClassifier: fit, MC-Dropout epistemic uncertainty estimation
  - DownstreamPredictor:
      - fit & multi-model window evaluation
      - evaluate metrics (F1, AUROC, Brier score)
      - compare_trust_impact (Clean vs Corrupted vs Reconstructed)
      - Model persistence (save / load checkpoint roundtrip)
"""

import os
import tempfile
import numpy as np
import pytest
import torch
from scripts.run_phase8_downstream import select_evaluation_split

from app.ml.prediction import (
    DownstreamPredictor,
    PredictionOutput,
    XGBoostFailureClassifier,
    LSTMFailureClassifier,
    extract_window_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_labeled_dataset():
    """Generates synthetic sensor dataset (N=100, W=20, S=6) with binary labels."""
    rng = np.random.default_rng(42)
    N, W, S = 100, 20, 6

    # Normal windows (y=0)
    X_normal = rng.normal(loc=0.5, scale=0.05, size=(80, W, S)).astype(np.float32)
    y_normal = np.zeros(80, dtype=int)

    # Failure windows (y=1) with elevated vibration & temperature
    X_failure = rng.normal(loc=0.8, scale=0.15, size=(20, W, S)).astype(np.float32)
    y_failure = np.ones(20, dtype=int)

    X = np.concatenate([X_normal, X_failure], axis=0)
    y = np.concatenate([y_normal, y_failure], axis=0)

    # Shuffle
    indices = rng.permutation(N)
    return X[indices], y[indices]


# ===========================================================================
# 1. Feature Extraction Tests
# ===========================================================================
class TestFeatureExtraction:
    def test_feature_shape(self):
        X = np.random.randn(15, 20, 6)
        feats = extract_window_features(X)
        assert feats.shape == (15, 6 * 7)

    def test_handles_nan_gracefully(self):
        X = np.random.randn(10, 20, 6)
        X[0, 5:10, 0] = np.nan
        feats = extract_window_features(X)
        assert not np.isnan(feats).any()


# ===========================================================================
# 2. XGBoost Failure Classifier Tests
# ===========================================================================
class TestXGBoostClassifier:
    def test_fit_and_predict(self, synthetic_labeled_dataset):
        X, y = synthetic_labeled_dataset
        clf = XGBoostFailureClassifier(n_estimators=20, max_depth=3)
        clf.fit(X, y)

        assert clf._is_fitted is True

        probs = clf.predict_proba(X[:10])
        assert probs.shape == (10,)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

        preds = clf.predict(X[:10])
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_before_fit_raises(self):
        clf = XGBoostFailureClassifier()
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict_proba(np.zeros((5, 10, 6)))


# ===========================================================================
# 3. LSTM Failure Classifier Tests
# ===========================================================================
class TestLSTMClassifier:
    def test_fit_and_mc_dropout(self, synthetic_labeled_dataset):
        X, y = synthetic_labeled_dataset
        clf = LSTMFailureClassifier(hidden_dim=16, num_layers=1, lr=0.01, device="cpu")
        clf.fit(X, y, epochs=5, batch_size=32)

        assert clf._is_fitted is True

        probs, uncertainties = clf.predict_proba_with_uncertainty(X[:10], n_mc_samples=8)
        assert probs.shape == (10,)
        assert uncertainties.shape == (10,)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()
        assert (uncertainties >= 0.0).all()


# ===========================================================================
# 4. DownstreamPredictor Engine Tests
# ===========================================================================
class TestDownstreamPredictor:
    def test_metrics_use_binary_positive_class(self, synthetic_labeled_dataset):
        X, y = synthetic_labeled_dataset
        predictor = DownstreamPredictor(device="cpu")
        predictor.fit(X, y, lstm_epochs=1)

        metrics = predictor.evaluate(y, X, use_model="xgboost")

        assert metrics["AUROC"] > 0.5
        assert metrics["Precision"] >= 0.0
        assert metrics["Recall"] >= 0.0
        assert metrics["F1"] >= 0.0

    def test_evaluation_split_contains_both_classes(self):
        X_val = np.zeros((2, 4, 2), dtype=np.float32)
        y_val = np.array([0, 1], dtype=np.int8)
        X_test = np.ones((2, 4, 2), dtype=np.float32)
        y_test = np.zeros(2, dtype=np.int8)

        X_eval, y_eval, source = select_evaluation_split(X_val, y_val, X_test, y_test)

        assert source == "X_val+X_test"
        assert X_eval.shape[0] == 4
        assert np.array_equal(np.unique(y_eval), np.array([0, 1]))

    @pytest.fixture
    def fitted_predictor(self, synthetic_labeled_dataset):
        X, y = synthetic_labeled_dataset
        predictor = DownstreamPredictor(device="cpu")
        predictor.fit(X, y, lstm_epochs=5)
        return predictor

    def test_predict_window_single(self, fitted_predictor, synthetic_labeled_dataset):
        X, _ = synthetic_labeled_dataset
        res_lstm = fitted_predictor.predict_window(X[0], use_model="lstm")
        assert isinstance(res_lstm, PredictionOutput)
        assert 0.0 <= res_lstm.failure_probability <= 1.0
        assert res_lstm.model_name == "lstm"

        res_xgb = fitted_predictor.predict_window(X[0], use_model="xgboost")
        assert isinstance(res_xgb, PredictionOutput)
        assert 0.0 <= res_xgb.failure_probability <= 1.0
        assert res_xgb.model_name == "xgboost"

    def test_evaluate_metrics(self, fitted_predictor, synthetic_labeled_dataset):
        X, y = synthetic_labeled_dataset
        metrics = fitted_predictor.evaluate(y, X, use_model="lstm")
        assert "F1" in metrics
        assert "AUROC" in metrics
        assert "Brier Score" in metrics
        assert 0.0 <= metrics["AUROC"] <= 1.0

    def test_compare_trust_impact(self, fitted_predictor, synthetic_labeled_dataset):
        X_clean, y = synthetic_labeled_dataset

        # Create corrupted version (all sensor 0 missing/extreme)
        X_corrupted = np.copy(X_clean)
        X_corrupted[:, :, 0] = 0.0

        # Create reconstructed version
        X_reconstructed = np.copy(X_clean)

        comparison = fitted_predictor.compare_trust_impact(
            y_true=y,
            X_clean=X_clean,
            X_corrupted=X_corrupted,
            X_reconstructed=X_reconstructed,
        )

        assert "1. Clean Ground-Truth" in comparison
        assert "2. Corrupted (Silent Degradation)" in comparison
        assert "3. Reconstructed (Trust Restored)" in comparison

    def test_save_and_load_roundtrip(self, fitted_predictor, synthetic_labeled_dataset):
        X, y = synthetic_labeled_dataset
        with tempfile.TemporaryDirectory() as tmpdir:
            fitted_predictor.save(tmpdir)

            loaded = DownstreamPredictor.load(tmpdir, device="cpu")
            assert loaded._is_fitted is True

            res = loaded.predict_window(X[0], use_model="lstm")
            assert 0.0 <= res.failure_probability <= 1.0
