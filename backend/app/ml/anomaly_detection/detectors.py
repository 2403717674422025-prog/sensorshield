"""
ml/anomaly_detection/detectors.py
====================================
Three baseline anomaly detectors for Phase 4.

All detectors share a common interface (BaseDetector):
    fit(X_normal)               — train on normal (clean) windows
    predict(X)                  — returns binary labels (0=normal, 1=anomaly)
    score(X)                    — returns anomaly scores (higher = more anomalous)
    fit_predict(X_normal, X)    — convenience wrapper

Input convention
----------------
X : np.ndarray of shape (N, window_size, n_sensors)
    Each row is one sliding window of multi-sensor readings.

All three detectors flatten windows to 2D for feature extraction:
  (N, window_size * n_sensors)  — for IsoForest and OneClassSVM
  (N, n_sensors * n_stats)      — for Rule-based (stats per sensor per window)

Design notes
------------
* Detectors are stateless until fit(); calling predict() before fit() raises.
* No global random state — seeds are passed explicitly.
* Sklearn detectors expose contamination as a tunable parameter.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature extraction helpers (shared)
# ---------------------------------------------------------------------------

def _window_stats(X: np.ndarray) -> np.ndarray:
    """
    Compute per-sensor statistics across each window.

    For each window (window_size, n_sensors) compute:
        mean, std, min, max, range, last_minus_first
    → output shape: (N, n_sensors * 6)
    """
    # X: (N, W, S)
    mean  = np.nanmean(X, axis=1)           # (N, S)
    std   = np.nanstd(X, axis=1)            # (N, S)
    mn    = np.nanmin(X, axis=1)            # (N, S)
    mx    = np.nanmax(X, axis=1)            # (N, S)
    rng   = mx - mn                         # (N, S)
    drift = X[:, -1, :] - X[:, 0, :]       # (N, S) — end minus start

    feats = np.concatenate([mean, std, mn, mx, rng, drift], axis=1)
    # Replace any remaining NaN with 0 (NaN windows from missing fault)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats  # (N, S*6)


def _flatten(X: np.ndarray) -> np.ndarray:
    """Flatten (N, W, S) → (N, W*S), NaN → 0."""
    flat = X.reshape(X.shape[0], -1)
    return np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseDetector(ABC):
    """Abstract base class for anomaly detectors."""

    def __init__(self, name: str):
        self.name      = name
        self._is_fitted = False

    @abstractmethod
    def fit(self, X_normal: np.ndarray) -> "BaseDetector":
        """Train on normal (fault-free) windows."""

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Return anomaly score per window.
        Higher values → more anomalous.
        Shape: (N,)
        """

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Return binary predictions (1 = anomaly).

        If threshold is None, uses the fitted threshold (set during fit()).
        """
        if not self._is_fitted:
            raise RuntimeError(f"{self.name}: call fit() before predict()")
        scores = self.score(X)
        thr    = threshold if threshold is not None else self._threshold
        return (scores >= thr).astype(int)

    def fit_predict(
        self,
        X_normal: np.ndarray,
        X_test:   np.ndarray,
        threshold: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Train on normal data, score+predict on test data."""
        self.fit(X_normal)
        return self.predict(X_test, threshold), self.score(X_test)

    def _set_threshold_from_normal(
        self, scores_normal: np.ndarray, percentile: float = 99.0
    ) -> None:
        """
        Set decision threshold as the Nth percentile of normal anomaly scores.
        Anything above this on test data is flagged as anomalous.
        """
        self._threshold = float(np.percentile(scores_normal, percentile))
        logger.debug(
            "%s: threshold set to %.4f (p%g of normal scores)",
            self.name, self._threshold, percentile,
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} fitted={self._is_fitted}>"


# ---------------------------------------------------------------------------
# 1. Rule-Based Detector — statistical thresholds
# ---------------------------------------------------------------------------

class RuleBasedDetector(BaseDetector):
    """
    Per-sensor z-score and IQR based threshold detector.

    For each sliding window, compute per-sensor statistics and flag the
    window as anomalous if ANY sensor crosses its trained threshold:
      - |z-score of window mean|  > z_threshold    (bias / drift detection)
      - window std > mean_std * std_multiplier      (noise detection)
      - window range < min_range_fraction * range    (stuck detection)
      - NaN fraction > nan_threshold                (missing / sampling failure)

    The thresholds are computed from normal training windows.
    No external dependencies beyond NumPy.

    Parameters
    ----------
    z_threshold : float
        How many std deviations the window mean can be from training mean.
    std_multiplier : float
        Window std must not exceed this multiple of training std.
    nan_threshold : float
        Maximum NaN fraction per window before flagging.
    percentile : float
        Percentile of normal scores used to set the decision boundary.
    """

    def __init__(
        self,
        z_threshold:    float = 3.0,
        std_multiplier: float = 3.0,
        nan_threshold:  float = 0.2,
        percentile:     float = 99.0,
    ):
        super().__init__("RuleBasedDetector")
        self.z_threshold    = z_threshold
        self.std_multiplier = std_multiplier
        self.nan_threshold  = nan_threshold
        self.percentile     = percentile

        # Learned from training
        self._train_mean: Optional[np.ndarray] = None  # (S,)
        self._train_std:  Optional[np.ndarray] = None  # (S,)
        self._train_range: Optional[np.ndarray] = None # (S,)

    def fit(self, X_normal: np.ndarray) -> "RuleBasedDetector":
        """Learn per-sensor statistics from normal windows."""
        # X_normal: (N, W, S)
        feats = _window_stats(X_normal)                    # (N, S*6)
        S = X_normal.shape[2]

        # Column order from _window_stats: mean(S), std(S), min(S), max(S), rng(S), drift(S)
        means  = feats[:, :S]          # (N, S) — window means
        stds   = feats[:, S:2*S]       # (N, S) — window stds
        ranges = feats[:, 4*S:5*S]     # (N, S) — window ranges

        self._train_mean  = np.mean(means,  axis=0)   # (S,)
        self._train_std   = np.std(means,   axis=0)   # std of window means across training
        self._train_wstd  = np.mean(stds,   axis=0)   # mean of within-window std
        self._train_range = np.mean(ranges, axis=0)   # mean range across training

        # Guard against zero std (constant sensors)
        self._train_std   = np.where(self._train_std   < 1e-9, 1.0, self._train_std)
        self._train_wstd  = np.where(self._train_wstd  < 1e-9, 1.0, self._train_wstd)
        self._train_range = np.where(self._train_range < 1e-9, 1.0, self._train_range)

        self._is_fitted   = True

        # Set threshold from normal scores
        normal_scores = self._compute_scores(X_normal)
        self._set_threshold_from_normal(normal_scores, self.percentile)
        logger.info(
            "RuleBasedDetector fitted on %d windows, threshold=%.4f",
            len(X_normal), self._threshold,
        )
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("RuleBasedDetector: call fit() first")
        return self._compute_scores(X)

    def _compute_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Compute a composite anomaly score in [0, ∞) for each window.

        Score = max across sensors of:
            max(|z|, std_ratio, nan_fraction * 10)
        This keeps the score interpretable (z=3 ≡ threshold when z_threshold=3).
        """
        S  = X.shape[2]
        feats = _window_stats(X)              # (N, S*6)
        means  = feats[:, :S]
        stds   = feats[:, S:2*S]
        ranges = feats[:, 4*S:5*S]

        # NaN fraction per window per sensor
        nan_frac = np.mean(np.isnan(X), axis=1)  # (N, S)

        # Component scores (all shape: N, S)
        z_scores    = np.abs(means - self._train_mean) / self._train_std
        std_ratios  = stds / self._train_wstd
        nan_scores  = nan_frac * (10.0 / max(self.nan_threshold, 1e-6))

        # Composite: take max component, then max across sensors
        per_sensor  = np.maximum(z_scores, np.maximum(std_ratios, nan_scores))
        scores      = np.max(per_sensor, axis=1)   # (N,)
        return scores


# ---------------------------------------------------------------------------
# 2. Isolation Forest
# ---------------------------------------------------------------------------

class IsolationForestDetector(BaseDetector):
    """
    Anomaly detection via sklearn IsolationForest.

    Features: per-window statistical summary (mean, std, min, max, range, drift)
    per sensor → (N, n_sensors * 6) feature matrix.

    The anomaly score is the negative of sklearn's decision_function
    (so higher = more anomalous, consistent with our interface).

    Parameters
    ----------
    n_estimators : int     — number of trees
    contamination : float  — expected fraction of anomalies in training data
    random_state : int     — reproducibility seed
    percentile : float     — threshold percentile from normal scores
    """

    def __init__(
        self,
        n_estimators:  int   = 200,
        contamination: float = 0.01,
        random_state:  int   = 42,
        percentile:    float = 99.0,
    ):
        super().__init__("IsolationForestDetector")
        self.n_estimators  = n_estimators
        self.contamination = contamination
        self.random_state  = random_state
        self.percentile    = percentile
        self._scaler       = StandardScaler()
        self._model        = IsolationForest(
            n_estimators  = n_estimators,
            contamination = contamination,
            random_state  = random_state,
            n_jobs        = -1,
        )

    def fit(self, X_normal: np.ndarray) -> "IsolationForestDetector":
        feats = _window_stats(X_normal)
        feats = self._scaler.fit_transform(feats)
        self._model.fit(feats)
        self._is_fitted = True

        normal_scores = self._raw_scores(X_normal)
        self._set_threshold_from_normal(normal_scores, self.percentile)
        logger.info(
            "IsolationForestDetector fitted on %d windows (%d trees), threshold=%.4f",
            len(X_normal), self.n_estimators, self._threshold,
        )
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("IsolationForestDetector: call fit() first")
        return self._raw_scores(X)

    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        feats = _window_stats(X)
        feats = self._scaler.transform(feats)
        # decision_function: higher = more normal → negate so higher = anomalous
        return -self._model.decision_function(feats)


# ---------------------------------------------------------------------------
# 3. One-Class SVM
# ---------------------------------------------------------------------------

class OneClassSVMDetector(BaseDetector):
    """
    Anomaly detection via sklearn OneClassSVM.

    Uses RBF kernel by default. Features same as IsolationForest
    (per-window stats). Scaled with StandardScaler before fitting.

    Parameters
    ----------
    kernel : str           — SVM kernel (default 'rbf')
    nu : float             — upper bound on fraction of outliers
    gamma : str or float   — kernel coefficient
    percentile : float     — threshold percentile from normal scores
    """

    def __init__(
        self,
        kernel:     str        = "rbf",
        nu:         float      = 0.01,
        gamma:      str|float  = "scale",
        percentile: float      = 99.0,
    ):
        super().__init__("OneClassSVMDetector")
        self.kernel     = kernel
        self.nu         = nu
        self.gamma      = gamma
        self.percentile = percentile
        self._scaler    = StandardScaler()
        self._model     = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)

    def fit(self, X_normal: np.ndarray) -> "OneClassSVMDetector":
        feats = _window_stats(X_normal)
        feats = self._scaler.fit_transform(feats)
        self._model.fit(feats)
        self._is_fitted = True

        normal_scores = self._raw_scores(X_normal)
        self._set_threshold_from_normal(normal_scores, self.percentile)
        logger.info(
            "OneClassSVMDetector fitted on %d windows (nu=%.3f), threshold=%.4f",
            len(X_normal), self.nu, self._threshold,
        )
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("OneClassSVMDetector: call fit() first")
        return self._raw_scores(X)

    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        feats = _window_stats(X)
        feats = self._scaler.transform(feats)
        # decision_function: positive = normal, negative = anomaly → negate
        return -self._model.decision_function(feats)
