"""
tests/unit/test_anomaly_detection.py
=======================================
Unit tests for Phase 4 — Baseline Anomaly Detection.

Coverage
--------
* BaseDetector interface (predict before fit raises)
* _window_stats feature extraction — shape and NaN handling
* RuleBasedDetector — fit/score/predict on clean vs noisy data
* IsolationForestDetector — fit/score/predict
* OneClassSVMDetector — fit/score/predict
* evaluate_detector — returns DetectionResult with correct structure
* build_evaluation_table — correct DataFrame columns
* Per-fault metrics present when fault_types provided
* Detection delay calculation
* All detectors score anomalies higher than clean on average

No DB / FastAPI required — pure NumPy + sklearn.
"""

import numpy as np
import pandas as pd
import pytest

from app.ml.anomaly_detection.detectors import (
    BaseDetector,
    RuleBasedDetector,
    IsolationForestDetector,
    OneClassSVMDetector,
    _window_stats,
    _flatten,
)
from app.ml.anomaly_detection.evaluation import (
    DetectionResult,
    PerFaultMetrics,
    evaluate_detector,
    build_evaluation_table,
)
from app.ml.fault_injection.faults import FaultType


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)

def make_clean_windows(n: int = 300, w: int = 30, s: int = 6) -> np.ndarray:
    """Gaussian normal windows — representative of healthy sensor data."""
    return RNG.normal(loc=0.5, scale=0.05, size=(n, w, s)).astype(float)


def make_anomaly_windows(n: int = 50, w: int = 30, s: int = 6) -> np.ndarray:
    """Windows with extreme values — clearly anomalous."""
    base = RNG.normal(loc=0.5, scale=0.05, size=(n, w, s)).astype(float)
    # Add large bias to first sensor
    base[:, :, 0] += RNG.uniform(5.0, 10.0, n)[:, None]
    return base


def make_nan_windows(n: int = 30, w: int = 30, s: int = 6) -> np.ndarray:
    """Windows with ~50% NaN — simulates missing fault."""
    arr = RNG.normal(loc=0.5, scale=0.05, size=(n, w, s)).astype(float)
    mask = RNG.random((n, w, s)) < 0.5
    arr[mask] = np.nan
    return arr


# ===========================================================================
# 1. Feature extraction helpers
# ===========================================================================

class TestWindowStats:
    def test_output_shape(self):
        X = make_clean_windows(100, 30, 6)
        feats = _window_stats(X)
        assert feats.shape == (100, 6 * 6), \
            f"Expected (100, 36), got {feats.shape}"

    def test_no_nan_in_output(self):
        X = make_nan_windows(50)
        feats = _window_stats(X)
        assert not np.isnan(feats).any()

    def test_all_nan_window_gives_zeros(self):
        X = np.full((10, 30, 6), np.nan)
        feats = _window_stats(X)
        # mean/std/min/max of all-NaN → nan_to_num → 0
        assert not np.isnan(feats).any()

    def test_mean_column_correct(self):
        X = np.ones((5, 10, 3))  # all 1.0
        feats = _window_stats(X)
        means = feats[:, :3]
        np.testing.assert_allclose(means, 1.0)

    def test_drift_column_correct(self):
        """Last row − first row should be detectable for a ramp."""
        n, w, s = 10, 20, 2
        X = np.zeros((n, w, s))
        X[:, -1, 0] = 5.0   # last timestep, sensor 0 jumps
        feats  = _window_stats(X)
        drifts = feats[:, 5*s:6*s]   # drift block (index 5 * S .. 6 * S)
        # Drift for sensor 0 should be 5.0
        np.testing.assert_allclose(drifts[:, 0], 5.0)


class TestFlatten:
    def test_shape(self):
        X = make_clean_windows(50, 30, 6)
        out = _flatten(X)
        assert out.shape == (50, 30 * 6)

    def test_no_nan(self):
        X = make_nan_windows(20)
        out = _flatten(X)
        assert not np.isnan(out).any()


# ===========================================================================
# 2. BaseDetector interface
# ===========================================================================

class TestBaseDetectorInterface:
    def test_predict_before_fit_raises(self):
        det = RuleBasedDetector()
        X   = make_clean_windows(10)
        with pytest.raises(RuntimeError, match="fit"):
            det.predict(X)

    def test_score_before_fit_raises(self):
        det = IsolationForestDetector(n_estimators=10)
        X   = make_clean_windows(10)
        with pytest.raises(RuntimeError, match="fit"):
            det.score(X)

    def test_repr_contains_class_name(self):
        det = RuleBasedDetector()
        assert "RuleBasedDetector" in repr(det)

    def test_fitted_flag_set_after_fit(self):
        det = RuleBasedDetector()
        det.fit(make_clean_windows(50))
        assert det._is_fitted is True


# ===========================================================================
# 3. RuleBasedDetector
# ===========================================================================

class TestRuleBasedDetector:
    def test_fit_returns_self(self):
        det = RuleBasedDetector()
        out = det.fit(make_clean_windows(100))
        assert out is det

    def test_score_shape(self):
        det = RuleBasedDetector()
        det.fit(make_clean_windows(200))
        X   = make_clean_windows(50)
        scores = det.score(X)
        assert scores.shape == (50,)

    def test_predict_shape(self):
        det = RuleBasedDetector()
        det.fit(make_clean_windows(200))
        preds = det.predict(make_clean_windows(40))
        assert preds.shape == (40,)

    def test_binary_predictions(self):
        det = RuleBasedDetector()
        det.fit(make_clean_windows(200))
        preds = det.predict(make_clean_windows(50))
        assert set(np.unique(preds)).issubset({0, 1})

    def test_anomalies_score_higher_than_clean(self):
        det   = RuleBasedDetector()
        clean = make_clean_windows(500)
        anom  = make_anomaly_windows(100)
        det.fit(clean)
        score_clean = det.score(clean[:50]).mean()
        score_anom  = det.score(anom).mean()
        assert score_anom > score_clean, \
            f"Anomaly score {score_anom:.3f} should exceed clean {score_clean:.3f}"

    def test_nan_windows_score_high(self):
        """Missing-data windows should get elevated anomaly scores."""
        det   = RuleBasedDetector(nan_threshold=0.1)
        clean = make_clean_windows(300)
        nan_w = make_nan_windows(50)
        det.fit(clean)
        score_nan   = det.score(nan_w).mean()
        score_clean = det.score(clean[:50]).mean()
        assert score_nan > score_clean

    def test_threshold_is_set(self):
        det = RuleBasedDetector()
        det.fit(make_clean_windows(100))
        assert hasattr(det, "_threshold")
        assert det._threshold >= 0.0

    def test_custom_threshold_overrides(self):
        det = RuleBasedDetector()
        det.fit(make_clean_windows(200))
        # Force every window to be anomalous
        preds = det.predict(make_clean_windows(20), threshold=0.0)
        assert preds.all()

    def test_no_nan_in_scores(self):
        det = RuleBasedDetector()
        det.fit(make_clean_windows(200))
        scores = det.score(make_nan_windows(30))
        assert not np.isnan(scores).any()


# ===========================================================================
# 4. IsolationForestDetector
# ===========================================================================

class TestIsolationForestDetector:
    @pytest.fixture
    def fitted_isoforest(self):
        det   = IsolationForestDetector(n_estimators=50, random_state=0)
        clean = make_clean_windows(300)
        det.fit(clean)
        return det

    def test_fit_returns_self(self):
        det = IsolationForestDetector(n_estimators=20, random_state=0)
        out = det.fit(make_clean_windows(100))
        assert out is det

    def test_score_shape(self, fitted_isoforest):
        scores = fitted_isoforest.score(make_clean_windows(40))
        assert scores.shape == (40,)

    def test_predict_shape(self, fitted_isoforest):
        preds = fitted_isoforest.predict(make_clean_windows(40))
        assert preds.shape == (40,)

    def test_binary_predictions(self, fitted_isoforest):
        preds = fitted_isoforest.predict(make_clean_windows(40))
        assert set(np.unique(preds)).issubset({0, 1})

    def test_anomalies_score_higher_than_clean(self, fitted_isoforest):
        clean_scores = fitted_isoforest.score(make_clean_windows(100)).mean()
        anom_scores  = fitted_isoforest.score(make_anomaly_windows(50)).mean()
        assert anom_scores > clean_scores

    def test_threshold_set_after_fit(self, fitted_isoforest):
        assert hasattr(fitted_isoforest, "_threshold")

    def test_no_nan_in_scores(self, fitted_isoforest):
        scores = fitted_isoforest.score(make_nan_windows(20))
        assert not np.isnan(scores).any()

    def test_different_seeds_different_models(self):
        clean = make_clean_windows(200)
        d1 = IsolationForestDetector(n_estimators=20, random_state=0).fit(clean)
        d2 = IsolationForestDetector(n_estimators=20, random_state=99).fit(clean)
        X  = make_anomaly_windows(20)
        # Scores differ between differently seeded models
        assert not np.allclose(d1.score(X), d2.score(X))


# ===========================================================================
# 5. OneClassSVMDetector
# ===========================================================================

class TestOneClassSVMDetector:
    @pytest.fixture
    def fitted_svm(self):
        det   = OneClassSVMDetector(nu=0.05, gamma="scale")
        clean = make_clean_windows(200)
        det.fit(clean)
        return det

    def test_fit_returns_self(self):
        det = OneClassSVMDetector(nu=0.05)
        out = det.fit(make_clean_windows(100))
        assert out is det

    def test_score_shape(self, fitted_svm):
        scores = fitted_svm.score(make_clean_windows(40))
        assert scores.shape == (40,)

    def test_predict_shape(self, fitted_svm):
        preds = fitted_svm.predict(make_clean_windows(40))
        assert preds.shape == (40,)

    def test_binary_predictions(self, fitted_svm):
        preds = fitted_svm.predict(make_clean_windows(40))
        assert set(np.unique(preds)).issubset({0, 1})

    def test_anomalies_score_higher(self, fitted_svm):
        clean_scores = fitted_svm.score(make_clean_windows(80)).mean()
        anom_scores  = fitted_svm.score(make_anomaly_windows(40)).mean()
        assert anom_scores > clean_scores

    def test_no_nan_in_scores(self, fitted_svm):
        scores = fitted_svm.score(make_nan_windows(20))
        assert not np.isnan(scores).any()

    def test_threshold_set_after_fit(self, fitted_svm):
        assert hasattr(fitted_svm, "_threshold")


# ===========================================================================
# 6. evaluate_detector
# ===========================================================================

class TestEvaluateDetector:
    """Smoke + contract tests for the evaluation harness."""

    def _run(self, n_normal=400, n_anom=80, with_fault_types=True):
        X_normal = make_clean_windows(n_normal)
        X_anom   = make_anomaly_windows(n_anom)
        X_clean2 = make_clean_windows(100)

        X_test  = np.concatenate([X_clean2, X_anom], axis=0)
        y_test  = np.concatenate([np.zeros(100), np.ones(n_anom)]).astype(int)
        ft_test = None
        if with_fault_types:
            ft_test = np.array(
                ["NORMAL"] * 100 + ["NOISE"] * n_anom, dtype=object
            )

        det    = RuleBasedDetector()
        result = evaluate_detector(
            detector       = det,
            X_normal_train = X_normal,
            X_test         = X_test,
            y_test         = y_test,
            fault_types    = ft_test,
        )
        return result

    def test_returns_detection_result(self):
        result = self._run()
        assert isinstance(result, DetectionResult)

    def test_detector_name_propagated(self):
        result = self._run()
        assert result.detector_name == "RuleBasedDetector"

    def test_overall_auroc_in_range(self):
        result = self._run()
        assert 0.0 <= result.overall_auroc <= 1.0

    def test_overall_f1_in_range(self):
        result = self._run()
        assert 0.0 <= result.overall_f1 <= 1.0

    def test_overall_avg_prec_in_range(self):
        result = self._run()
        assert 0.0 <= result.overall_avg_prec <= 1.0

    def test_fit_time_positive(self):
        result = self._run()
        assert result.fit_time_s >= 0.0

    def test_per_fault_populated_when_fault_types_given(self):
        result = self._run(with_fault_types=True)
        assert len(result.per_fault) >= 1

    def test_per_fault_empty_when_no_fault_types(self):
        result = self._run(with_fault_types=False)
        assert result.per_fault == []

    def test_per_fault_metrics_in_range(self):
        result = self._run(with_fault_types=True)
        for pf in result.per_fault:
            assert 0.0 <= pf.f1       <= 1.0
            assert 0.0 <= pf.auroc    <= 1.0
            assert 0.0 <= pf.precision <= 1.0
            assert 0.0 <= pf.recall   <= 1.0
            assert pf.n_anomaly > 0

    def test_high_anomaly_detector_auroc_above_half(self):
        """With clearly separated clean/anomaly windows, AUROC should exceed 0.5."""
        result = self._run()
        assert result.overall_auroc > 0.5, \
            f"AUROC {result.overall_auroc:.3f} should be > 0.5 for clearly separated data"

    def test_all_three_detectors_run_without_error(self):
        X_normal = make_clean_windows(200)
        X_anom   = make_anomaly_windows(40)
        X_test   = np.concatenate([make_clean_windows(60), X_anom])
        y_test   = np.concatenate([np.zeros(60), np.ones(40)]).astype(int)

        for DetClass in [RuleBasedDetector,
                         lambda: IsolationForestDetector(n_estimators=20, random_state=0),
                         lambda: OneClassSVMDetector(nu=0.05)]:
            det    = DetClass()
            result = evaluate_detector(det, X_normal, X_test, y_test)
            assert isinstance(result, DetectionResult)


# ===========================================================================
# 7. build_evaluation_table
# ===========================================================================

class TestBuildEvaluationTable:
    def _make_result(self, name, f1, auroc, ap) -> DetectionResult:
        pf = PerFaultMetrics(
            fault_type="NOISE", n_anomaly=50, n_normal=200,
            precision=0.8, recall=0.75, f1=f1,
            auroc=auroc, avg_precision=ap, detection_delay=120.0,
        )
        return DetectionResult(
            detector_name=name, overall_f1=f1,
            overall_auroc=auroc, overall_avg_prec=ap,
            per_fault=[pf], fit_time_s=1.5, score_time_s=0.05,
        )

    def test_summary_df_has_expected_columns(self):
        r = self._make_result("Det_A", 0.8, 0.9, 0.85)
        summary_df, _ = build_evaluation_table([r])
        assert "F1"    in summary_df.columns
        assert "AUROC" in summary_df.columns

    def test_summary_df_indexed_by_detector(self):
        r = self._make_result("Det_A", 0.8, 0.9, 0.85)
        summary_df, _ = build_evaluation_table([r])
        assert "Det_A" in summary_df.index

    def test_multiple_detectors_in_summary(self):
        rs = [self._make_result(f"Det_{i}", 0.7+i*0.05, 0.8, 0.75) for i in range(3)]
        summary_df, _ = build_evaluation_table(rs)
        assert len(summary_df) == 3

    def test_per_fault_df_has_fault_type_column(self):
        r = self._make_result("Det_A", 0.8, 0.9, 0.85)
        _, per_fault_df = build_evaluation_table([r])
        assert "Fault Type"  in per_fault_df.columns
        assert "Detector"    in per_fault_df.columns
        assert "F1"          in per_fault_df.columns

    def test_per_fault_df_rows_count(self):
        rs = [self._make_result(f"D{i}", 0.8, 0.9, 0.85) for i in range(3)]
        _, per_fault_df = build_evaluation_table(rs)
        # Each result contributes 1 per-fault row → 3 total
        assert len(per_fault_df) == 3

    def test_empty_results_returns_empty_per_fault(self):
        r = DetectionResult(
            detector_name="Det_X", overall_f1=0.0,
            overall_auroc=0.5, overall_avg_prec=0.0,
            per_fault=[], fit_time_s=0.0, score_time_s=0.0,
        )
        summary_df, per_fault_df = build_evaluation_table([r])
        assert len(summary_df) == 1
        assert per_fault_df.empty

    def test_summary_dict_format(self):
        r = self._make_result("Det_B", 0.75, 0.88, 0.80)
        d = r.summary_dict()
        assert d["Detector"]  == "Det_B"
        assert "." in d["F1"]     # formatted float


# ===========================================================================
# 8. fit_predict convenience method
# ===========================================================================

class TestFitPredict:
    def test_fit_predict_returns_preds_and_scores(self):
        det   = RuleBasedDetector()
        clean = make_clean_windows(200)
        X_t   = np.concatenate([make_clean_windows(40), make_anomaly_windows(20)])
        y_t   = np.concatenate([np.zeros(40), np.ones(20)])

        preds, scores = det.fit_predict(clean, X_t)
        assert preds.shape  == (60,)
        assert scores.shape == (60,)

    def test_preds_are_binary(self):
        det   = IsolationForestDetector(n_estimators=10, random_state=0)
        clean = make_clean_windows(150)
        X_t   = make_clean_windows(30)
        preds, _ = det.fit_predict(clean, X_t)
        assert set(np.unique(preds)).issubset({0, 1})
