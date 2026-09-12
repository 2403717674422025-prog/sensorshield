"""
tests/unit/test_reliability_engine.py
======================================
Unit tests for Phase 6 — Sensor Reliability & Health Scoring Engine.

Coverage:
  - Metric functions: drift, noise, consistency, missing, stuck
  - SensorReliabilityEngine:
      - fit_baselines
      - evaluate_sensor (HEALTHY, DRIFTING, NOISY, STUCK, MISSING, INTERMITTENT, FAILED)
      - Weighted composite health score (0-100 range)
      - Diagnostic explainability reason generation
"""

from datetime import datetime
import numpy as np
import pytest

from app.models.sensor import SensorStatus
from app.ml.reliability.metrics import (
    compute_drift_score,
    compute_noise_score,
    compute_consistency_score,
    compute_missing_data_score,
    is_stuck_signal,
)
from app.ml.reliability.engine import (
    SensorReliabilityEngine,
    SensorBaseline,
    SensorHealthEvaluation,
)


# ---------------------------------------------------------------------------
# 1. Metric Unit Tests
# ---------------------------------------------------------------------------
class TestReliabilityMetrics:
    def test_drift_score_healthy(self):
        # Stationary signal
        rng = np.random.default_rng(42)
        signal = rng.normal(50.0, 1.0, 30)
        score, slope = compute_drift_score(signal, baseline_std=1.0)
        assert score > 80.0
        assert abs(slope) < 0.2

    def test_drift_score_drifting(self):
        # Monotonically increasing ramp
        t = np.linspace(0, 10, 30)
        signal = 50.0 + 2.0 * t
        score, slope = compute_drift_score(signal, baseline_std=1.0)
        assert score < 40.0
        assert slope > 0.0

    def test_noise_score_healthy(self):
        rng = np.random.default_rng(42)
        signal = rng.normal(50.0, 0.5, 30)
        base_diff_std = float(np.std(np.diff(signal)))
        score = compute_noise_score(signal, baseline_diff_std=base_diff_std)
        assert score == 100.0

    def test_noise_score_high_noise(self):
        rng = np.random.default_rng(42)
        signal = rng.normal(50.0, 5.0, 30)
        score = compute_noise_score(signal, baseline_diff_std=0.5)
        assert score < 30.0

    def test_consistency_score_within_bounds(self):
        score = compute_consistency_score(sensor_val=50.0, valid_range=(0.0, 100.0))
        assert score == 100.0

    def test_consistency_score_out_of_bounds(self):
        score = compute_consistency_score(sensor_val=150.0, valid_range=(0.0, 100.0))
        assert score < 50.0

    def test_missing_data_score_clean(self):
        signal = np.ones(30)
        score, rate, burst = compute_missing_data_score(signal)
        assert score == 100.0
        assert rate == 0.0
        assert burst == 0

    def test_missing_data_score_bursty(self):
        signal = np.ones(30)
        signal[10:20] = np.nan  # 10 consecutive NaNs out of 30
        score, rate, burst = compute_missing_data_score(signal)
        assert rate == pytest.approx(10 / 30)
        assert burst == 10
        assert score < 60.0

    def test_is_stuck_signal_frozen(self):
        signal = np.full(30, 42.0)
        stuck, w_std = is_stuck_signal(signal, baseline_std=2.0)
        assert stuck is True
        assert w_std == 0.0

    def test_is_stuck_signal_normal(self):
        rng = np.random.default_rng(42)
        signal = rng.normal(50.0, 2.0, 30)
        stuck, w_std = is_stuck_signal(signal, baseline_std=2.0)
        assert stuck is False
        assert w_std > 0.5


# ---------------------------------------------------------------------------
# 2. SensorReliabilityEngine Tests
# ---------------------------------------------------------------------------
class TestSensorReliabilityEngine:
    @pytest.fixture
    def engine(self):
        rng = np.random.default_rng(42)
        clean_data = rng.normal(loc=50.0, scale=2.0, size=(500, 30, 3))
        engine = SensorReliabilityEngine()
        engine.fit_baselines(
            df_normal=clean_data,
            sensor_ids=["TEMP_01", "PRESSURE_01", "VIBRATION_01"],
            valid_ranges={"TEMP_01": (0.0, 100.0), "PRESSURE_01": (0.0, 10.0)},
        )
        return engine

    def test_evaluate_healthy_sensor(self, engine):
        rng = np.random.default_rng(10)
        healthy_window = rng.normal(50.0, 2.0, 30)

        eval_res = engine.evaluate_sensor(
            sensor_id="TEMP_01",
            signal_window=healthy_window,
            reconstruction_error=0.005,
        )

        assert isinstance(eval_res, SensorHealthEvaluation)
        assert eval_res.health_score >= 80.0
        assert eval_res.status == SensorStatus.HEALTHY
        assert "nominal" in eval_res.reason.lower() or "health" in eval_res.reason.lower()

    def test_evaluate_drifting_sensor(self, engine):
        drift_window = 50.0 + np.linspace(0, 15.0, 30)

        eval_res = engine.evaluate_sensor(
            sensor_id="TEMP_01",
            signal_window=drift_window,
            reconstruction_error=0.01,
        )

        assert eval_res.status == SensorStatus.DRIFTING
        assert eval_res.drift_score < 50.0
        assert "drift" in eval_res.reason.lower()

    def test_evaluate_noisy_sensor(self, engine):
        rng = np.random.default_rng(99)
        noisy_window = rng.normal(50.0, 12.0, 30)

        eval_res = engine.evaluate_sensor(
            sensor_id="TEMP_01",
            signal_window=noisy_window,
            reconstruction_error=0.01,
        )

        assert eval_res.status == SensorStatus.NOISY
        assert eval_res.noise_score < 40.0
        assert "noise" in eval_res.reason.lower()

    def test_evaluate_stuck_sensor(self, engine):
        stuck_window = np.full(30, 52.3)

        eval_res = engine.evaluate_sensor(
            sensor_id="TEMP_01",
            signal_window=stuck_window,
            reconstruction_error=0.02,
        )

        assert eval_res.status == SensorStatus.STUCK
        assert eval_res.health_score <= 25.0
        assert "stuck" in eval_res.reason.lower()

    def test_evaluate_missing_sensor(self, engine):
        missing_window = np.full(30, np.nan)

        eval_res = engine.evaluate_sensor(
            sensor_id="TEMP_01",
            signal_window=missing_window,
        )

        assert eval_res.status == SensorStatus.MISSING
        assert eval_res.health_score <= 20.0
        assert "missing" in eval_res.reason.lower()

    def test_evaluate_intermittent_sensor(self, engine):
        intermittent_window = np.ones(30) * 50.0
        intermittent_window[5:12] = np.nan
        intermittent_window[20:25] = np.nan

        eval_res = engine.evaluate_sensor(
            sensor_id="TEMP_01",
            signal_window=intermittent_window,
        )

        assert eval_res.status == SensorStatus.INTERMITTENT
        assert eval_res.missing_data_score < 70.0

    def test_weights_normalization(self):
        engine = SensorReliabilityEngine(
            weights={"drift": 2.0, "noise": 2.0, "consistency": 2.0, "missing": 2.0, "anomaly": 2.0}
        )
        assert pytest.approx(sum(engine.weights.values())) == 1.0
        assert pytest.approx(engine.weights["drift"]) == 0.2

    def test_to_dict_export(self, engine):
        window = np.ones(30) * 50.0
        eval_res = engine.evaluate_sensor("TEMP_01", window)
        d = eval_res.to_dict()
        assert "health_score" in d
        assert "status" in d
        assert "drift_score" in d
        assert d["sensor_id"] == "TEMP_01"
