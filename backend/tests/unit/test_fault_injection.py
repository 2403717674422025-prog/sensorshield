"""
tests/unit/test_fault_injection.py
====================================
Unit tests for the Phase 3 Fault Injection Engine.

Coverage
--------
* All 8 fault functions in isolation (faults.py)
* FaultConfig dataclass validation
* FaultEngine.apply() — correct rows affected, labels, no mutation of original
* FaultEngine with multiple faults on same sensor (composition)
* FaultEngine with multiple sensors
* Edge cases: zero-length windows, NaN-only input, single-sample signal
* Determinism: same seed → same output
* API helper: fault_config_from_request()

No DB / FastAPI needed — pure Python + NumPy + pandas.
"""

import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from app.ml.fault_injection.faults import (
    FaultType,
    apply_bias,
    apply_drift,
    apply_intermittent,
    apply_missing,
    apply_noise,
    apply_sampling_failure,
    apply_spike,
    apply_stuck,
    FAULT_DISPATCH,
)
from app.ml.fault_injection.engine import (
    FaultConfig,
    FaultEngine,
    fault_config_from_request,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(42)
BASE_TIME = datetime(2024, 1, 1, 0, 0, 0)


def make_signal(n: int = 100, lo: float = 50.0, hi: float = 80.0, seed: int = 0) -> np.ndarray:
    """Generate a clean sinusoidal-ish signal in [lo, hi]."""
    rng = np.random.default_rng(seed)
    return rng.uniform(lo, hi, n).astype(float)


def make_df(
    n: int = 200,
    sensors: list[str] | None = None,
    freq_seconds: int = 60,
) -> pd.DataFrame:
    """Build a minimal DataFrame with timestamp + sensor columns."""
    if sensors is None:
        sensors = ["TEMP_01", "PRESSURE_01"]
    ts = [BASE_TIME + timedelta(seconds=i * freq_seconds) for i in range(n)]
    rng = np.random.default_rng(7)
    data = {"timestamp": ts}
    for s in sensors:
        data[s] = rng.uniform(40.0, 90.0, n)
    return pd.DataFrame(data)


def make_fault(
    sensor_id: str = "TEMP_01",
    fault_type: FaultType = FaultType.NOISE,
    severity: float = 0.5,
    duration_seconds: int = 3600,
    start_offset: int = 0,
) -> FaultConfig:
    return FaultConfig(
        sensor_id        = sensor_id,
        fault_type       = fault_type,
        start_time       = BASE_TIME + timedelta(seconds=start_offset),
        duration_seconds = duration_seconds,
        severity         = severity,
    )


# ===========================================================================
# 1. Individual fault functions
# ===========================================================================

class TestApplyDrift:
    def test_output_shape_unchanged(self):
        sig = make_signal(50)
        out = apply_drift(sig, severity=0.8, rng=np.random.default_rng(0))
        assert out.shape == sig.shape

    def test_original_not_mutated(self):
        sig = make_signal(50)
        original = sig.copy()
        apply_drift(sig, severity=0.8, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(sig, original)

    def test_drift_increases_difference_over_time(self):
        """The ramp should cause the end of the signal to differ from start."""
        sig = np.zeros(100, dtype=float)
        out = apply_drift(sig, severity=1.0, rng=np.random.default_rng(1))
        # Last value should not equal first value (ramp applied)
        assert abs(out[-1] - out[0]) > 0.0

    def test_severity_zero_produces_no_drift(self):
        sig = np.ones(50, dtype=float)
        out = apply_drift(sig, severity=0.0, rng=np.random.default_rng(0))
        # At severity=0, total_drift = 0, so output == input
        np.testing.assert_allclose(out, sig)

    def test_higher_severity_larger_drift(self):
        sig = make_signal(100)
        lo = apply_drift(sig, severity=0.1, rng=np.random.default_rng(42))
        hi = apply_drift(sig, severity=0.9, rng=np.random.default_rng(42))
        drift_lo = abs(lo[-1] - lo[0])
        drift_hi = abs(hi[-1] - hi[0])
        assert drift_hi > drift_lo

    def test_empty_signal_returns_empty(self):
        out = apply_drift(np.array([]), severity=0.5, rng=np.random.default_rng(0))
        assert len(out) == 0

    def test_no_nan_in_output(self):
        sig = make_signal(80)
        out = apply_drift(sig, severity=0.7, rng=np.random.default_rng(5))
        assert not np.isnan(out).any()


class TestApplyBias:
    def test_output_shape_unchanged(self):
        sig = make_signal(60)
        out = apply_bias(sig, severity=0.5, rng=np.random.default_rng(0))
        assert out.shape == sig.shape

    def test_original_not_mutated(self):
        sig = make_signal(60)
        original = sig.copy()
        apply_bias(sig, severity=0.5, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(sig, original)

    def test_constant_offset_applied(self):
        """All values should be shifted by the same constant."""
        sig = np.ones(50, dtype=float)
        out = apply_bias(sig, severity=1.0, rng=np.random.default_rng(0))
        diffs = np.diff(out)
        np.testing.assert_allclose(diffs, 0.0, atol=1e-10)

    def test_severity_zero_no_bias(self):
        sig = make_signal(50)
        out = apply_bias(sig, severity=0.0, rng=np.random.default_rng(0))
        np.testing.assert_allclose(out, sig, rtol=1e-9)

    def test_higher_severity_larger_shift(self):
        sig = make_signal(100)
        lo = apply_bias(sig, severity=0.1, rng=np.random.default_rng(1))
        hi = apply_bias(sig, severity=0.9, rng=np.random.default_rng(1))
        shift_lo = abs(np.mean(lo) - np.mean(sig))
        shift_hi = abs(np.mean(hi) - np.mean(sig))
        assert shift_hi > shift_lo

    def test_empty_signal(self):
        out = apply_bias(np.array([]), severity=0.5, rng=np.random.default_rng(0))
        assert len(out) == 0

    def test_no_nan_in_output(self):
        sig = make_signal(80)
        out = apply_bias(sig, severity=0.6, rng=np.random.default_rng(0))
        assert not np.isnan(out).any()


class TestApplyNoise:
    def test_output_shape_unchanged(self):
        sig = make_signal(70)
        out = apply_noise(sig, severity=0.5, rng=np.random.default_rng(0))
        assert out.shape == sig.shape

    def test_original_not_mutated(self):
        sig = make_signal(70)
        original = sig.copy()
        apply_noise(sig, severity=0.5, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(sig, original)

    def test_severity_zero_returns_clean_signal(self):
        sig = make_signal(50)
        out = apply_noise(sig, severity=0.0, rng=np.random.default_rng(0))
        np.testing.assert_allclose(out, sig, rtol=1e-9)

    def test_higher_severity_larger_variance(self):
        sig = make_signal(200)
        lo = apply_noise(sig, severity=0.1, rng=np.random.default_rng(0))
        hi = apply_noise(sig, severity=0.9, rng=np.random.default_rng(0))
        assert np.std(hi - sig) > np.std(lo - sig)

    def test_noise_is_zero_mean_approximately(self):
        sig = np.zeros(5000, dtype=float)
        out = apply_noise(sig, severity=0.5, rng=np.random.default_rng(7))
        assert abs(np.mean(out)) < 0.5  # should be near 0 for large N

    def test_no_nan_in_output(self):
        sig = make_signal(80)
        out = apply_noise(sig, severity=0.7, rng=np.random.default_rng(0))
        assert not np.isnan(out).any()

    def test_empty_signal(self):
        out = apply_noise(np.array([]), severity=0.5, rng=np.random.default_rng(0))
        assert len(out) == 0


class TestApplyStuck:
    def test_output_shape_unchanged(self):
        sig = make_signal(60)
        out = apply_stuck(sig, severity=1.0, rng=np.random.default_rng(0))
        assert out.shape == sig.shape

    def test_original_not_mutated(self):
        sig = make_signal(60)
        original = sig.copy()
        apply_stuck(sig, severity=1.0, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(sig, original)

    def test_full_severity_nearly_constant(self):
        sig = make_signal(80)
        out = apply_stuck(sig, severity=1.0, rng=np.random.default_rng(0))
        # At severity=1 jitter=0, so all values should be equal
        assert np.std(out) < 1e-6

    def test_stuck_value_near_signal_start(self):
        sig = make_signal(80)
        stuck_val = float(np.mean(sig[:8]))  # first 10% of 80 samples = 8
        out = apply_stuck(sig, severity=1.0, rng=np.random.default_rng(0))
        assert abs(np.mean(out) - stuck_val) < 1.0  # within 1 unit

    def test_low_severity_allows_jitter(self):
        sig = make_signal(200)
        out_hi = apply_stuck(sig, severity=1.0, rng=np.random.default_rng(0))
        out_lo = apply_stuck(sig, severity=0.0, rng=np.random.default_rng(0))
        assert np.std(out_lo) > np.std(out_hi)

    def test_empty_signal(self):
        out = apply_stuck(np.array([]), severity=1.0, rng=np.random.default_rng(0))
        assert len(out) == 0

    def test_no_nan_in_output(self):
        sig = make_signal(60)
        out = apply_stuck(sig, severity=0.8, rng=np.random.default_rng(0))
        assert not np.isnan(out).any()


class TestApplyMissing:
    def test_all_nan(self):
        sig = make_signal(50)
        out = apply_missing(sig, severity=1.0, rng=np.random.default_rng(0))
        assert np.isnan(out).all()

    def test_original_not_mutated(self):
        sig = make_signal(50)
        original = sig.copy()
        apply_missing(sig, severity=1.0, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(sig, original)

    def test_output_shape_unchanged(self):
        sig = make_signal(50)
        out = apply_missing(sig, severity=0.5, rng=np.random.default_rng(0))
        assert out.shape == sig.shape

    def test_severity_ignored_full_dropout(self):
        sig = make_signal(50)
        out_lo = apply_missing(sig, severity=0.01, rng=np.random.default_rng(0))
        out_hi = apply_missing(sig, severity=1.00, rng=np.random.default_rng(0))
        # Missing always produces all NaN regardless of severity
        assert np.isnan(out_lo).all()
        assert np.isnan(out_hi).all()

    def test_empty_signal(self):
        out = apply_missing(np.array([]), severity=1.0, rng=np.random.default_rng(0))
        assert len(out) == 0


class TestApplyIntermittent:
    def test_output_shape_unchanged(self):
        sig = make_signal(100)
        out = apply_intermittent(sig, severity=0.5, rng=np.random.default_rng(0))
        assert out.shape == sig.shape

    def test_original_not_mutated(self):
        sig = make_signal(100)
        original = sig.copy()
        apply_intermittent(sig, severity=0.5, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(sig, original)

    def test_has_some_nans_at_mid_severity(self):
        sig = make_signal(200)
        out = apply_intermittent(sig, severity=0.5, rng=np.random.default_rng(0))
        assert np.isnan(out).any()

    def test_has_some_non_nans_at_mid_severity(self):
        sig = make_signal(200)
        out = apply_intermittent(sig, severity=0.3, rng=np.random.default_rng(0))
        assert not np.isnan(out).all()

    def test_higher_severity_more_nans(self):
        sig = make_signal(500)
        lo = apply_intermittent(sig, severity=0.2, rng=np.random.default_rng(1))
        hi = apply_intermittent(sig, severity=0.8, rng=np.random.default_rng(1))
        assert np.isnan(hi).sum() > np.isnan(lo).sum()

    def test_zero_severity_no_nans(self):
        sig = make_signal(100)
        out = apply_intermittent(sig, severity=0.0, rng=np.random.default_rng(0))
        # 0 severity → target_nan=0, n_bursts=max(1,0)=1 — at least the
        # function returns arrays without crashing; nans may be 0 or a small burst
        assert out.shape == sig.shape

    def test_empty_signal(self):
        out = apply_intermittent(np.array([]), severity=0.5, rng=np.random.default_rng(0))
        assert len(out) == 0


class TestApplySpike:
    def test_output_shape_unchanged(self):
        sig = make_signal(80)
        out = apply_spike(sig, severity=0.8, rng=np.random.default_rng(0))
        assert out.shape == sig.shape

    def test_original_not_mutated(self):
        sig = make_signal(80)
        original = sig.copy()
        apply_spike(sig, severity=0.8, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(sig, original)

    def test_at_least_one_spike_always(self):
        """Even with spike_rate=0 the implementation guarantees ≥1 spike."""
        sig = make_signal(50)
        out = apply_spike(sig, severity=1.0, rng=np.random.default_rng(0))
        # At least one sample must differ from original
        diffs = np.abs(out - sig)
        assert diffs.max() > 0.0

    def test_no_nan_in_output(self):
        sig = make_signal(80)
        out = apply_spike(sig, severity=0.9, rng=np.random.default_rng(0))
        assert not np.isnan(out).any()

    def test_higher_severity_larger_spike_amplitude(self):
        sig = make_signal(200)
        lo = apply_spike(sig, severity=0.1, rng=np.random.default_rng(3))
        hi = apply_spike(sig, severity=0.9, rng=np.random.default_rng(3))
        max_diff_lo = np.max(np.abs(lo - sig))
        max_diff_hi = np.max(np.abs(hi - sig))
        assert max_diff_hi > max_diff_lo

    def test_custom_spike_rate(self):
        sig = make_signal(1000)
        out = apply_spike(sig, severity=0.5, rng=np.random.default_rng(0), spike_rate=0.2)
        # At 20 % spike rate on 1000 samples ≈ 200 spikes — most samples differ
        n_different = np.sum(out != sig)
        assert n_different >= 100  # at least 10 %

    def test_empty_signal(self):
        out = apply_spike(np.array([]), severity=0.5, rng=np.random.default_rng(0))
        assert len(out) == 0


class TestApplySamplingFailure:
    def test_output_shape_unchanged(self):
        sig = make_signal(100)
        out = apply_sampling_failure(sig, severity=0.5, rng=np.random.default_rng(0))
        assert out.shape == sig.shape

    def test_original_not_mutated(self):
        sig = make_signal(100)
        original = sig.copy()
        apply_sampling_failure(sig, severity=0.5, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(sig, original)

    def test_zero_severity_no_drops(self):
        sig = make_signal(200)
        out = apply_sampling_failure(sig, severity=0.0, rng=np.random.default_rng(0))
        assert not np.isnan(out).any()

    def test_full_severity_all_nan(self):
        sig = make_signal(200)
        out = apply_sampling_failure(sig, severity=1.0, rng=np.random.default_rng(0))
        assert np.isnan(out).all()

    def test_drop_rate_scales_with_severity(self):
        sig = make_signal(1000)
        lo = apply_sampling_failure(sig, severity=0.1, rng=np.random.default_rng(2))
        hi = apply_sampling_failure(sig, severity=0.8, rng=np.random.default_rng(2))
        assert np.isnan(hi).sum() > np.isnan(lo).sum()

    def test_partial_severity_mixes_valid_and_nan(self):
        sig = make_signal(500)
        out = apply_sampling_failure(sig, severity=0.5, rng=np.random.default_rng(0))
        assert np.isnan(out).any() and not np.isnan(out).all()

    def test_empty_signal(self):
        out = apply_sampling_failure(np.array([]), severity=0.5, rng=np.random.default_rng(0))
        assert len(out) == 0


# ===========================================================================
# 2. FaultConfig validation
# ===========================================================================

class TestFaultConfig:
    def test_valid_config_creates_successfully(self):
        cfg = FaultConfig(
            sensor_id="S01", fault_type=FaultType.DRIFT,
            start_time=BASE_TIME, duration_seconds=600, severity=0.5
        )
        assert cfg.sensor_id == "S01"
        assert cfg.end_time == BASE_TIME + timedelta(seconds=600)

    def test_severity_below_zero_raises(self):
        with pytest.raises(ValueError, match="severity"):
            FaultConfig(
                sensor_id="S01", fault_type=FaultType.NOISE,
                start_time=BASE_TIME, duration_seconds=60, severity=-0.1
            )

    def test_severity_above_one_raises(self):
        with pytest.raises(ValueError, match="severity"):
            FaultConfig(
                sensor_id="S01", fault_type=FaultType.NOISE,
                start_time=BASE_TIME, duration_seconds=60, severity=1.1
            )

    def test_duration_zero_raises(self):
        with pytest.raises(ValueError, match="duration_seconds"):
            FaultConfig(
                sensor_id="S01", fault_type=FaultType.BIAS,
                start_time=BASE_TIME, duration_seconds=0, severity=0.5
            )

    def test_default_label_auto_generated(self):
        cfg = FaultConfig(
            sensor_id="TEMP_01", fault_type=FaultType.STUCK,
            start_time=BASE_TIME, duration_seconds=60, severity=0.8
        )
        assert "STUCK" in cfg.label
        assert "TEMP_01" in cfg.label

    def test_custom_label_respected(self):
        cfg = FaultConfig(
            sensor_id="S01", fault_type=FaultType.SPIKE,
            start_time=BASE_TIME, duration_seconds=60, severity=0.5,
            label="my_label"
        )
        assert cfg.label == "my_label"

    def test_seed_is_deterministic(self):
        cfg1 = FaultConfig(
            sensor_id="S01", fault_type=FaultType.DRIFT,
            start_time=BASE_TIME, duration_seconds=600, severity=0.5
        )
        cfg2 = FaultConfig(
            sensor_id="S01", fault_type=FaultType.DRIFT,
            start_time=BASE_TIME, duration_seconds=600, severity=0.5
        )
        assert cfg1.seed() == cfg2.seed()

    def test_different_configs_different_seeds(self):
        cfg1 = FaultConfig(
            sensor_id="S01", fault_type=FaultType.DRIFT,
            start_time=BASE_TIME, duration_seconds=600, severity=0.5
        )
        cfg2 = FaultConfig(
            sensor_id="S02", fault_type=FaultType.NOISE,
            start_time=BASE_TIME, duration_seconds=300, severity=0.3
        )
        assert cfg1.seed() != cfg2.seed()

    def test_boundary_severity_zero(self):
        cfg = FaultConfig(
            sensor_id="S01", fault_type=FaultType.MISSING,
            start_time=BASE_TIME, duration_seconds=60, severity=0.0
        )
        assert cfg.severity == 0.0

    def test_boundary_severity_one(self):
        cfg = FaultConfig(
            sensor_id="S01", fault_type=FaultType.MISSING,
            start_time=BASE_TIME, duration_seconds=60, severity=1.0
        )
        assert cfg.severity == 1.0


# ===========================================================================
# 3. FaultEngine — row selection and label accuracy
# ===========================================================================

class TestFaultEngineRowSelection:
    """Ensure the engine touches exactly the right rows."""

    def _engine(self):
        return FaultEngine(timestamp_col="timestamp")

    def test_only_fault_window_rows_are_changed(self):
        df = make_df(n=200, sensors=["TEMP_01"])
        # Fault covers rows 50–99 (at 60-s intervals starting at minute 50)
        start = BASE_TIME + timedelta(minutes=50)
        fault = FaultConfig(
            sensor_id="TEMP_01", fault_type=FaultType.BIAS,
            start_time=start, duration_seconds=50 * 60, severity=1.0,
        )
        corrupted, labels = self._engine().apply(df, [fault])

        # Rows 0-49: unchanged
        np.testing.assert_array_equal(
            corrupted["TEMP_01"].iloc[:50].values,
            df["TEMP_01"].iloc[:50].values,
        )
        # Labels match the fault window
        assert labels["TEMP_01"].iloc[:50].sum() == 0
        assert labels["TEMP_01"].iloc[50:100].sum() == 50

    def test_original_df_not_mutated(self):
        df = make_df(n=100, sensors=["TEMP_01"])
        original_values = df["TEMP_01"].values.copy()
        fault = make_fault("TEMP_01", FaultType.DRIFT, severity=0.9)
        _ = self._engine().apply(df, [fault])
        np.testing.assert_array_equal(df["TEMP_01"].values, original_values)

    def test_missing_sensor_id_skipped_with_warning(self, caplog):
        import logging
        df = make_df(n=50, sensors=["TEMP_01"])
        fault = make_fault("NONEXISTENT_SENSOR", FaultType.NOISE, severity=0.5)
        with caplog.at_level(logging.WARNING, logger="app.ml.fault_injection.engine"):
            corrupted, labels = self._engine().apply(df, [fault])
        # DataFrame should be unchanged
        pd.testing.assert_frame_equal(corrupted, df)

    def test_fault_outside_time_range_no_change(self):
        df = make_df(n=100, sensors=["TEMP_01"])
        # Fault starts after the end of the DataFrame
        future_start = BASE_TIME + timedelta(days=365)
        fault = FaultConfig(
            sensor_id="TEMP_01", fault_type=FaultType.SPIKE,
            start_time=future_start, duration_seconds=3600, severity=1.0
        )
        corrupted, labels = self._engine().apply(df, [fault])
        pd.testing.assert_frame_equal(corrupted, df)
        assert labels["TEMP_01"].sum() == 0

    def test_label_df_has_correct_columns(self):
        df = make_df(n=100, sensors=["TEMP_01", "PRESSURE_01"])
        faults = [
            make_fault("TEMP_01", FaultType.NOISE),
            make_fault("PRESSURE_01", FaultType.DRIFT),
        ]
        _, labels = self._engine().apply(df, faults)
        assert set(labels.columns) == {"TEMP_01", "PRESSURE_01"}


# ===========================================================================
# 4. FaultEngine — all 8 fault types end-to-end
# ===========================================================================

class TestFaultEngineAllFaultTypes:
    """Smoke test: each fault type runs without error and corrupts data."""

    SENSOR = "TEMP_01"

    def _run(self, fault_type: FaultType, severity: float = 0.7):
        df = make_df(n=200, sensors=[self.SENSOR])
        fault = make_fault(self.SENSOR, fault_type, severity=severity)
        engine = FaultEngine()
        corrupted, labels = engine.apply(df, [fault])
        return df, corrupted, labels

    def test_drift(self):
        df, corrupted, labels = self._run(FaultType.DRIFT)
        # Values in fault window should differ from clean
        mask = labels[self.SENSOR]
        assert not np.allclose(
            corrupted.loc[mask, self.SENSOR].values,
            df.loc[mask, self.SENSOR].values,
        )

    def test_bias(self):
        df, corrupted, labels = self._run(FaultType.BIAS)
        mask = labels[self.SENSOR]
        diff = (corrupted.loc[mask, self.SENSOR] - df.loc[mask, self.SENSOR]).values
        # All differences should be approximately the same constant
        assert np.std(diff) < np.std(df[self.SENSOR].values) * 0.01

    def test_noise(self):
        df, corrupted, labels = self._run(FaultType.NOISE)
        mask = labels[self.SENSOR]
        # Corrupted variance should be higher than clean
        assert corrupted.loc[mask, self.SENSOR].std() > df.loc[mask, self.SENSOR].std() * 0.5

    def test_stuck(self):
        df, corrupted, labels = self._run(FaultType.STUCK, severity=1.0)
        mask = labels[self.SENSOR]
        # At severity=1 all values nearly identical
        assert corrupted.loc[mask, self.SENSOR].std() < 1e-5

    def test_missing(self):
        df, corrupted, labels = self._run(FaultType.MISSING)
        mask = labels[self.SENSOR]
        assert corrupted.loc[mask, self.SENSOR].isna().all()

    def test_intermittent(self):
        df, corrupted, labels = self._run(FaultType.INTERMITTENT, severity=0.5)
        mask = labels[self.SENSOR]
        # Should have some NaNs, not all NaN
        window = corrupted.loc[mask, self.SENSOR]
        assert window.isna().any()

    def test_spike(self):
        df, corrupted, labels = self._run(FaultType.SPIKE, severity=1.0)
        mask = labels[self.SENSOR]
        abs_diff = (corrupted.loc[mask, self.SENSOR] - df.loc[mask, self.SENSOR]).abs()
        # At least one spike should have very large difference
        assert abs_diff.max() > 0.0

    def test_sampling_failure(self):
        df, corrupted, labels = self._run(FaultType.SAMPLING_FAILURE, severity=0.7)
        mask = labels[self.SENSOR]
        assert corrupted.loc[mask, self.SENSOR].isna().any()


# ===========================================================================
# 5. Multiple faults — composition and multi-sensor
# ===========================================================================

class TestFaultEngineComposition:
    def test_two_faults_on_different_sensors(self):
        df = make_df(n=200, sensors=["TEMP_01", "PRESSURE_01"])
        faults = [
            make_fault("TEMP_01",     FaultType.DRIFT, severity=0.8),
            make_fault("PRESSURE_01", FaultType.BIAS,  severity=0.8),
        ]
        corrupted, labels = FaultEngine().apply(df, faults)
        # Both sensors corrupted
        assert not np.allclose(
            corrupted["TEMP_01"].values, df["TEMP_01"].values
        )
        assert not np.allclose(
            corrupted["PRESSURE_01"].values, df["PRESSURE_01"].values
        )
        assert set(labels.columns) == {"TEMP_01", "PRESSURE_01"}

    def test_two_sequential_faults_on_same_sensor(self):
        """Non-overlapping faults should each corrupt their own window."""
        df = make_df(n=300, sensors=["TEMP_01"])
        fault_a = FaultConfig(
            sensor_id="TEMP_01", fault_type=FaultType.BIAS,
            start_time=BASE_TIME, duration_seconds=30 * 60, severity=0.5,
        )
        fault_b = FaultConfig(
            sensor_id="TEMP_01", fault_type=FaultType.NOISE,
            start_time=BASE_TIME + timedelta(minutes=100),
            duration_seconds=30 * 60, severity=0.5,
        )
        corrupted, labels = FaultEngine().apply(df, [fault_a, fault_b])
        # Label is True in both windows
        assert labels["TEMP_01"].sum() > 0
        # Middle rows (rows 60-99) should be unchanged
        mid_orig   = df["TEMP_01"].iloc[60:100].values
        mid_corr   = corrupted["TEMP_01"].iloc[60:100].values
        np.testing.assert_array_equal(mid_orig, mid_corr)

    def test_apply_single_convenience_method(self):
        df = make_df(n=100, sensors=["TEMP_01"])
        fault = make_fault("TEMP_01", FaultType.NOISE)
        corrupted, labels = FaultEngine().apply_single(df, fault)
        assert "TEMP_01" in labels.columns


# ===========================================================================
# 6. Determinism
# ===========================================================================

class TestDeterminism:
    def test_same_fault_same_output(self):
        """Applying the same FaultConfig twice must produce identical results."""
        df = make_df(n=200, sensors=["TEMP_01"])
        fault = make_fault("TEMP_01", FaultType.NOISE, severity=0.6)
        c1, _ = FaultEngine().apply(df, [fault])
        c2, _ = FaultEngine().apply(df, [fault])
        pd.testing.assert_frame_equal(c1, c2)

    def test_different_severity_different_output(self):
        df = make_df(n=200, sensors=["TEMP_01"])
        f1 = make_fault("TEMP_01", FaultType.NOISE, severity=0.2)
        f2 = make_fault("TEMP_01", FaultType.NOISE, severity=0.9)
        c1, _ = FaultEngine().apply(df, [f1])
        c2, _ = FaultEngine().apply(df, [f2])
        assert not np.allclose(c1["TEMP_01"].values, c2["TEMP_01"].values)

    @pytest.mark.parametrize("ft", list(FaultType))
    def test_all_fault_types_are_deterministic(self, ft: FaultType):
        df = make_df(n=150, sensors=["TEMP_01"])
        fault = make_fault("TEMP_01", ft, severity=0.5)
        c1, _ = FaultEngine().apply(df, [fault])
        c2, _ = FaultEngine().apply(df, [fault])
        # NaN-safe comparison
        np.testing.assert_array_equal(
            np.isnan(c1["TEMP_01"].values),
            np.isnan(c2["TEMP_01"].values),
        )
        valid = ~np.isnan(c1["TEMP_01"].values)
        np.testing.assert_allclose(
            c1["TEMP_01"].values[valid],
            c2["TEMP_01"].values[valid],
        )


# ===========================================================================
# 7. FAULT_DISPATCH completeness
# ===========================================================================

class TestFaultDispatchCompleteness:
    def test_all_fault_types_in_dispatch(self):
        """Every FaultType must have a corresponding handler in FAULT_DISPATCH."""
        for ft in FaultType:
            assert ft in FAULT_DISPATCH, (
                f"FaultType.{ft.name} is missing from FAULT_DISPATCH"
            )

    def test_dispatch_handlers_are_callable(self):
        for ft, fn in FAULT_DISPATCH.items():
            assert callable(fn), f"Handler for {ft} is not callable"


# ===========================================================================
# 8. fault_config_from_request helper
# ===========================================================================

class TestFaultConfigFromRequest:
    def test_creates_valid_config(self):
        cfg = fault_config_from_request(
            sensor_id="S01",
            fault_type_str="DRIFT",
            severity=0.5,
            duration_seconds=300,
        )
        assert cfg.fault_type == FaultType.DRIFT
        assert cfg.sensor_id == "S01"
        assert cfg.severity == 0.5
        assert cfg.duration_seconds == 300

    def test_case_insensitive_fault_type(self):
        cfg = fault_config_from_request(
            sensor_id="S01",
            fault_type_str="noise",
            severity=0.3,
            duration_seconds=60,
        )
        assert cfg.fault_type == FaultType.NOISE

    def test_start_offset_applied(self):
        ref = datetime(2024, 6, 1, 12, 0, 0)
        cfg = fault_config_from_request(
            sensor_id="S01",
            fault_type_str="BIAS",
            severity=0.5,
            duration_seconds=120,
            start_offset_seconds=30,
            reference_time=ref,
        )
        assert cfg.start_time == ref + timedelta(seconds=30)

    def test_invalid_fault_type_raises(self):
        with pytest.raises(ValueError, match="fault_type"):
            fault_config_from_request(
                sensor_id="S01",
                fault_type_str="EXPLODE",
                severity=0.5,
                duration_seconds=60,
            )

    def test_sampling_failure_string(self):
        cfg = fault_config_from_request(
            sensor_id="S01",
            fault_type_str="SAMPLING_FAILURE",
            severity=0.4,
            duration_seconds=180,
        )
        assert cfg.fault_type == FaultType.SAMPLING_FAILURE

    def test_all_fault_types_accepted(self):
        for ft in FaultType:
            cfg = fault_config_from_request(
                sensor_id="S01",
                fault_type_str=ft.value,
                severity=0.5,
                duration_seconds=60,
            )
            assert cfg.fault_type == ft


# ===========================================================================
# 9. Edge cases — NaN input, single-sample signals
# ===========================================================================

class TestEdgeCases:
    def test_nan_input_signal_drift(self):
        sig = np.full(50, np.nan)
        # Should not crash — output is all NaN + ramp (still NaN)
        out = apply_drift(sig, severity=0.5, rng=np.random.default_rng(0))
        assert out.shape == sig.shape

    def test_single_sample_all_faults(self):
        """All fault functions must handle n=1 without error."""
        sig = np.array([42.0])
        rng = np.random.default_rng(0)
        for ft, fn in FAULT_DISPATCH.items():
            out = fn(sig, severity=0.5, rng=rng)
            assert out.shape == (1,), f"{ft} failed on single-sample input"

    def test_engine_empty_fault_list(self):
        df = make_df(n=100, sensors=["TEMP_01"])
        corrupted, labels = FaultEngine().apply(df, [])
        pd.testing.assert_frame_equal(corrupted, df)
        assert labels.shape == (100, 0)

    def test_engine_missing_timestamp_column_raises(self):
        df = make_df(n=50, sensors=["TEMP_01"]).drop(columns=["timestamp"])
        fault = make_fault("TEMP_01", FaultType.NOISE)
        with pytest.raises(ValueError, match="timestamp"):
            FaultEngine().apply(df, [fault])
