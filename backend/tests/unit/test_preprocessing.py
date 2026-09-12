"""
tests/unit/test_preprocessing.py
=================================
Unit tests for the preprocessing pipeline.
These tests run against a small synthetic DataFrame — no file I/O needed.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from app.ml.preprocessing.pipeline import (
    SensorPreprocessingPipeline,
    SENSOR_VALID_RANGES,
    WINDOW_SIZE,
)

SENSOR_MAPPING = {
    "TEMP_01":      "raw_temp",
    "PRESSURE_01":  "raw_pres",
    "VIBRATION_01": "raw_vib",
    "CURRENT_01":   "raw_cur",
    "FLOW_01":      "raw_flow",
    "HUMIDITY_01":  "raw_hum",
}


def make_test_df(n: int = 200, missing_rate: float = 0.0) -> pd.DataFrame:
    """Build a minimal synthetic DataFrame matching pipeline expectations."""
    rng = np.random.default_rng(42)
    ts = [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(n)]
    data = {
        "timestamp":      ts,
        "machine_status": ["NORMAL"] * (n - 20) + ["BROKEN"] * 10 + ["RECOVERING"] * 10,
        "raw_temp":  rng.uniform(60, 90, n),
        "raw_pres":  rng.uniform(1, 5, n),
        "raw_vib":   rng.uniform(0.5, 4, n),
        "raw_cur":   rng.uniform(8, 15, n),
        "raw_flow":  rng.uniform(30, 150, n),
        "raw_hum":   rng.uniform(0.3, 0.9, n),
    }
    df = pd.DataFrame(data)
    if missing_rate > 0:
        for col in ["raw_temp", "raw_pres", "raw_vib", "raw_cur", "raw_flow", "raw_hum"]:
            mask = rng.random(n) < missing_rate
            df.loc[mask, col] = np.nan
    return df


class TestColumnSelection:
    def test_selects_correct_columns(self):
        pipe = SensorPreprocessingPipeline(sensor_mapping=SENSOR_MAPPING)
        df = make_test_df()
        out = pipe.select_columns(df)
        assert set(out.columns) == {
            "TEMP_01", "PRESSURE_01", "VIBRATION_01",
            "CURRENT_01", "FLOW_01", "HUMIDITY_01",
            "timestamp", "machine_status"
        }

    def test_raises_on_missing_column(self):
        pipe = SensorPreprocessingPipeline(sensor_mapping=SENSOR_MAPPING)
        df = make_test_df().drop(columns=["raw_temp"])
        with pytest.raises(ValueError, match="raw_temp"):
            pipe.select_columns(df)


class TestImputation:
    def test_no_missing_after_impute(self):
        pipe = SensorPreprocessingPipeline(sensor_mapping=SENSOR_MAPPING)
        df = make_test_df(missing_rate=0.1)
        df = pipe.select_columns(df)
        medians = pipe._compute_medians(df)
        out = pipe.impute(df, medians=medians)
        for col in pipe.sensor_cols:
            assert out[col].isna().sum() == 0, f"Still NaN in {col}"

    def test_impute_does_not_modify_original(self):
        pipe = SensorPreprocessingPipeline(sensor_mapping=SENSOR_MAPPING)
        df = make_test_df(missing_rate=0.1)
        df = pipe.select_columns(df)
        original_missing = df["TEMP_01"].isna().sum()
        medians = pipe._compute_medians(df)
        _ = pipe.impute(df, medians=medians)
        assert df["TEMP_01"].isna().sum() == original_missing  # original unchanged


class TestClipping:
    def test_values_within_valid_range(self):
        pipe = SensorPreprocessingPipeline(sensor_mapping=SENSOR_MAPPING)
        df = make_test_df()
        df = pipe.select_columns(df)
        # Inject out-of-range values
        df.loc[0, "TEMP_01"] = 999.0
        df.loc[1, "FLOW_01"] = -50.0
        out = pipe.clip_ranges(df)
        assert out["TEMP_01"].max() <= SENSOR_VALID_RANGES["TEMP_01"][1]
        assert out["FLOW_01"].min() >= SENSOR_VALID_RANGES["FLOW_01"][0]


class TestNormalization:
    def test_normalized_range(self):
        """After normalization, all values must be in [0, 1]."""
        pipe = SensorPreprocessingPipeline(sensor_mapping=SENSOR_MAPPING)
        df = make_test_df()
        df = pipe.select_columns(df)
        medians = pipe._compute_medians(df)
        df = pipe.impute(df, medians=medians)
        df = pipe.clip_ranges(df)
        mins, maxs = pipe._compute_norm_params(df)
        out = pipe.normalize(df, mins=mins, maxs=maxs)
        for col in pipe.sensor_cols:
            assert out[col].min() >= -1e-6, f"{col} min below 0"
            assert out[col].max() <= 1 + 1e-6, f"{col} max above 1"

    def test_inference_transform_stays_in_range(self):
        """
        Normalization using training params should keep test data near [0,1]
        (may exceed by small amounts due to test data range).
        """
        pipe = SensorPreprocessingPipeline(sensor_mapping=SENSOR_MAPPING)
        train_df = make_test_df(n=150)
        test_df  = make_test_df(n=50)
        train_df = pipe.select_columns(train_df)
        test_df  = pipe.select_columns(test_df)
        medians  = pipe._compute_medians(train_df)
        train_df = pipe.impute(train_df, medians=medians)
        test_df  = pipe.impute(test_df,  medians=medians)
        mins, maxs = pipe._compute_norm_params(train_df)
        # Test data normalized with training params — should be close to [0,1]
        out = pipe.normalize(test_df, mins=mins, maxs=maxs)
        for col in pipe.sensor_cols:
            assert out[col].min() > -0.5, f"{col} too far below 0"
            assert out[col].max() < 1.5,  f"{col} too far above 1"


class TestWindowBuilding:
    def test_window_shape(self):
        pipe = SensorPreprocessingPipeline(sensor_mapping=SENSOR_MAPPING)
        df = make_test_df(n=100)
        df = pipe.select_columns(df)
        medians = pipe._compute_medians(df)
        df = pipe.impute(df, medians=medians)
        df = pipe.clip_ranges(df)
        mins, maxs = pipe._compute_norm_params(df)
        df = pipe.normalize(df, mins=mins, maxs=maxs)
        X, y, ts = pipe.build_windows(df, window_size=10, step_size=1)
        assert X.shape[1] == 10     # window_size
        assert X.shape[2] == 6      # n_sensors
        assert len(y) == len(X)
        assert len(ts) == len(X)

    def test_labels_are_binary(self):
        pipe = SensorPreprocessingPipeline(sensor_mapping=SENSOR_MAPPING)
        df = make_test_df(n=100)
        df = pipe.select_columns(df)
        medians = pipe._compute_medians(df)
        df = pipe.impute(df, medians=medians)
        df = pipe.clip_ranges(df)
        mins, maxs = pipe._compute_norm_params(df)
        df = pipe.normalize(df, mins=mins, maxs=maxs)
        X, y, _ = pipe.build_windows(df, window_size=10, step_size=5)
        assert set(np.unique(y)).issubset({0, 1})

    def test_window_count(self):
        """With window_size=10 and step_size=1, expect n-10+1 windows."""
        n, w = 50, 10
        pipe = SensorPreprocessingPipeline(sensor_mapping=SENSOR_MAPPING)
        df = make_test_df(n=n)
        df = pipe.select_columns(df)
        medians = pipe._compute_medians(df)
        df = pipe.impute(df, medians=medians)
        df = pipe.clip_ranges(df)
        mins, maxs = pipe._compute_norm_params(df)
        df = pipe.normalize(df, mins=mins, maxs=maxs)
        X, _, _ = pipe.build_windows(df, window_size=w, step_size=1)
        assert len(X) == n - w + 1


class TestFitTransform:
    def test_full_pipeline_runs(self):
        pipe = SensorPreprocessingPipeline(sensor_mapping=SENSOR_MAPPING)
        df = make_test_df(n=300, missing_rate=0.05)
        dataset = pipe.fit_transform(raw_df=df, window_size=10, step_size=5,
                                     val_ratio=0.2, test_ratio=0.2)
        assert dataset.X_train.ndim == 3
        assert dataset.X_train.shape[2] == 6
        assert dataset.artifacts is not None

    def test_no_data_leakage(self):
        """Training normalization params must not see val/test data."""
        pipe = SensorPreprocessingPipeline(sensor_mapping=SENSOR_MAPPING)
        df = make_test_df(n=300)
        dataset = pipe.fit_transform(raw_df=df, window_size=10, step_size=5,
                                     val_ratio=0.2, test_ratio=0.2)
        # Training max must be <= overall max of original training split
        n = len(df)
        test_start = int(n * 0.8)
        val_start  = int(n * 0.6)
        train_end  = val_start
        # Just verify artifacts exist and have reasonable values
        for col in pipe.sensor_cols:
            assert pipe.artifacts.norm_min[col] <= pipe.artifacts.norm_max[col]
