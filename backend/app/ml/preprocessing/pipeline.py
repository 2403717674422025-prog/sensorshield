"""
ml/preprocessing/pipeline.py
=============================
Production preprocessing pipeline for SensorShield.

Responsibilities:
  1. Load raw sensor CSV (or accept a DataFrame directly for streaming use)
  2. Map dataset columns → machine sensor IDs using sensor_mapping.json
  3. Impute missing values (forward fill → linear interpolation → median)
  4. Clip outliers beyond valid sensor ranges
  5. Normalize to [0,1] using per-sensor min-max from training data
  6. Build sliding windows for ML models
  7. Create binary target label from machine_status
  8. Save processed artifacts to data/processed/

Design principle:
  - The Normalizer is fitted ONLY on the training split.
  - The same fitted Normalizer is applied to val, test, and live inference.
  - All fitted parameters are saved to disk so inference can reload them
    without reprocessing training data.

Usage:
    python -m app.ml.preprocessing.pipeline   # from backend/
    OR import and call programmatically from tests / other modules.
"""

import json
import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths (resolved relative to this file so the module works from any cwd)
# ---------------------------------------------------------------------------
_HERE       = os.path.dirname(os.path.abspath(__file__))
_BACKEND    = os.path.join(_HERE, "..", "..", "..", "..")  # sensorshield/
_DATA_RAW   = os.path.join(_BACKEND, "data", "raw", "sensor_data.csv")
_DATA_PROC  = os.path.join(_BACKEND, "data", "processed")
_MAPPING    = os.path.join(_DATA_PROC, "sensor_mapping.json")
_ARTIFACTS  = os.path.join(_DATA_PROC, "pipeline_artifacts.pkl")

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Sensor valid ranges — used for outlier clipping and data validation
# These match the ranges defined in the seed script for machine M001
# ---------------------------------------------------------------------------
SENSOR_VALID_RANGES: Dict[str, Tuple[float, float]] = {
    "TEMP_01":      (20.0,  150.0),   # °C
    "PRESSURE_01":  (0.0,   10.0),    # bar
    "VIBRATION_01": (0.0,   20.0),    # mm/s
    "CURRENT_01":   (0.0,   30.0),    # A
    "FLOW_01":      (0.0,   300.0),   # L/min
    "HUMIDITY_01":  (0.0,   1.0),     # ratio (efficiency metric 0–1)
}

# Window size for sliding window (minutes = samples at 1-min rate)
WINDOW_SIZE = 30   # 30-minute lookback window
STEP_SIZE   = 1    # stride 1 for maximum density (reduce in production)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class PipelineArtifacts:
    """
    Everything needed to reproduce the exact same transformation at inference
    time without re-reading training data.
    """
    sensor_cols: List[str]                    # e.g. ["TEMP_01", ...]
    sensor_mapping: Dict[str, str]            # machine_id → dataset_col
    norm_min: Dict[str, float]                # per-sensor training min
    norm_max: Dict[str, float]                # per-sensor training max
    median_fill: Dict[str, float]             # per-sensor training median
    window_size: int = WINDOW_SIZE
    step_size: int   = STEP_SIZE
    version: str     = "1.0"


@dataclass
class ProcessedDataset:
    """Output of the full pipeline."""
    X_train: np.ndarray       # (N_train, window_size, n_sensors)
    X_val:   np.ndarray
    X_test:  np.ndarray
    y_train: np.ndarray       # (N_train,) binary labels
    y_val:   np.ndarray
    y_test:  np.ndarray
    ts_train: np.ndarray      # timestamps for each window (last timestamp)
    ts_val:   np.ndarray
    ts_test:  np.ndarray
    artifacts: PipelineArtifacts
    sensor_cols: List[str]


# ---------------------------------------------------------------------------
# Core pipeline class
# ---------------------------------------------------------------------------
class SensorPreprocessingPipeline:
    """
    Stateful preprocessing pipeline.

    Two modes:
      fit_transform(df)  — learn parameters from data, then transform
      transform(df)      — apply previously fitted parameters (inference)
    """

    def __init__(self, sensor_mapping: Optional[Dict[str, str]] = None):
        """
        sensor_mapping: maps machine sensor ID → dataset column name.
        If None, loaded from data/processed/sensor_mapping.json.
        """
        if sensor_mapping is None:
            with open(_MAPPING) as f:
                sensor_mapping = json.load(f)
        self.sensor_mapping   = sensor_mapping
        self.sensor_cols      = list(sensor_mapping.keys())  # canonical order
        self.artifacts: Optional[PipelineArtifacts] = None

    # ------------------------------------------------------------------
    # Step 1: Load raw CSV
    # ------------------------------------------------------------------
    def load_raw(self, path: str = _DATA_RAW) -> pd.DataFrame:
        df = pd.read_csv(path, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Step 2: Select and rename columns
    # ------------------------------------------------------------------
    def select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Keep only the 6 selected sensor columns + timestamp + machine_status.
        Rename dataset columns to canonical machine sensor IDs.
        """
        rename = {v: k for k, v in self.sensor_mapping.items()}
        keep   = list(self.sensor_mapping.values()) + ["timestamp"]
        if "machine_status" in df.columns:
            keep.append("machine_status")

        missing_cols = [c for c in keep if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in dataframe: {missing_cols}")

        df = df[keep].rename(columns=rename).copy()
        return df

    # ------------------------------------------------------------------
    # Step 3: Imputation  (fit-aware)
    # ------------------------------------------------------------------
    def _compute_medians(self, df: pd.DataFrame) -> Dict[str, float]:
        return {col: float(df[col].median()) for col in self.sensor_cols}

    def impute(self, df: pd.DataFrame,
               medians: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Three-stage imputation:
          1. Forward fill (propagate last good reading)
          2. Linear interpolation (fill gaps between valid readings)
          3. Median fill (handles leading NaN at start of series)

        medians: training medians (must be passed at inference time).
                 If None, computed from df (training mode).
        """
        df = df.copy()
        for col in self.sensor_cols:
            df[col] = (
                df[col]
                .ffill()
                .interpolate(method="linear", limit_direction="both")
            )
            fill_val = medians[col] if medians else float(df[col].median())
            df[col] = df[col].fillna(fill_val)
        return df

    # ------------------------------------------------------------------
    # Step 4: Clip to valid range
    # ------------------------------------------------------------------
    def clip_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, (lo, hi) in SENSOR_VALID_RANGES.items():
            if col in df.columns:
                df[col] = df[col].clip(lo, hi)
        return df

    # ------------------------------------------------------------------
    # Step 5: Normalize  (fit-aware)
    # ------------------------------------------------------------------
    def _compute_norm_params(self, df: pd.DataFrame
                              ) -> Tuple[Dict[str, float], Dict[str, float]]:
        mins = {col: float(df[col].min()) for col in self.sensor_cols}
        maxs = {col: float(df[col].max()) for col in self.sensor_cols}
        return mins, maxs

    def normalize(self, df: pd.DataFrame,
                  mins: Optional[Dict[str, float]] = None,
                  maxs: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Min-max normalize each sensor to [0, 1].
        mins/maxs: from training data (must be passed at inference time).
        """
        df = df.copy()
        for col in self.sensor_cols:
            lo = mins[col] if mins else float(df[col].min())
            hi = maxs[col] if maxs else float(df[col].max())
            rng = hi - lo
            if rng < 1e-9:
                df[col] = 0.0
            else:
                df[col] = (df[col] - lo) / rng
        return df

    # ------------------------------------------------------------------
    # Step 6: Build sliding windows
    # ------------------------------------------------------------------
    def build_windows(
        self,
        df: pd.DataFrame,
        window_size: int = WINDOW_SIZE,
        step_size:   int = STEP_SIZE,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          X  : (N_windows, window_size, n_sensors)  float32
          y  : (N_windows,)                          int8  (1=abnormal, 0=normal)
          ts : (N_windows,)                          datetime64 (last ts in window)
        """
        sensor_data = df[self.sensor_cols].values.astype(np.float32)
        timestamps  = df["timestamp"].values

        # Build binary label: 1 if machine_status != NORMAL
        if "machine_status" in df.columns:
            labels = (df["machine_status"] != "NORMAL").astype(np.int8).values
        else:
            labels = np.zeros(len(df), dtype=np.int8)

        n = len(df)
        windows, window_labels, window_ts = [], [], []

        for start in range(0, n - window_size + 1, step_size):
            end = start + window_size
            windows.append(sensor_data[start:end])
            # Label = 1 if ANY reading in the window is abnormal
            window_labels.append(int(labels[start:end].max()))
            window_ts.append(timestamps[end - 1])

        X  = np.array(windows, dtype=np.float32)
        y  = np.array(window_labels, dtype=np.int8)
        ts = np.array(window_ts)
        return X, y, ts

    # ------------------------------------------------------------------
    # Main: fit_transform (training)
    # ------------------------------------------------------------------
    def fit_transform(
        self,
        raw_df: Optional[pd.DataFrame] = None,
        window_size: int = WINDOW_SIZE,
        step_size:   int = STEP_SIZE,
        val_ratio:   float = 0.15,
        test_ratio:  float = 0.15,
    ) -> ProcessedDataset:
        """
        Full pipeline on raw data — learns all parameters from training split.
        Temporal split: train | val | test  (no shuffling to prevent leakage).
        """
        if raw_df is None:
            print(f"Loading raw data from {_DATA_RAW} …")
            raw_df = self.load_raw()

        print(f"Raw shape: {raw_df.shape}")

        # Select columns
        df = self.select_columns(raw_df)
        print(f"After column selection: {df.shape}")

        # Temporal train/val/test split BEFORE any fitting
        n = len(df)
        test_start = int(n * (1 - test_ratio))
        val_start  = int(n * (1 - val_ratio - test_ratio))

        train_df = df.iloc[:val_start].copy()
        val_df   = df.iloc[val_start:test_start].copy()
        test_df  = df.iloc[test_start:].copy()

        print(f"Split: train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

        # Fit imputation parameters on training set
        medians = self._compute_medians(train_df)

        # Impute all splits using TRAINING medians
        train_df = self.impute(train_df, medians=medians)
        val_df   = self.impute(val_df,   medians=medians)
        test_df  = self.impute(test_df,  medians=medians)

        # Clip ranges
        train_df = self.clip_ranges(train_df)
        val_df   = self.clip_ranges(val_df)
        test_df  = self.clip_ranges(test_df)

        # Fit normalization on TRAINING set only
        mins, maxs = self._compute_norm_params(train_df)

        # Normalize all splits
        train_df = self.normalize(train_df, mins=mins, maxs=maxs)
        val_df   = self.normalize(val_df,   mins=mins, maxs=maxs)
        test_df  = self.normalize(test_df,  mins=mins, maxs=maxs)

        # Build windows
        X_train, y_train, ts_train = self.build_windows(train_df, window_size, step_size)
        X_val,   y_val,   ts_val   = self.build_windows(val_df,   window_size, step_size)
        X_test,  y_test,  ts_test  = self.build_windows(test_df,  window_size, step_size)

        print(f"X_train: {X_train.shape}  y_train: {y_train.shape}  "
              f"pos_rate={y_train.mean():.3f}")
        print(f"X_val  : {X_val.shape}    y_val:   {y_val.shape}    "
              f"pos_rate={y_val.mean():.3f}")
        print(f"X_test : {X_test.shape}   y_test:  {y_test.shape}   "
              f"pos_rate={y_test.mean():.3f}")

        # Save fitted artifacts
        self.artifacts = PipelineArtifacts(
            sensor_cols    = self.sensor_cols,
            sensor_mapping = self.sensor_mapping,
            norm_min       = mins,
            norm_max       = maxs,
            median_fill    = medians,
            window_size    = window_size,
            step_size      = step_size,
        )

        return ProcessedDataset(
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train=y_train, y_val=y_val, y_test=y_test,
            ts_train=ts_train, ts_val=ts_val, ts_test=ts_test,
            artifacts=self.artifacts,
            sensor_cols=self.sensor_cols,
        )

    # ------------------------------------------------------------------
    # Main: transform (inference / streaming)
    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame,
                  artifacts: Optional[PipelineArtifacts] = None
                  ) -> pd.DataFrame:
        """
        Apply the FITTED pipeline to new data (no re-fitting).
        Used in the live inference pipeline (Phase 12).
        """
        art = artifacts or self.artifacts
        if art is None:
            raise RuntimeError("Pipeline has no fitted artifacts. Call fit_transform() first "
                               "or load artifacts with load_artifacts().")
        df = self.select_columns(df)
        df = self.impute(df, medians=art.median_fill)
        df = self.clip_ranges(df)
        df = self.normalize(df, mins=art.norm_min, maxs=art.norm_max)
        return df

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_artifacts(self, path: str = _ARTIFACTS) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.artifacts, f)
        print(f"Pipeline artifacts saved → {path}")

    @classmethod
    def load_artifacts(cls, path: str = _ARTIFACTS) -> "SensorPreprocessingPipeline":
        with open(path, "rb") as f:
            artifacts = pickle.load(f)
        pipeline = cls(sensor_mapping=artifacts.sensor_mapping)
        pipeline.artifacts = artifacts
        return pipeline


# ---------------------------------------------------------------------------
# Save processed splits as .npy files for ML notebooks
# ---------------------------------------------------------------------------
def save_splits(dataset: ProcessedDataset, out_dir: str = _DATA_PROC) -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X_train.npy"), dataset.X_train)
    np.save(os.path.join(out_dir, "X_val.npy"),   dataset.X_val)
    np.save(os.path.join(out_dir, "X_test.npy"),  dataset.X_test)
    np.save(os.path.join(out_dir, "y_train.npy"), dataset.y_train)
    np.save(os.path.join(out_dir, "y_val.npy"),   dataset.y_val)
    np.save(os.path.join(out_dir, "y_test.npy"),  dataset.y_test)
    np.save(os.path.join(out_dir, "ts_train.npy"), dataset.ts_train.astype(str))
    np.save(os.path.join(out_dir, "ts_val.npy"),   dataset.ts_val.astype(str))
    np.save(os.path.join(out_dir, "ts_test.npy"),  dataset.ts_test.astype(str))
    print(f"Processed splits saved → {out_dir}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("SensorShield — Preprocessing Pipeline")
    print("=" * 50)

    pipeline = SensorPreprocessingPipeline()
    dataset  = pipeline.fit_transform()
    pipeline.save_artifacts()
    save_splits(dataset)

    print("\nArtifact summary:")
    art = dataset.artifacts
    print(f"  Window size : {art.window_size}")
    print(f"  Sensors     : {art.sensor_cols}")
    print(f"  Norm ranges :")
    for s in art.sensor_cols:
        print(f"    {s}: [{art.norm_min[s]:.4f}, {art.norm_max[s]:.4f}]")
    print("\nPreprocessing complete.")
    print("Next step: Phase 3 — Fault Injection Engine")
