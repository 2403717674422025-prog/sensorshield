"""
scripts/run_phase4_baselines.py
================================
Phase 4 — Baseline Anomaly Detection benchmark.

Pipeline
--------
1. Load raw sensor CSV + pipeline artifacts (norm params from Phase 2)
2. Build a CLEAN evaluation dataset (normal windows only)
3. Inject all 8 fault types at three severities (0.3, 0.6, 0.9)
   → labeled windows via FaultEngine
4. Fit each of 3 detectors on the clean TRAIN split
5. Evaluate on the fault-injected TEST split
6. Save results:
     data/processed/phase4_evaluation_summary.csv
     data/processed/phase4_per_fault_metrics.csv
     data/processed/phase4_report.txt

Usage (from repo root):
    python scripts/run_phase4_baselines.py
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ── make backend importable ──────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_HERE)
_BACKEND = os.path.join(_ROOT, "backend")
sys.path.insert(0, _BACKEND)

from app.ml.preprocessing.pipeline import SensorPreprocessingPipeline, PipelineArtifacts, WINDOW_SIZE, STEP_SIZE
from app.ml.fault_injection.engine import FaultConfig, FaultEngine
from app.ml.fault_injection.faults import FaultType
from app.ml.anomaly_detection import (
    RuleBasedDetector,
    IsolationForestDetector,
    OneClassSVMDetector,
    evaluate_detector,
    build_evaluation_table,
)

import pickle

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(_ROOT, "data")
RAW_CSV    = os.path.join(DATA_DIR, "raw",       "sensor_data.csv")
PROC_DIR   = os.path.join(DATA_DIR, "processed")
ARTS_PKL   = os.path.join(PROC_DIR, "pipeline_artifacts.pkl")
OUT_SUMM   = os.path.join(PROC_DIR, "phase4_evaluation_summary.csv")
OUT_PFAULT = os.path.join(PROC_DIR, "phase4_per_fault_metrics.csv")
OUT_REPORT = os.path.join(PROC_DIR, "phase4_report.txt")

WINDOW_SIZE_EVAL = 30   # match Phase 2
STEP_SIZE_EVAL   = 5    # larger stride for faster evaluation build


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}".encode("ascii", "replace").decode("ascii"))


# ── 1. Load data ─────────────────────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, PipelineArtifacts, SensorPreprocessingPipeline]:
    log("Loading raw CSV ...")
    df = pd.read_csv(RAW_CSV, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    log(f"  {len(df):,} rows, columns: {list(df.columns[:6])} ...")
    with open(ARTS_PKL, "rb") as f:
        arts = pickle.load(f)
    log(f"  Artifacts loaded -- sensors: {arts.sensor_cols}")

    pipe = SensorPreprocessingPipeline()
    return df, arts, pipe


# ── 2. Preprocess clean data ──────────────────────────────────────────────────
def preprocess_clean(
    df: pd.DataFrame,
    pipe: SensorPreprocessingPipeline,
    arts: PipelineArtifacts,
) -> pd.DataFrame:
    """Apply Phase 2 transforms (select, impute, clip, normalize) on the raw DF."""
    df = pipe.select_columns(df)
    df = pipe.impute(df, medians=arts.median_fill)
    df = pipe.clip_ranges(df)
    df = pipe.normalize(df, mins=arts.norm_min, maxs=arts.norm_max)
    return df


# ── 3. Build fault-injected evaluation dataset ────────────────────────────────
def build_eval_dataset(
    raw_df: pd.DataFrame,
    pipe:   SensorPreprocessingPipeline,
    arts:   PipelineArtifacts,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    X_normal : (N_normal, W, S)   — clean windows (for training)
    X_test   : (N_test,  W, S)   — clean + corrupted windows (for evaluation)
    y_test   : (N_test,)          — 1 if window overlaps fault, else 0
    ft_test  : (N_test,)   str   — fault type label (or 'NORMAL')
    ts_test  : (N_test,)          — window end timestamps
    """
    sensor_cols = arts.sensor_cols
    ts_series   = raw_df["timestamp"] if "timestamp" in raw_df.columns else None

    # ── split raw data 60/40 (train / fault-test) ────────────────────────────
    n      = len(raw_df)
    split  = int(n * 0.60)
    df_train = raw_df.iloc[:split].copy()
    df_test  = raw_df.iloc[split:].copy().reset_index(drop=True)

    # ── clean train — preprocess + window ────────────────────────────────────
    log("  Preprocessing clean train split …")
    df_train_clean = preprocess_clean(df_train, pipe, arts)
    X_normal, _, _ = _build_windows(df_train_clean, sensor_cols,
                                     WINDOW_SIZE_EVAL, STEP_SIZE_EVAL)
    log(f"  X_normal shape: {X_normal.shape}")

    # ── fault injection on test split ────────────────────────────────────────
    ts_min = df_test["timestamp"].min()
    ts_max = df_test["timestamp"].max()
    total_s = (ts_max - ts_min).total_seconds()

    faults = _build_fault_configs(sensor_cols, ts_min, total_s)
    log(f"  Injecting {len(faults)} faults …")

    engine             = FaultEngine(timestamp_col="timestamp")
    df_test_raw        = df_test.copy()  # keep raw col names for FaultEngine
    # FaultEngine works on raw (pre-normalised) column names from mapping
    # Inject on the RAW test df then preprocess
    # We need sensor mapping columns in df_test_raw
    mapping_rev = {v: k for k, v in pipe.sensor_mapping.items()}  # raw_col→sensor_id
    # Select only the raw sensor columns + timestamp
    raw_sensor_cols = list(pipe.sensor_mapping.values())
    available = [c for c in raw_sensor_cols if c in df_test_raw.columns]
    df_inject = df_test_raw[["timestamp"] + available].copy()
    # Rename to sensor IDs for FaultEngine
    df_inject = df_inject.rename(columns=mapping_rev)

    corrupted_df, label_df = engine.apply(df_inject, faults)

    # ── preprocess the corrupted DF ──────────────────────────────────────────
    # Rename back to raw col names for the pipeline
    rename_to_raw = {k: v for k, v in pipe.sensor_mapping.items()}
    corrupted_raw = corrupted_df.rename(columns=rename_to_raw)
    # Add machine_status for pipeline compatibility
    corrupted_raw["machine_status"] = "NORMAL"

    log("  Preprocessing corrupted test split …")
    corrupted_proc = preprocess_clean(corrupted_raw, pipe, arts)

    # Build windows on corrupted preprocessed data
    X_test_arr, ts_test, _ = _build_windows(corrupted_proc, sensor_cols,
                                              WINDOW_SIZE_EVAL, STEP_SIZE_EVAL)

    # ── build per-window labels ──────────────────────────────────────────────
    # For each window, check if its rows overlap any fault label
    y_test, ft_test = _window_labels(corrupted_proc, label_df, faults,
                                      sensor_cols, WINDOW_SIZE_EVAL, STEP_SIZE_EVAL)

    log(f"  X_test shape: {X_test_arr.shape}  anomaly rate: {y_test.mean():.2%}")
    return X_normal, X_test_arr, y_test, ft_test, ts_test


def _build_windows(
    df: pd.DataFrame,
    sensor_cols: list[str],
    window_size: int,
    step_size:   int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sliding windows on a preprocessed DataFrame."""
    vals = df[sensor_cols].values.astype(float)   # (N, S)
    ts   = df["timestamp"].values if "timestamp" in df.columns else np.arange(len(df))
    n    = len(vals)
    starts = range(0, n - window_size + 1, step_size)
    X  = np.stack([vals[i:i+window_size] for i in starts])
    ts_out = np.array([ts[i + window_size - 1] for i in starts])
    y  = np.zeros(len(X), dtype=np.int8)
    return X, ts_out, y


def _window_labels(
    corrupted_proc: pd.DataFrame,
    label_df:       pd.DataFrame,
    faults:         list[FaultConfig],
    sensor_cols:    list[str],
    window_size:    int,
    step_size:      int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each window, y=1 if ANY sample in the window is labelled as faulty.
    ft_labels contains the fault type of the first fault active in the window.
    """
    n = len(corrupted_proc)
    # Per-row anomaly flag
    if label_df.empty or len(label_df) == 0:
        row_labels = np.zeros(n, dtype=bool)
    else:
        # label_df is indexed to df_inject rows; re-index to corrupted_proc
        # label_df cols are sensor IDs; OR across sensors per row
        row_any = label_df.any(axis=1).values  # (n_test_raw,)
        # Align by length
        if len(row_any) >= n:
            row_labels = row_any[:n]
        else:
            row_labels = np.zeros(n, dtype=bool)
            row_labels[:len(row_any)] = row_any

    # Per-row fault type (first fault whose window is active)
    row_ftype = np.full(n, "", dtype=object)
    ts_arr    = corrupted_proc["timestamp"].values if "timestamp" in corrupted_proc.columns else None
    if ts_arr is not None:
        for fault in faults:
            start = np.datetime64(fault.start_time)
            end   = np.datetime64(fault.end_time)
            mask  = (ts_arr >= start) & (ts_arr < end)
            row_ftype[mask] = fault.fault_type.value

    starts = range(0, n - window_size + 1, step_size)
    y_test  = np.array([int(row_labels[i:i+window_size].any()) for i in starts], dtype=np.int8)
    ft_test = np.array([
        next((row_ftype[i+j] for j in range(window_size) if row_ftype[i+j]), "NORMAL")
        for i in starts
    ], dtype=object)

    return y_test, ft_test


def _build_fault_configs(
    sensor_cols: list[str],
    ts_min:      datetime,
    total_s:     float,
) -> list[FaultConfig]:
    """Define one fault per type, spread across the test timeline."""
    def t(frac: float) -> datetime:
        return ts_min + timedelta(seconds=total_s * frac)

    def s(idx: int) -> str:
        return sensor_cols[idx % len(sensor_cols)]

    # Duration: 5 % of timeline each, severity mix of low/mid/high
    dur = max(60, int(total_s * 0.05))
    return [
        FaultConfig(sensor_id=s(0), fault_type=FaultType.DRIFT,            start_time=t(0.02), duration_seconds=dur, severity=0.7),
        FaultConfig(sensor_id=s(1), fault_type=FaultType.BIAS,             start_time=t(0.14), duration_seconds=dur, severity=0.8),
        FaultConfig(sensor_id=s(2), fault_type=FaultType.NOISE,            start_time=t(0.26), duration_seconds=dur, severity=0.8),
        FaultConfig(sensor_id=s(3), fault_type=FaultType.STUCK,            start_time=t(0.38), duration_seconds=dur, severity=1.0),
        FaultConfig(sensor_id=s(4), fault_type=FaultType.MISSING,          start_time=t(0.50), duration_seconds=dur, severity=1.0),
        FaultConfig(sensor_id=s(5), fault_type=FaultType.INTERMITTENT,     start_time=t(0.62), duration_seconds=dur, severity=0.5),
        FaultConfig(sensor_id=s(0), fault_type=FaultType.SPIKE,            start_time=t(0.74), duration_seconds=dur, severity=0.9),
        FaultConfig(sensor_id=s(1), fault_type=FaultType.SAMPLING_FAILURE, start_time=t(0.86), duration_seconds=dur, severity=0.6),
    ]


# ── 4. Run detectors ──────────────────────────────────────────────────────────
def run_all_detectors(X_normal, X_test, y_test, ft_test):
    detectors = [
        RuleBasedDetector(z_threshold=3.0, std_multiplier=3.0, nan_threshold=0.2),
        IsolationForestDetector(n_estimators=200, contamination=0.01, random_state=42),
        OneClassSVMDetector(nu=0.01, gamma="scale"),
    ]
    results = []
    for det in detectors:
        log(f"  Evaluating {det.name} …")
        result = evaluate_detector(
            detector       = det,
            X_normal_train = X_normal,
            X_test         = X_test,
            y_test         = y_test,
            fault_types    = ft_test,
            window_step_s  = 60 * STEP_SIZE_EVAL,
        )
        log(f"    F1={result.overall_f1:.3f}  AUROC={result.overall_auroc:.3f}  "
            f"AP={result.overall_avg_prec:.3f}  "
            f"fit={result.fit_time_s:.1f}s  score={result.score_time_s:.2f}s")
        results.append(result)
    return results


# ── 5. Save outputs ───────────────────────────────────────────────────────────
def save_results(results, summary_df, per_fault_df):
    summary_df.to_csv(OUT_SUMM)
    log(f"  Saved summary -> {OUT_SUMM}")

    if not per_fault_df.empty:
        per_fault_df.to_csv(OUT_PFAULT, index=False)
        log(f"  Saved per-fault -> {OUT_PFAULT}")

    # Text report
    lines = [
        "=" * 72,
        "SensorShield Phase 4 -- Baseline Anomaly Detection Report",
        f"Generated : {datetime.now().isoformat()}",
        "=" * 72,
        "",
        "OVERALL METRICS",
        "-" * 72,
        summary_df.to_string(),
        "",
    ]
    if not per_fault_df.empty:
        lines += [
            "PER-FAULT-TYPE BREAKDOWN",
            "-" * 72,
            per_fault_df.to_string(index=False),
            "",
        ]
    report_text = "\n".join(lines)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report_text)
    log(f"  Saved report  -> {OUT_REPORT}")
    print()
    print(report_text.encode("ascii", "replace").decode("ascii"))


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print()
    log("=" * 60)
    log("Phase 4 -- Baseline Anomaly Detection")
    log("=" * 60)

    log("\n[1/4] Loading data ...")
    df, arts, pipe = load_data()

    log("\n[2/4] Building evaluation dataset ...")
    X_normal, X_test, y_test, ft_test, ts_test = build_eval_dataset(df, pipe, arts)

    log(f"\n[3/4] Running 3 detectors ...")
    results = run_all_detectors(X_normal, X_test, y_test, ft_test)

    log("\n[4/4] Saving results ...")
    summary_df, per_fault_df = build_evaluation_table(results)
    save_results(results, summary_df, per_fault_df)

    log("Phase 4 complete. [OK]")


if __name__ == "__main__":
    main()
