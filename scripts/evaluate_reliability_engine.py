"""
scripts/evaluate_reliability_engine.py
======================================
Phase 6 — Sensor Reliability & Health Scoring Engine Evaluation.

Demonstrates:
  1. Baseline fitting from healthy training sensor windows.
  2. Per-sensor evaluation on clean and fault-injected windows (all 8 fault types).
  3. Diagnostic sub-scores breakdown (Drift, Noise, Consistency, Missing, ML Anomaly).
  4. Automatic state classification (HEALTHY, DRIFTING, NOISY, STUCK, MISSING, INTERMITTENT, FAILED).
  5. Human-readable explainable reason generation.
  6. Generates evaluation tables and report saved to `data/processed/`.

Usage:
    python scripts/evaluate_reliability_engine.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd

# Paths and imports
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_BACKEND = os.path.join(_ROOT, "backend")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _BACKEND)

from app.ml.preprocessing.pipeline import (
    SensorPreprocessingPipeline,
    PipelineArtifacts,
    SENSOR_VALID_RANGES,
)
from app.ml.anomaly_detection.lstm_autoencoder import LSTMAutoencoderDetector
from app.ml.reliability import SensorReliabilityEngine
from scripts.run_phase4_baselines import load_data, build_eval_dataset

DATA_DIR = os.path.join(_ROOT, "data")
PROC_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(_ROOT, "models", "anomaly")
MODEL_CKPT = os.path.join(MODELS_DIR, "lstm_autoencoder.pt")

OUT_SUMMARY = os.path.join(PROC_DIR, "phase6_reliability_summary.csv")
OUT_REPORT = os.path.join(PROC_DIR, "phase6_reliability_report.txt")


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}".encode("ascii", "replace").decode("ascii"))


def main():
    print()
    log("=" * 75)
    log("Phase 6 -- Sensor Reliability & Health Scoring Engine Benchmark")
    log("=" * 75)

    # 1. Load Data
    log("\n[1/4] Loading sensor dataset and pipeline artifacts ...")
    df, arts, pipe = load_data()
    X_normal, X_test, y_test, ft_test, ts_test = build_eval_dataset(df, pipe, arts)

    # 2. Fit Engine Baselines
    log("\n[2/4] Initializing and fitting SensorReliabilityEngine baselines ...")
    engine = SensorReliabilityEngine()
    engine.fit_baselines(
        df_normal=X_normal,
        sensor_ids=arts.sensor_cols,
        valid_ranges=SENSOR_VALID_RANGES,
    )
    log(f"  Fitted baselines for {len(engine.baselines)} sensors: {arts.sensor_cols}")

    # Load LSTM Autoencoder for reconstruction error if available
    lstm_detector = None
    if os.path.exists(MODEL_CKPT):
        log(f"  Loading trained LSTM Autoencoder from {MODEL_CKPT} ...")
        lstm_detector = LSTMAutoencoderDetector.load(MODEL_CKPT, device="cpu")

    # 3. Evaluate Reliability Across All Fault Types
    log("\n[3/4] Evaluating sensor windows across healthy and fault conditions ...")
    unique_conditions = ["NORMAL"] + sorted(list(set(ft for ft in ft_test if ft and ft != "NORMAL")))

    results = []
    sample_explanations = []

    for condition in unique_conditions:
        mask = (ft_test == condition) if condition != "NORMAL" else (y_test == 0)
        idx_matches = np.where(mask)[0]

        if len(idx_matches) == 0:
            continue

        # Evaluate sample windows for this condition
        cond_windows = X_test[idx_matches]

        # Compute per-sensor reconstruction errors if LSTM available
        if lstm_detector is not None:
            recon_errors = lstm_detector.score_per_sensor(cond_windows)
        else:
            recon_errors = np.zeros((len(cond_windows), len(arts.sensor_cols)))

        # Map fault condition to the sensor it was injected on
        fault_to_sensor = {
            "NORMAL": arts.sensor_cols[0],
            "DRIFT": arts.sensor_cols[0],             # TEMP_01
            "BIAS": arts.sensor_cols[1],              # PRESSURE_01
            "NOISE": arts.sensor_cols[2],             # VIBRATION_01
            "STUCK": arts.sensor_cols[3],             # CURRENT_01
            "MISSING": arts.sensor_cols[4],           # FLOW_01
            "INTERMITTENT": arts.sensor_cols[5],      # HUMIDITY_01
            "SPIKE": arts.sensor_cols[0],             # TEMP_01
            "SAMPLING_FAILURE": arts.sensor_cols[1],  # PRESSURE_01
        }
        target_sensor = fault_to_sensor.get(condition, arts.sensor_cols[0])
        sensor_idx = arts.sensor_cols.index(target_sensor)

        health_scores = []
        drift_scores = []
        noise_scores = []
        consistency_scores = []
        missing_scores = []
        statuses = []

        for i in range(min(50, len(cond_windows))):
            sig_win = cond_windows[i, :, sensor_idx]
            recon_e = float(recon_errors[i, sensor_idx])

            eval_res = engine.evaluate_sensor(
                sensor_id=target_sensor,
                signal_window=sig_win,
                reconstruction_error=recon_e,
            )

            health_scores.append(eval_res.health_score)
            drift_scores.append(eval_res.drift_score)
            noise_scores.append(eval_res.noise_score)
            consistency_scores.append(eval_res.consistency_score)
            missing_scores.append(eval_res.missing_data_score)
            statuses.append(eval_res.status.value)

            if i == 0 and len(sample_explanations) < 8:
                sample_explanations.append((condition, eval_res.status.value, eval_res.health_score, eval_res.reason))

        # Majority status
        vals, counts = np.unique(statuses, return_counts=True)
        majority_status = vals[np.argmax(counts)]

        results.append({
            "Condition": condition,
            "Target Sensor": target_sensor,
            "Health Score (0-100)": round(float(np.mean(health_scores)), 1),
            "Drift Score": round(float(np.mean(drift_scores)), 1),
            "Noise Score": round(float(np.mean(noise_scores)), 1),
            "Consistency Score": round(float(np.mean(consistency_scores)), 1),
            "Missing Score": round(float(np.mean(missing_scores)), 1),
            "Inferred Status": majority_status,
        })

    summary_df = pd.DataFrame(results)

    # 4. Save and Report
    log("\n[4/4] Generating reports and saving artifacts ...")
    summary_df.to_csv(OUT_SUMMARY, index=False)

    lines = [
        "=" * 78,
        "SensorShield Phase 6 -- Sensor Reliability & Health Scoring Engine Report",
        f"Generated : {datetime.now().isoformat()}",
        "=" * 78,
        "",
        "RELIABILITY SCORES & HEALTH STATE PER CONDITION",
        "-" * 78,
        summary_df.to_string(index=False),
        "",
        "SAMPLE EXPLAINABLE DIAGNOSTIC REASONS",
        "-" * 78,
    ]
    for cond, st, score, r in sample_explanations:
        lines.append(f"[{cond:<18}] -> Status: {st:<12} (Health: {score:5.1f}/100) | {r}")

    lines.append("")
    report_text = "\n".join(lines)

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report_text)

    log(f"  Saved summary table -> {OUT_SUMMARY}")
    log(f"  Saved report        -> {OUT_REPORT}")
    print()
    print(report_text.encode("ascii", "replace").decode("ascii"))
    log("Phase 6 complete. [OK]")


if __name__ == "__main__":
    main()
