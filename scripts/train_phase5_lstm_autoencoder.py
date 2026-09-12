"""
scripts/train_phase5_lstm_autoencoder.py
=========================================
Phase 5 — Train Multivariate LSTM Autoencoder & Compare Against Baselines.

Workflow
--------
1. Load sensor data & Phase 2 preprocessing artifacts.
2. Prepare clean training split (NORMAL windows only).
3. Build evaluation test split with all 8 injected fault types.
4. Train LSTMAutoencoderDetector on clean normal windows.
5. Save model weights to `models/anomaly/lstm_autoencoder.pt`.
6. Evaluate LSTM Autoencoder on clean vs. corrupted test stream.
7. Combine with Phase 4 baseline results to produce comparative evaluation table.
8. Save results & human-readable report in `data/processed/`.

Usage:
    python scripts/train_phase5_lstm_autoencoder.py
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
import pickle

import numpy as np
import pandas as pd
import torch

# Make backend and scripts importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_BACKEND = os.path.join(_ROOT, "backend")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _BACKEND)

from app.ml.preprocessing.pipeline import (
    SensorPreprocessingPipeline,
    PipelineArtifacts,
)
from app.ml.anomaly_detection import (
    RuleBasedDetector,
    IsolationForestDetector,
    OneClassSVMDetector,
    LSTMAutoencoderDetector,
    evaluate_detector,
    build_evaluation_table,
)
from scripts.run_phase4_baselines import (
    load_data,
    build_eval_dataset,
    STEP_SIZE_EVAL,
)

# Paths
DATA_DIR = os.path.join(_ROOT, "data")
PROC_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(_ROOT, "models", "anomaly")
MODEL_CKPT = os.path.join(MODELS_DIR, "lstm_autoencoder.pt")

OUT_SUMM = os.path.join(PROC_DIR, "phase5_lstm_summary.csv")
OUT_PFAULT = os.path.join(PROC_DIR, "phase5_per_fault_metrics.csv")
OUT_COMPARISON = os.path.join(PROC_DIR, "phase5_model_comparison.csv")
OUT_REPORT = os.path.join(PROC_DIR, "phase5_report.txt")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}".encode("ascii", "replace").decode("ascii"))


def main():
    print()
    log("=" * 70)
    log("Phase 5 -- Multivariate LSTM Autoencoder Anomaly Detection")
    log("=" * 70)

    # 1. Load Data & Prepare Splits
    log("\n[1/4] Loading data and building evaluation dataset ...")
    df, arts, pipe = load_data()
    X_normal, X_test, y_test, ft_test, ts_test = build_eval_dataset(df, pipe, arts)
    log(f"  Training clean normal windows: {X_normal.shape}")
    log(f"  Test windows (clean + 8 faults): {X_test.shape} (anomaly rate: {y_test.mean():.2%})")

    # 2. Train LSTM Autoencoder
    log("\n[2/4] Training LSTM Autoencoder on clean NORMAL windows ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"  Using compute device: {device}")

    lstm_detector = LSTMAutoencoderDetector(
        hidden_dim=64,
        latent_dim=16,
        dropout=0.1,
        lr=0.002,
        weight_decay=1e-5,
        percentile=98.0,
        device=device,
    )

    t0 = time.perf_counter()
    lstm_detector.fit(
        X_normal,
        val_split=0.15,
        epochs=30,
        batch_size=128,
        patience=6,
        verbose=True,
    )
    fit_duration = time.perf_counter() - t0
    log(f"  Training finished in {fit_duration:.1f}s")

    # 3. Save Model Checkpoint
    log(f"\n[3/4] Saving model checkpoint to {MODEL_CKPT} ...")
    lstm_detector.save(MODEL_CKPT)

    # 4. Comprehensive Evaluation & Comparison with Baselines
    log("\n[4/4] Evaluating all models (Baselines + LSTM Autoencoder) ...")
    all_detectors = [
        RuleBasedDetector(z_threshold=3.0, std_multiplier=3.0, nan_threshold=0.2),
        IsolationForestDetector(n_estimators=200, contamination=0.01, random_state=42),
        OneClassSVMDetector(nu=0.01, gamma="scale"),
        lstm_detector,
    ]

    all_results = []
    for det in all_detectors:
        log(f"  Evaluating {det.name} ...")
        res = evaluate_detector(
            detector=det,
            X_normal_train=X_normal,
            X_test=X_test,
            y_test=y_test,
            fault_types=ft_test,
            window_step_s=60 * STEP_SIZE_EVAL,
        )
        log(f"    F1={res.overall_f1:.3f}  AUROC={res.overall_auroc:.3f}  "
            f"AP={res.overall_avg_prec:.3f}  fit={res.fit_time_s:.1f}s  score={res.score_time_s:.2f}s")
        all_results.append(res)

    summary_df, per_fault_df = build_evaluation_table(all_results)

    # Save Tables
    summary_df.to_csv(OUT_COMPARISON)
    per_fault_df.to_csv(OUT_PFAULT, index=False)

    # Generate Report
    lines = [
        "=" * 75,
        "SensorShield Phase 5 -- LSTM Autoencoder vs. Baselines Evaluation Report",
        f"Generated : {datetime.now().isoformat()}",
        f"Checkpoint: {MODEL_CKPT}",
        "=" * 75,
        "",
        "OVERALL MODEL BENCHMARK COMPARISON",
        "-" * 75,
        summary_df.to_string(),
        "",
        "PER-FAULT-TYPE BREAKDOWN",
        "-" * 75,
        per_fault_df.to_string(index=False),
        "",
    ]
    report_text = "\n".join(lines)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report_text)

    log(f"  Saved comparison table -> {OUT_COMPARISON}")
    log(f"  Saved per-fault metrics -> {OUT_PFAULT}")
    log(f"  Saved full report      -> {OUT_REPORT}")
    print()
    print(report_text.encode("ascii", "replace").decode("ascii"))
    log("Phase 5 complete. [OK]")


if __name__ == "__main__":
    main()
