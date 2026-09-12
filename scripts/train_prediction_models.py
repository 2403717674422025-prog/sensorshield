"""
scripts/train_prediction_models.py
====================================
Phase 2 (backend completion): Train and persist the DownstreamPredictor.

Uses the processed numpy arrays already produced by the preprocessing
pipeline (X_train/val/test.npy, y_train/val/test.npy).

Usage (from repo root):
    python scripts/train_prediction_models.py

Outputs (both files are required by DownstreamPredictor.load()):
    models/prediction/xgboost_model.pkl
    models/prediction/lstm_model.pt

Also writes a brief validation report to:
    data/processed/phase2_prediction_training_report.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "backend"))

from app.ml.prediction import DownstreamPredictor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR    = _ROOT / "data" / "processed"
MODEL_DIR   = _ROOT / "models" / "prediction"
REPORT_PATH = DATA_DIR / "phase2_prediction_training_report.txt"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_arrays() -> tuple:
    required = [
        "X_train.npy", "y_train.npy",
        "X_val.npy",   "y_val.npy",
        "X_test.npy",  "y_test.npy",
    ]
    missing = [n for n in required if not (DATA_DIR / n).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing preprocessed arrays: {', '.join(missing)}\n"
            "Run the preprocessing pipeline first."
        )
    arrays = tuple(np.load(DATA_DIR / n) for n in required)
    return arrays


def describe(name: str, arr: np.ndarray) -> str:
    vals, cnts = np.unique(arr, return_counts=True)
    dist = ", ".join(f"{v}={c}" for v, c in zip(vals.tolist(), cnts.tolist()))
    return f"  {name}: shape={arr.shape}  labels=[{dist}]"


def select_eval_split(
    X_val, y_val, X_test, y_test
) -> tuple[np.ndarray, np.ndarray]:
    """Return a held-out split that contains both classes."""
    if np.unique(y_test).size == 2:
        return X_test, y_test
    X_eval = np.concatenate([X_val, X_test], axis=0)
    y_eval = np.concatenate([y_val, y_test], axis=0)
    return X_eval, y_eval


def main() -> None:
    print("=" * 60)
    print("SensorShield — Phase 2: Train Downstream Prediction Models")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load processed splits
    # ------------------------------------------------------------------
    print("\n[1/5] Loading processed arrays...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_arrays()
    print(describe("X_train", y_train))
    print(describe("X_val  ", y_val))
    print(describe("X_test ", y_test))

    n_sensors = X_train.shape[2]
    print(f"\n  Windows: {X_train.shape}  (N, seq_len={X_train.shape[1]}, sensors={n_sensors})")

    # ------------------------------------------------------------------
    # 2. Validate label distribution — both classes required for training
    # ------------------------------------------------------------------
    print("\n[2/5] Validating labels...")
    unique_labels = np.unique(y_train)
    if not np.array_equal(unique_labels, np.array([0, 1], dtype=unique_labels.dtype)):
        raise ValueError(
            f"Training labels must contain exactly [0, 1]. Got: {unique_labels.tolist()}"
        )
    pos_rate = float(y_train.mean())
    print(f"  Class balance — normal: {1 - pos_rate:.1%}  failure: {pos_rate:.1%}")

    # ------------------------------------------------------------------
    # 3. Instantiate and fit the DownstreamPredictor
    # ------------------------------------------------------------------
    print("\n[3/5] Training DownstreamPredictor (XGBoost + LSTM)...")
    print("  XGBoost: 150 estimators, max_depth=5")
    print("  LSTM:    15 epochs, hidden_dim=64, MC-Dropout")

    predictor = DownstreamPredictor(device="cpu")
    predictor.fit(X_train, y_train, lstm_epochs=15, verbose=True)
    print("  Training complete.")

    # ------------------------------------------------------------------
    # 4. Save models
    # ------------------------------------------------------------------
    print(f"\n[4/5] Saving models to {MODEL_DIR} ...")
    predictor.save(str(MODEL_DIR))
    xgb_path  = MODEL_DIR / "xgboost_model.pkl"
    lstm_path = MODEL_DIR / "lstm_model.pt"
    print(f"  xgboost_model.pkl  : {xgb_path.stat().st_size / 1024:.1f} KB")
    print(f"  lstm_model.pt      : {lstm_path.stat().st_size / 1024:.1f} KB")

    # ------------------------------------------------------------------
    # 5. Validate load + inference roundtrip
    # ------------------------------------------------------------------
    print("\n[5/5] Validation — load & infer on held-out set...")
    loaded = DownstreamPredictor.load(str(MODEL_DIR), device="cpu")
    X_eval, y_eval = select_eval_split(X_val, y_val, X_test, y_test)

    xgb_metrics  = loaded.evaluate(y_eval, X_eval, use_model="xgboost")
    lstm_metrics = loaded.evaluate(y_eval, X_eval, use_model="lstm")

    single_output = loaded.predict_window(X_eval[0], use_model="lstm")
    print(f"  Single-window inference OK: prob={single_output.failure_probability:.4f} "
          f"uncertainty={single_output.model_uncertainty:.4f}")

    report_lines = [
        "SensorShield — Phase 2 Downstream Prediction Training Report",
        "=" * 60,
        "",
        "Dataset",
        f"  X_train  : {X_train.shape}",
        f"  X_eval   : {X_eval.shape}",
        f"  Pos rate : {pos_rate:.3f}",
        "",
        "XGBoost Evaluation",
        *[f"  {k}: {v}" for k, v in xgb_metrics.items()],
        "",
        "LSTM Evaluation (deterministic)",
        *[f"  {k}: {v}" for k, v in lstm_metrics.items()],
        "",
        "Saved Files",
        f"  {xgb_path}",
        f"  {lstm_path}",
        "",
        "Load roundtrip: OK",
    ]

    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\n  XGBoost metrics:")
    for k, v in xgb_metrics.items():
        print(f"    {k}: {v}")
    print("\n  LSTM metrics:")
    for k, v in lstm_metrics.items():
        print(f"    {k}: {v}")

    print(f"\n  Report saved -> {REPORT_PATH}")
    print("\n[OK] Phase 2 complete. Prediction models are ready.")
    print(f"     Load with: DownstreamPredictor.load('{MODEL_DIR}')")


if __name__ == "__main__":
    main()
