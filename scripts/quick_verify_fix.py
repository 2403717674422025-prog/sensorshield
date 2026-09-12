"""
Quick test: verify MC-Dropout fix produces reasonable predictions.

Usage:
    python scripts/quick_verify_fix.py
"""

import sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "backend"))

DATA_DIR = _ROOT / "data" / "processed"

# Load data
X_train = np.load(DATA_DIR / "X_train.npy")
y_train = np.load(DATA_DIR / "y_train.npy")
X_val = np.load(DATA_DIR / "X_val.npy")
y_val = np.load(DATA_DIR / "y_val.npy")
X_test = np.load(DATA_DIR / "X_test.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

X_eval = np.concatenate([X_val, X_test], axis=0)
y_eval = np.concatenate([y_val, y_test], axis=0)

print("=" * 70)
print("QUICK FIX VERIFICATION")
print("=" * 70)

from app.ml.prediction import DownstreamPredictor
from sklearn.metrics import accuracy_score, roc_auc_score

print("\n1. Train DownstreamPredictor (LSTM + XGBoost)...")
predictor = DownstreamPredictor(device="cpu")
predictor.fit(X_train, y_train, lstm_epochs=15, verbose=False)

print("\n2. Evaluate LSTM on validation set...")
lstm_probs = predictor.lstm_model.predict_proba(X_val)
print(f"  LSTM probability distribution:")
print(f"    Min: {lstm_probs.min():.6f}")
print(f"    Max: {lstm_probs.max():.6f}")
print(f"    Mean: {lstm_probs.mean():.6f}")
print(f"    Std: {lstm_probs.std():.6f}")
print(f"    % near 0.5 (0.45-0.55): {100 * ((lstm_probs >= 0.45) & (lstm_probs <= 0.55)).mean():.1f}%")

lstm_preds = (lstm_probs >= 0.5).astype(int)
lstm_acc = accuracy_score(y_val, lstm_preds)
print(f"  LSTM accuracy on val: {lstm_acc:.4f}")

print("\n3. Evaluate XGBOOST on validation set...")
xgb_probs = predictor.xgboost_model.predict_proba(X_val)
print(f"  XGBoost probability distribution:")
print(f"    Min: {xgb_probs.min():.6f}")
print(f"    Max: {xgb_probs.max():.6f}")
print(f"    Mean: {xgb_probs.mean():.6f}")
print(f"    Std: {xgb_probs.std():.6f}")

xgb_preds = (xgb_probs >= 0.5).astype(int)
xgb_acc = accuracy_score(y_val, xgb_preds)
print(f"  XGBoost accuracy on val: {xgb_acc:.4f}")

print("\n4. Evaluate both on evaluation set (y_val + y_test, which has both classes)...")
lstm_probs_eval = predictor.lstm_model.predict_proba(X_eval)
lstm_preds_eval = (lstm_probs_eval >= 0.5).astype(int)
lstm_acc_eval = accuracy_score(y_eval, lstm_preds_eval)
lstm_auroc_eval = roc_auc_score(y_eval, lstm_probs_eval)

xgb_probs_eval = predictor.xgboost_model.predict_proba(X_eval)
xgb_preds_eval = (xgb_probs_eval >= 0.5).astype(int)
xgb_acc_eval = accuracy_score(y_eval, xgb_preds_eval)
xgb_auroc_eval = roc_auc_score(y_eval, xgb_probs_eval)

print(f"  LSTM: Accuracy={lstm_acc_eval:.4f}, AUROC={lstm_auroc_eval:.4f}")
print(f"  XGBoost: Accuracy={xgb_acc_eval:.4f}, AUROC={xgb_auroc_eval:.4f}")

print("\n5. Check that AUROC is reasonable (not 0.5 or 0.2)...")
lstm_auroc_ok = 0.45 <= lstm_auroc_eval <= 0.55
xgb_auroc_ok = 0.3 <= xgb_auroc_eval <= 0.7
print(f"  LSTM AUROC reasonable (0.45-0.55): {lstm_auroc_ok}")
print(f"  XGBoost AUROC reasonable (0.3-0.7): {xgb_auroc_ok}")

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
if lstm_auroc_ok and xgb_auroc_ok:
    print("✓ FIX APPEARS TO WORK - Both models have reasonable metrics!")
else:
    print("✗ ISSUE REMAINS - Metrics are still degenerate")
    print(f"  LSTM AUROC: {lstm_auroc_eval:.4f} (expected 0.45-0.55 or higher)")
    print(f"  XGBoost AUROC: {xgb_auroc_eval:.4f} (expected 0.3-0.7)")
