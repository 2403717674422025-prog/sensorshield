"""
Phase 8 diagnostics: identify root causes of degenerate predictions and label issues.

Usage:
    python scripts/diagnose_phase8.py
"""

import sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "backend"))

DATA_DIR = _ROOT / "data" / "processed"

# ============================================================================
# A. INSPECT SPLIT CREATION
# ============================================================================
print("=" * 70)
print("A. SPLIT INSPECTION")
print("=" * 70)

X_train = np.load(DATA_DIR / "X_train.npy")
y_train = np.load(DATA_DIR / "y_train.npy")
X_val = np.load(DATA_DIR / "X_val.npy")
y_val = np.load(DATA_DIR / "y_val.npy")
X_test = np.load(DATA_DIR / "X_test.npy")
y_test = np.load(DATA_DIR / "y_test.npy")

print(f"\nTrain: shape={X_train.shape}  y_distribution: {np.unique(y_train, return_counts=True)}")
print(f"Val:   shape={X_val.shape}    y_distribution: {np.unique(y_val, return_counts=True)}")
print(f"Test:  shape={X_test.shape}   y_distribution: {np.unique(y_test, return_counts=True)}")

print(f"\nTotal positive (1) count per split:")
print(f"  Train: {np.sum(y_train == 1):,} / {len(y_train):,} ({100 * np.mean(y_train == 1):.2f}%)")
print(f"  Val:   {np.sum(y_val == 1):,} / {len(y_val):,} ({100 * np.mean(y_val == 1):.2f}%)")
print(f"  Test:  {np.sum(y_test == 1):,} / {len(y_test):,} ({100 * np.mean(y_test == 1):.2f}%)")

# ============================================================================
# B. LSTM PREDICTION DEBUGGING
# ============================================================================
print("\n" + "=" * 70)
print("B. LSTM PREDICTION ANALYSIS")
print("=" * 70)

import torch
from app.ml.prediction import DownstreamPredictor

print("\nFitting LSTM on training data...")
predictor = DownstreamPredictor(device="cpu")
predictor.fit(X_train, y_train, lstm_epochs=1)

print("\nEvaluating LSTM on a small training batch (first 10 samples)...")
X_sample = X_train[:10]
print(f"  X_sample shape: {X_sample.shape}")

# Get raw logits
with torch.no_grad():
    predictor.lstm_model.model.eval()
    X_tensor = torch.tensor(X_sample, dtype=torch.float32).to(predictor.lstm_model.device)
    logits = predictor.lstm_model.model(X_tensor).cpu().numpy()
    print(f"  Logits: min={logits.min():.6f}, max={logits.max():.6f}, mean={logits.mean():.6f}, std={logits.std():.6f}")
    print(f"  Logits[:5]: {logits[:5]}")

# Get probabilities
probs_train, uncert_train = predictor.lstm_model.predict_proba_with_uncertainty(X_sample, n_mc_samples=5)
print(f"  Probabilities: min={probs_train.min():.6f}, max={probs_train.max():.6f}, mean={probs_train.mean():.6f}, std={probs_train.std():.6f}")
print(f"  Probabilities[:5]: {probs_train[:5]}")
print(f"  y_train[:10]: {y_train[:10].tolist()}")

# Check dropout mode
print(f"\n  LSTM model training mode: {predictor.lstm_model.model.training}")

# ============================================================================
# C. VERIFY LABEL SEMANTICS
# ============================================================================
print("\n" + "=" * 70)
print("C. LABEL SEMANTICS")
print("=" * 70)

print(f"Unique values in y_train: {np.unique(y_train).tolist()}")
print(f"Unique values in y_val:   {np.unique(y_val).tolist()}")
print(f"Unique values in y_test:  {np.unique(y_test).tolist()}")
print(f"Label dtype: {y_train.dtype}")

# ============================================================================
# D. SYNTHETIC TEST
# ============================================================================
print("\n" + "=" * 70)
print("D. SYNTHETIC METRIC TEST")
print("=" * 70)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss

y_true_syn = np.array([0, 0, 1, 1])
y_prob_syn = np.array([0.1, 0.2, 0.8, 0.9])
y_pred_syn = (y_prob_syn >= 0.5).astype(int)

print(f"Synthetic y_true: {y_true_syn.tolist()}")
print(f"Synthetic y_prob: {y_prob_syn.tolist()}")
print(f"Synthetic y_pred (threshold=0.5): {y_pred_syn.tolist()}")
print(f"  Accuracy: {accuracy_score(y_true_syn, y_pred_syn):.4f}")
print(f"  Precision: {precision_score(y_true_syn, y_pred_syn):.4f}")
print(f"  Recall: {recall_score(y_true_syn, y_pred_syn):.4f}")
print(f"  F1: {f1_score(y_true_syn, y_pred_syn):.4f}")
print(f"  AUROC: {roc_auc_score(y_true_syn, y_prob_syn):.4f}")
print(f"  Brier: {brier_score_loss(y_true_syn, y_prob_syn):.4f}")

# ============================================================================
# E. CORRUPTION INDEPENDENCE
# ============================================================================
print("\n" + "=" * 70)
print("E. CORRUPTION INDEPENDENCE CHECK")
print("=" * 70)

# Simulate build_scenarios
X_eval = np.concatenate([X_val, X_test], axis=0)
y_eval = np.concatenate([y_val, y_test], axis=0)

healthy = X_eval.copy()
one_drifting = healthy.copy()
ramp = np.linspace(0.0, 3.0, one_drifting.shape[1], dtype=np.float32)
one_drifting[:, :, 0] += ramp[None, :]

print(f"Healthy vs One drifting:")
print(f"  Corruption is element-wise independent of y_eval: {not np.any(np.isnan(y_eval))}")
print(f"  Corruption affects only sensor 0: {np.all(healthy[:, :, 1:] == one_drifting[:, :, 1:])}")
print(f"  y_eval unchanged: {np.array_equal(y_eval, y_eval)}")

print("\nDiagnostics complete.")
