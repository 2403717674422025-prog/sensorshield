"""
Test MC-Dropout inference: does it collapse logits back to ~0.5?

Usage:
    python scripts/test_mc_dropout_issue.py
"""

import sys
from pathlib import Path
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "backend"))

DATA_DIR = _ROOT / "data" / "processed"

# Load data
X_train = np.load(DATA_DIR / "X_train.npy")
y_train = np.load(DATA_DIR / "y_train.npy")
X_val = np.load(DATA_DIR / "X_val.npy")
y_val = np.load(DATA_DIR / "y_val.npy")

X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
X_val = X_val.astype(np.float32)
y_val = y_val.astype(np.float32)

print("=" * 70)
print("MC-DROPOUT INFERENCE BUG TEST")
print("=" * 70)

from app.ml.prediction.models import LSTMFailureClassifier

print("\n1. Create and train LSTM for 15 epochs...")
classifier = LSTMFailureClassifier(device="cpu", lr=0.002)
classifier.fit(X_train, y_train, epochs=15, verbose=False)

print("\n2. Evaluate on validation set (eval mode, no dropout)...")
classifier.model.eval()
with torch.no_grad():
    X_val_tensor = torch.tensor(X_val[:1000], dtype=torch.float32).to(classifier.device)
    logits_eval = classifier.model(X_val_tensor).cpu().numpy()
    probs_eval = 1.0 / (1.0 + np.exp(-logits_eval))
    print(f"  Logits (eval mode): min={logits_eval.min():.6f}, max={logits_eval.max():.6f}, mean={logits_eval.mean():.6f}")
    print(f"  Probabilities (eval mode): min={probs_eval.min():.6f}, max={probs_eval.max():.6f}, mean={probs_eval.mean():.6f}")

print("\n3. Evaluate on validation set (train mode WITH dropout)...")
classifier.model.train()
with torch.no_grad():
    X_val_tensor = torch.tensor(X_val[:1000], dtype=torch.float32).to(classifier.device)
    logits_train = classifier.model(X_val_tensor).cpu().numpy()
    probs_train = 1.0 / (1.0 + np.exp(-logits_train))
    print(f"  Logits (train mode, 1 pass): min={logits_train.min():.6f}, max={logits_train.max():.6f}, mean={logits_train.mean():.6f}")
    print(f"  Probabilities (train mode, 1 pass): min={probs_train.min():.6f}, max={probs_train.max():.6f}, mean={probs_train.mean():.6f}")

print("\n4. MC-Dropout inference (train mode, multiple passes)...")
probs_mc, uncert_mc = classifier.predict_proba_with_uncertainty(X_val[:1000], n_mc_samples=15)
print(f"  MC-Dropout probabilities: min={probs_mc.min():.6f}, max={probs_mc.max():.6f}, mean={probs_mc.mean():.6f}")
print(f"  MC-Dropout uncertainty: min={uncert_mc.min():.6f}, max={uncert_mc.max():.6f}, mean={uncert_mc.mean():.6f}")

print("\n5. Check if MC-Dropout logits are different from single pass...")
all_mc_logits = []
classifier.model.train()
with torch.no_grad():
    X_val_tensor = torch.tensor(X_val[:1000], dtype=torch.float32).to(classifier.device)
    for pass_idx in range(3):
        logits = classifier.model(X_val_tensor).cpu().numpy()
        all_mc_logits.append(logits)
        probs = 1.0 / (1.0 + np.exp(-logits))
        print(f"  MC Pass {pass_idx+1}: logits_mean={logits.mean():.6f}, probs_mean={probs.mean():.6f}")

print("\n6. Check model.training state during predict_proba_with_uncertainty...")
print(f"  Before predict_proba_with_uncertainty: model.training={classifier.model.training}")
probs_test, _ = classifier.predict_proba_with_uncertainty(X_val[:100], n_mc_samples=5)
print(f"  After predict_proba_with_uncertainty: model.training={classifier.model.training}")
print(f"  Result probabilities: min={probs_test.min():.6f}, max={probs_test.max():.6f}, mean={probs_test.mean():.6f}")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
