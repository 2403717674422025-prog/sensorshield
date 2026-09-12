"""Focused, reproducible Phase 8 LSTM learning diagnosis.

This diagnostic does not alter the saved data, labels, thresholds, or Phase 8
model configuration.  It writes a report to data/processed for review.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.prediction.models import LSTMFailureClassifier
from scripts.run_phase8_downstream import build_scenarios

DATA = ROOT / "data" / "processed"
REPORT = DATA / "phase8_lstm_diagnosis_report.txt"


def deterministic_probabilities(model: LSTMFailureClassifier, X: np.ndarray) -> np.ndarray:
    """Inference-only probabilities, explicitly with dropout disabled."""
    model.model.eval()
    values = np.nan_to_num(X, nan=0.0).astype(np.float32)
    output = []
    with torch.no_grad():
        for start in range(0, len(values), 128):
            batch = torch.tensor(values[start:start + 128]).to(model.device)
            output.append(torch.sigmoid(model.model(batch)).cpu().numpy())
    return np.concatenate(output)


def profile(name: str, model: LSTMFailureClassifier, X: np.ndarray, y: np.ndarray) -> list[str]:
    probs = deterministic_probabilities(model, X)
    pred = (probs >= 0.5).astype(int)
    return [
        name,
        f"  probability min/max/mean/std = {probs.min():.6f} / {probs.max():.6f} / {probs.mean():.6f} / {probs.std():.6f}",
        f"  predicted_positive_rate={pred.mean():.6f}; true_positive_rate={y.mean():.6f}",
        f"  predicted_class_distribution={np.unique(pred, return_counts=True)}",
        f"  AUROC={roc_auc_score(y, probs):.6f}; Brier={brier_score_loss(y, probs):.6f}",
    ]


def fmt_history(history: list[dict[str, float]]) -> list[str]:
    lines = ["epoch,train_loss,train_accuracy,train_auroc,val_loss,val_accuracy,val_auroc"]
    for row in history:
        lines.append(
            f"{int(row['epoch'])},{row['train_loss']:.6f},{row.get('train_accuracy', float('nan')):.6f},"
            f"{row.get('train_auroc', float('nan')):.6f},{row.get('val_loss', float('nan')):.6f},"
            f"{row.get('val_accuracy', float('nan')):.6f},{row.get('val_auroc', float('nan')):.6f}"
        )
    return lines


def main() -> None:
    print("Starting Phase 8 LSTM learning diagnosis", flush=True)
    torch.manual_seed(42)
    np.random.seed(42)
    X_train, y_train = np.load(DATA / "X_train.npy"), np.load(DATA / "y_train.npy")
    X_val, y_val = np.load(DATA / "X_val.npy"), np.load(DATA / "y_val.npy")
    X_test, y_test = np.load(DATA / "X_test.npy"), np.load(DATA / "y_test.npy")
    X_final, y_final = np.concatenate([X_val, X_test]), np.concatenate([y_val, y_test])
    ratio = float((y_train == 0).sum() / (y_train == 1).sum())

    lines = [
        "SensorShield Phase 8 -- LSTM Learning Diagnosis",
        "",
        "CURRENT CONFIGURATION",
        "LSTM: input=6, hidden=64, layers=2, LSTM dropout=0.25; head=Linear(64,32), ReLU, Dropout(0.25), Linear(32,1)",
        "Output: one raw logit per window; sigmoid is applied only at inference.",
        "Loss: BCEWithLogitsLoss; targets: float32; optimizer: Adam; learning_rate=0.002; batch_size=128; epochs=15.",
        "Preprocessing: per-feature min-max scaling fit on train only; no additional LSTM scaler.",
        f"Shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}, final={X_final.shape}",
        f"Class imbalance: negatives={(y_train == 0).sum()}, positives={(y_train == 1).sum()}, neg/pos={ratio:.6f}.",
        f"Current documented BCEWithLogitsLoss pos_weight={ratio:.6f}; comparison configuration pos_weight=1.0.",
        "",
        "WINDOW / LABEL SEMANTICS",
        "Windows are 30 consecutive timesteps by 6 features, created separately inside temporal train/val/test partitions.",
        "A window label is 1 when any timestep in that window is abnormal; windows cannot cross split boundaries.",
    ]
    for label, X, y in (("negative", X_train, y_train), ("positive", X_train, y_train)):
        idx = int(np.flatnonzero(y == (label == "positive"))[0])
        lines.append(
            f"Example {label}: index={idx}, shape={X[idx].shape}, label={int(y[idx])}, "
            f"sensor_means={np.mean(X[idx], axis=0).round(4).tolist()}, "
            f"sensor_stds={np.std(X[idx], axis=0).round(4).tolist()}"
        )
    lines.append("")

    current = LSTMFailureClassifier(device="cpu")
    current.fit(X_train, y_train, epochs=15, validation_data=(X_val, y_val), verbose=False)
    lines.extend(["CURRENT-WEIGHT TRAINING HISTORY", *fmt_history(current.training_history), ""])
    lines.extend(profile("CURRENT-WEIGHT TRAIN", current, X_train, y_train))
    lines.extend(profile("CURRENT-WEIGHT VALIDATION", current, X_val, y_val))
    lines.extend(profile("CURRENT-WEIGHT FINAL EVALUATION", current, X_final, y_final))
    lines.append("")

    torch.manual_seed(42)
    comparison = LSTMFailureClassifier(device="cpu")
    comparison.fit(
        X_train, y_train, epochs=15, validation_data=(X_val, y_val),
        pos_weight_override=1.0, verbose=False,
    )
    lines.extend(["UNWEIGHTED COMPARISON (DIAGNOSTIC ONLY)", *fmt_history(comparison.training_history), ""])
    lines.extend(profile("UNWEIGHTED TRAIN", comparison, X_train, y_train))
    lines.extend(profile("UNWEIGHTED VALIDATION", comparison, X_val, y_val))
    lines.append("")

    rng = np.random.default_rng(42)
    neg = rng.choice(np.flatnonzero(y_train == 0), size=50, replace=False)
    pos = rng.choice(np.flatnonzero(y_train == 1), size=50, replace=False)
    tiny_idx = rng.permutation(np.concatenate([neg, pos]))
    torch.manual_seed(42)
    overfit = LSTMFailureClassifier(device="cpu")
    overfit.fit(X_train[tiny_idx], y_train[tiny_idx], epochs=100, batch_size=32, pos_weight_override=1.0)
    lines.extend(["BALANCED 100-SAMPLE OVERFIT SANITY TEST (DIAGNOSTIC ONLY)", *profile("OVERFIT SUBSET", overfit, X_train[tiny_idx], y_train[tiny_idx]), ""])

    scenarios = build_scenarios(X_final)
    clean = scenarios["Healthy"]
    corrupt = scenarios["Multiple sensors corrupted"]
    lines.extend(["MULTIPLE-CORRUPTION CLASS-CONDITIONAL INPUT CHECK"])
    for cls in (0, 1):
        mask = y_final == cls
        clean_means = clean[mask].mean(axis=(0, 1))
        corrupt_means = corrupt[mask].mean(axis=(0, 1))
        delta = corrupt_means - clean_means
        lines.append(
            f"class={cls}: clean_feature_means={clean_means.round(4).tolist()}; "
            f"corrupted_feature_means={corrupt_means.round(4).tolist()}; delta={delta.round(4).tolist()}"
        )
    lines.append("The seeded Gaussian corruption is sampled solely from X shape and does not access y.")

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Diagnosis saved -> {REPORT}")


if __name__ == "__main__":
    main()
