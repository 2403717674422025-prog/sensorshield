"""
Phase 8 experiment: downstream failure prediction under sensor corruption.

Usage from the repository root:
    python scripts/run_phase8_downstream.py

Outputs:
    data/processed/phase8_downstream_comparison.csv
    data/processed/phase8_downstream_report.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "backend"))

from app.ml.prediction import DownstreamPredictor
from app.ml.prediction.models import predictive_entropy

DATA_DIR = _ROOT / "data" / "processed"
COMPARISON_CSV = DATA_DIR / "phase8_downstream_comparison.csv"
REPORT_TXT = DATA_DIR / "phase8_downstream_report.txt"


def load_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = [
        "X_train.npy", "y_train.npy", "X_val.npy", "y_val.npy",
        "X_test.npy", "y_test.npy",
    ]
    missing = [name for name in required if not (DATA_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing processed arrays: {', '.join(missing)}. "
            "Run the dataset preparation scripts first."
        )
    return tuple(np.load(DATA_DIR / name) for name in required)


def describe_labels(name: str, labels: np.ndarray) -> str:
    values, counts = np.unique(labels, return_counts=True)
    distribution = ", ".join(
        f"{value}={count}" for value, count in zip(values.tolist(), counts.tolist())
    )
    return f"{name}: shape={labels.shape}, dtype={labels.dtype}, {distribution}"


def require_binary_labels(name: str, labels: np.ndarray) -> None:
    values = np.unique(labels)
    if not np.array_equal(values, np.array([0, 1], dtype=values.dtype)):
        raise ValueError(
            f"{name} must contain exactly binary labels [0, 1], got {values.tolist()}"
        )


def select_evaluation_split(
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Use a held-out split with both classes, without training on it."""
    if np.unique(y_test).size == 2:
        return X_test, y_test, "X_test"

    X_eval = np.concatenate([X_val, X_test], axis=0)
    y_eval = np.concatenate([y_val, y_test], axis=0)
    if np.unique(y_eval).size != 2:
        raise ValueError(
            "Phase 8 evaluation requires both labels, but the held-out "
            f"validation/test pool contains {np.unique(y_eval).tolist()}."
        )
    return X_eval, y_eval, "X_val+X_test"


def build_scenarios(X: np.ndarray) -> dict[str, np.ndarray]:
    """Create controlled test conditions with identical labels and windows."""
    healthy = np.array(X, copy=True)

    one_drifting = healthy.copy()
    ramp = np.linspace(0.0, 3.0, one_drifting.shape[1], dtype=np.float32)
    one_drifting[:, :, 0] += ramp[None, :]

    multiple_corrupted = healthy.copy()
    rng = np.random.default_rng(42)
    multiple_corrupted[:, :, : min(3, X.shape[2])] += rng.normal(
        0.0, 1.5, size=multiple_corrupted[:, :, : min(3, X.shape[2])].shape
    ).astype(np.float32)

    sensor_missing = healthy.copy()
    sensor_missing[:, :, 0] = np.nan

    unknown_fault = healthy.copy()
    spike_indices = np.arange(0, unknown_fault.shape[1], max(1, unknown_fault.shape[1] // 5))
    unknown_fault[:, spike_indices, -1] += 8.0

    return {
        "Healthy": healthy,
        "One sensor drifting": one_drifting,
        "Multiple sensors corrupted": multiple_corrupted,
        "Sensor missing": sensor_missing,
        "Unknown fault": unknown_fault,
    }


def summarize_input_changes(
    healthy: np.ndarray,
    scenarios: dict[str, np.ndarray],
) -> list[str]:
    lines = ["INPUT CHANGE SUMMARY", "--------------------"]
    for name, values in scenarios.items():
        changed = np.not_equal(
            np.nan_to_num(values, nan=0.0), np.nan_to_num(healthy, nan=0.0)
        )
        lines.append(
            f"{name}: mean={np.nanmean(values):.4f}, std={np.nanstd(values):.4f}, "
            f"min={np.nanmin(values):.4f}, max={np.nanmax(values):.4f}, "
            f"NaN%={100 * np.isnan(values).mean():.4f}, "
            f"changed%={100 * changed.mean():.4f}"
        )
    return lines


def evaluate_scenarios(
    predictor: DownstreamPredictor,
    y_true: np.ndarray,
    scenarios: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario_name, X_scenario in scenarios.items():
        if np.unique(y_true).size != 2:
            raise ValueError("Scenario evaluation requires both class labels.")
        for model_name in ("xgboost", "lstm"):
            metrics = predictor.evaluate(y_true, X_scenario, use_model=model_name)
            if model_name == "xgboost":
                probabilities = predictor.xgboost_model.predict_proba(X_scenario)
                # XGBoost produces one probability per sample; use binary
                # predictive entropy as its probability-based uncertainty.
                uncertainties = predictive_entropy(probabilities)
            else:
                probabilities, uncertainties = (
                    predictor.lstm_model.predict_proba_with_uncertainty(
                        X_scenario, n_mc_samples=10
                    )
                )
            confidence = np.abs(probabilities - 0.5) * 2.0
            predictions = (probabilities >= 0.5).astype(int)

            rows.append(
                {
                    "scenario": scenario_name,
                    "model": model_name,
                    "accuracy": metrics["Accuracy"],
                    "precision": metrics["Precision"],
                    "recall": metrics["Recall"],
                    "f1": metrics["F1"],
                    "auroc": metrics["AUROC"],
                    "brier_score": metrics["Brier Score"],
                    "mean_confidence": round(float(np.mean(confidence)), 4),
                    "mean_failure_probability": round(float(np.mean(probabilities)), 4),
                    "mean_uncertainty": round(float(np.mean(uncertainties)), 4),
                    "predicted_positive_rate": round(float(np.mean(predictions)), 4),
                }
            )
    return pd.DataFrame(rows)


def summarize_prediction_changes(comparison: pd.DataFrame) -> list[str]:
    lines = ["HEALTHY-RELATIVE SUMMARY", "------------------------"]
    for model_name in comparison["model"].unique():
        healthy = comparison[
            (comparison["model"] == model_name) &
            (comparison["scenario"] == "Healthy")
        ].iloc[0]
        for _, row in comparison[comparison["model"] == model_name].iterrows():
            if row["scenario"] == "Healthy":
                continue
            lines.append(
                f"{model_name} / {row['scenario']}: "
                f"accuracy_delta={row['accuracy'] - healthy['accuracy']:+.4f}, "
                f"brier_delta={row['brier_score'] - healthy['brier_score']:+.4f}, "
                f"confidence_delta={row['mean_confidence'] - healthy['mean_confidence']:+.4f}, "
                f"uncertainty_delta={row['mean_uncertainty'] - healthy['mean_uncertainty']:+.4f}"
            )
    return lines


def prediction_diagnostics(
    predictor: DownstreamPredictor,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> list[str]:
    lines = ["PREDICTION DIAGNOSTICS", "----------------------"]
    lines.append(f"y_true[:20]={y_eval[:20].tolist()}")
    lines.append(f"true_distribution={np.unique(y_eval, return_counts=True)}")
    for model_name in ("xgboost", "lstm"):
        if model_name == "xgboost":
            probabilities = predictor.xgboost_model.predict_proba(X_eval)
        else:
            probabilities, _ = predictor.lstm_model.predict_proba_with_uncertainty(
                X_eval, n_mc_samples=10
            )
        predictions = (probabilities >= 0.5).astype(int)
        lines.extend([
            f"{model_name}.y_pred[:20]={predictions[:20].tolist()}",
            f"{model_name}.probabilities[:5]={probabilities[:5].round(6).tolist()}",
            f"{model_name}.probability_shape={probabilities.shape}",
            f"{model_name}.predicted_distribution={np.unique(predictions, return_counts=True)}",
        ])
    return lines


def main() -> None:
    print("[Phase 8] Downstream prediction corruption experiment")
    X_train, y_train, X_val, y_val, X_test, y_test = load_arrays()
    label_lines = [
        describe_labels("y_train", y_train),
        describe_labels("y_val", y_val),
        describe_labels("y_test", y_test),
    ]
    require_binary_labels("y_train", y_train)
    X_eval, y_eval, evaluation_source = select_evaluation_split(
        X_val, y_val, X_test, y_test
    )
    require_binary_labels("y_eval", y_eval)
    predictor = DownstreamPredictor(device="cpu")
    predictor.fit(X_train, y_train, lstm_epochs=15)

    scenarios = build_scenarios(X_eval)
    comparison = evaluate_scenarios(predictor, y_eval, scenarios)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(COMPARISON_CSV, index=False)

    report = [
        "SensorShield Phase 8 -- Downstream Prediction Report",
        f"Training windows: {len(X_train):,}",
        f"Evaluation source: {evaluation_source}",
        f"Evaluation windows: {len(X_eval):,}",
        "",
        "LABEL DISTRIBUTIONS",
        "-------------------",
        *label_lines,
        "",
        *summarize_input_changes(scenarios["Healthy"], scenarios),
        "",
        *summarize_prediction_changes(comparison),
        "",
        *prediction_diagnostics(predictor, X_eval, y_eval),
        "",
        "The positive class is 1 (abnormal/failure); confidence is the "
        "distance from the 0.5 decision boundary; uncertainty is MC-Dropout "
        "standard deviation for LSTM and predictive entropy for XGBoost.",
        "",
        comparison.to_string(index=False),
    ]
    REPORT_TXT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Comparison saved -> {COMPARISON_CSV}")
    print(f"Report saved -> {REPORT_TXT}")


if __name__ == "__main__":
    main()
