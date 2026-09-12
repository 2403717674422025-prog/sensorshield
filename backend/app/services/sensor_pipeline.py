"""
services/sensor_pipeline.py
============================
Real-time sensor processing pipeline (Phase 4).

Connects every stage of the ML stack for a single incoming reading:

  SensorReading
    → fetch recent history from DB
    → SensorPreprocessingPipeline.transform()  (normalise the window)
    → LSTMAutoencoderDetector.score()          (reconstruction error)
    → SensorReliabilityEngine.evaluate_sensor() (health score + status)
    → SensorHealth DB record
    → DownstreamPredictor.predict_window()     (failure probability)
    → Prediction DB record
    → AlertService.evaluate_and_create_alert() (threshold check)
    → WebSocket broadcast

Design rules
------------
- Every ML stage is wrapped in try/except so a single model failure
  does not crash the API and the raw reading is always persisted.
- No ML state is held on the instance — models are loaded once at startup
  and stored on the FastAPI app state.
- The pipeline is async-safe; all DB I/O uses the injected AsyncSession.
"""

from __future__ import annotations

import logging
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.reading import SensorReading
from app.models.health import SensorHealth
from app.models.prediction import Prediction, TrustStatus
from app.models.sensor import Sensor, SensorStatus
from app.ml.preprocessing.pipeline import SensorPreprocessingPipeline, PipelineArtifacts
from app.ml.anomaly_detection.lstm_autoencoder import LSTMAutoencoderDetector
from app.ml.reliability.engine import SensorReliabilityEngine, SensorBaseline
from app.ml.prediction.engine import DownstreamPredictor
from app.services.alert_service import evaluate_and_create_alert
from app.services.websocket_manager import ws_manager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (resolved relative to repo root)
# ---------------------------------------------------------------------------
_HERE        = Path(__file__).resolve()
_REPO_ROOT   = _HERE.parents[3]          # …/SensorShield
_MODEL_DIR   = _REPO_ROOT / "models"
_DATA_PROC   = _REPO_ROOT / "data" / "processed"

ANOMALY_MODEL_PATH       = _MODEL_DIR / "anomaly"  / "lstm_autoencoder.pt"
PREDICTION_MODEL_DIR     = _MODEL_DIR / "prediction"
PIPELINE_ARTIFACTS_PATH  = _DATA_PROC / "pipeline_artifacts.pkl"
SENSOR_MAPPING_PATH      = _DATA_PROC / "sensor_mapping.json"

# Number of recent readings to use as a sensor window for evaluation
WINDOW_SIZE = 30


# ---------------------------------------------------------------------------
# Model registry — loaded once, reused for every request
# ---------------------------------------------------------------------------
class ModelRegistry:
    """
    Holds all loaded ML models. Populated by load_all() at app startup.
    Each field is None until loaded successfully — individual failures are
    logged but do not block the rest of the pipeline.
    """

    def __init__(self) -> None:
        self.pipeline:    Optional[SensorPreprocessingPipeline] = None
        self.artifacts:   Optional[PipelineArtifacts]            = None
        self.autoencoder: Optional[LSTMAutoencoderDetector]      = None
        self.reliability: Optional[SensorReliabilityEngine]      = None
        self.predictor:   Optional[DownstreamPredictor]          = None

    def load_all(self) -> None:
        """Load every ML model. Called once from main.py lifespan startup."""

        # 1. Preprocessing artifacts
        try:
            self.pipeline  = SensorPreprocessingPipeline.load_artifacts(
                str(PIPELINE_ARTIFACTS_PATH)
            )
            self.artifacts = self.pipeline.artifacts
            logger.info("Preprocessing pipeline artifacts loaded.")
        except Exception as exc:
            logger.warning("Could not load pipeline artifacts: %s", exc)

        # 2. LSTM Autoencoder
        try:
            self.autoencoder = LSTMAutoencoderDetector.load(
                str(ANOMALY_MODEL_PATH), device="cpu"
            )
            logger.info("LSTM Autoencoder detector loaded.")
        except Exception as exc:
            logger.warning("Could not load autoencoder: %s", exc)

        # 3. Reliability Engine — fit baselines from training data if available
        self.reliability = SensorReliabilityEngine()
        try:
            X_train = np.load(str(_DATA_PROC / "X_train.npy"))
            if self.artifacts:
                self.reliability.fit_baselines(
                    df_normal   = X_train,
                    sensor_ids  = self.artifacts.sensor_cols,
                    valid_ranges= {
                        "TEMP_01":     (20.0, 150.0),
                        "PRESSURE_01": (0.0,  10.0),
                        "VIBRATION_01":(0.0,  20.0),
                        "CURRENT_01":  (0.0,  30.0),
                        "FLOW_01":     (0.0,  300.0),
                        "HUMIDITY_01": (0.0,  1.0),
                    },
                )
                logger.info("Reliability engine baselines fitted from training data.")
        except Exception as exc:
            logger.warning("Could not fit reliability baselines: %s", exc)

        # 4. Downstream predictor
        try:
            pred_dir = str(PREDICTION_MODEL_DIR)
            if (PREDICTION_MODEL_DIR / "xgboost_model.pkl").exists() and \
               (PREDICTION_MODEL_DIR / "lstm_model.pt").exists():
                self.predictor = DownstreamPredictor.load(pred_dir, device="cpu")
                logger.info("DownstreamPredictor loaded.")
            else:
                logger.warning(
                    "Prediction models not found at %s — run scripts/train_prediction_models.py",
                    pred_dir,
                )
        except Exception as exc:
            logger.warning("Could not load DownstreamPredictor: %s", exc)


# Module-level singleton — imported by main.py and the sensor route
model_registry = ModelRegistry()


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------
async def process_reading(
    db:      AsyncSession,
    reading: SensorReading,
) -> Dict:
    """
    Run the full processing pipeline for one new sensor reading.

    Returns a summary dict suitable for the API response and WebSocket broadcast.
    Never raises — all failures are logged and a degraded result is returned.
    """
    sensor_id  = reading.sensor_id
    machine_id = reading.machine_id
    result: Dict = {
        "sensor_id":           sensor_id,
        "reading_id":          reading.id,
        "value":               reading.value,
        "timestamp":           reading.timestamp.isoformat() if reading.timestamp else None,
        "health_score":        None,
        "status":              "UNKNOWN",
        "anomaly_score":       None,
        "failure_probability": None,
        "trust_status":        "UNKNOWN",
        "alert_created":       False,
        "pipeline_errors":     [],
    }

    # ------------------------------------------------------------------ #
    # Step 1 — Fetch recent history window from DB
    # ------------------------------------------------------------------ #
    window_raw: Optional[np.ndarray] = None
    try:
        window_raw = await _fetch_window(db, sensor_id, machine_id)
    except Exception as exc:
        logger.warning("Could not fetch window for %s: %s", sensor_id, exc)
        result["pipeline_errors"].append(f"window_fetch: {exc}")

    # ------------------------------------------------------------------ #
    # Step 2 — Normalise window via preprocessing pipeline
    # ------------------------------------------------------------------ #
    window_norm: Optional[np.ndarray] = None
    if window_raw is not None and model_registry.pipeline is not None:
        try:
            window_norm = _preprocess_window(window_raw, sensor_id)
        except Exception as exc:
            logger.warning("Preprocessing failed for %s: %s", sensor_id, exc)
            result["pipeline_errors"].append(f"preprocess: {exc}")

    # ------------------------------------------------------------------ #
    # Step 3 — Anomaly score (LSTM Autoencoder reconstruction error)
    # ------------------------------------------------------------------ #
    recon_error = 0.0
    if window_norm is not None and model_registry.autoencoder is not None:
        try:
            recon_error = _compute_anomaly_score(window_norm)
        except Exception as exc:
            logger.warning("Anomaly scoring failed for %s: %s", sensor_id, exc)
            result["pipeline_errors"].append(f"anomaly: {exc}")

    result["anomaly_score"] = round(recon_error, 6)

    # ------------------------------------------------------------------ #
    # Step 4 — Reliability / Health evaluation
    # ------------------------------------------------------------------ #
    health_eval = None
    if window_raw is not None and model_registry.reliability is not None:
        try:
            # Use the raw (un-normalised) 1-D signal for the selected sensor
            sensor_signal = _extract_sensor_signal(window_raw, sensor_id)
            health_eval = model_registry.reliability.evaluate_sensor(
                sensor_id          = sensor_id,
                signal_window      = sensor_signal,
                timestamp          = reading.timestamp or datetime.now(timezone.utc),
                reconstruction_error = recon_error,
            )
            result["health_score"] = round(health_eval.health_score, 2)
            result["status"]       = health_eval.status.value
        except Exception as exc:
            logger.warning("Reliability evaluation failed for %s: %s", sensor_id, exc)
            result["pipeline_errors"].append(f"reliability: {exc}")

    # ------------------------------------------------------------------ #
    # Step 5 — Persist SensorHealth record
    # ------------------------------------------------------------------ #
    if health_eval is not None:
        try:
            health_record = SensorHealth(
                sensor_id            = sensor_id,
                timestamp            = health_eval.timestamp,
                health_score         = health_eval.health_score,
                anomaly_score        = health_eval.anomaly_score,
                drift_score          = health_eval.drift_score,
                noise_score          = health_eval.noise_score,
                missing_data_score   = health_eval.missing_data_score,
                consistency_score    = health_eval.consistency_score,
                reconstruction_error = health_eval.reconstruction_error,
                status               = health_eval.status,
                status_confidence    = health_eval.status_confidence,
                reason               = health_eval.reason,
            )
            db.add(health_record)
            await db.flush()

            # Also update the Sensor row's current status
            sensor_row = await db.get(Sensor, sensor_id)
            if sensor_row:
                sensor_row.status = health_eval.status
        except Exception as exc:
            logger.warning("SensorHealth persistence failed for %s: %s", sensor_id, exc)
            result["pipeline_errors"].append(f"health_persist: {exc}")

    # ------------------------------------------------------------------ #
    # Step 6 — Downstream failure prediction
    # ------------------------------------------------------------------ #
    failure_prob  = None
    uncertainty   = None
    trust_status  = TrustStatus.UNKNOWN
    trust_score   = None

    if window_norm is not None and model_registry.predictor is not None:
        try:
            pred_out = model_registry.predictor.predict_window(
                window_norm, use_model="lstm"
            )
            failure_prob = pred_out.failure_probability
            uncertainty  = pred_out.model_uncertainty

            # Compute trust score from sensor health
            avg_health = result["health_score"] or 50.0
            trust_score = float(avg_health)

            if trust_score >= settings.TRUST_HIGH_THRESHOLD:
                trust_status = TrustStatus.TRUSTED
            elif trust_score >= settings.TRUST_LOW_THRESHOLD:
                trust_status = TrustStatus.CAUTION
            else:
                trust_status = TrustStatus.UNTRUSTED

            result["failure_probability"] = round(failure_prob, 4)
            result["trust_status"]        = trust_status.value
        except Exception as exc:
            logger.warning("Prediction failed for %s: %s", sensor_id, exc)
            result["pipeline_errors"].append(f"prediction: {exc}")

    # ------------------------------------------------------------------ #
    # Step 7 — Persist Prediction record
    # ------------------------------------------------------------------ #
    if failure_prob is not None:
        try:
            pred_record = Prediction(
                machine_id             = machine_id,
                timestamp              = reading.timestamp or datetime.now(timezone.utc),
                failure_probability    = failure_prob,
                model_uncertainty      = uncertainty,
                avg_sensor_reliability = result["health_score"],
                min_sensor_reliability = result["health_score"],
                trust_score            = trust_score,
                trust_status           = trust_status,
                trust_reason           = (
                    f"Sensor health: {result['health_score']}. "
                    f"Status: {result['status']}."
                ),
            )
            db.add(pred_record)
            await db.flush()
        except Exception as exc:
            logger.warning("Prediction persistence failed for %s: %s", sensor_id, exc)
            result["pipeline_errors"].append(f"pred_persist: {exc}")

    # ------------------------------------------------------------------ #
    # Step 8 — Alert evaluation
    # ------------------------------------------------------------------ #
    if health_eval is not None:
        try:
            alert = await evaluate_and_create_alert(
                db         = db,
                evaluation = health_eval,
                machine_id = machine_id,
                trust_score= trust_score,
            )
            if alert:
                result["alert_created"] = True
                await ws_manager.broadcast_alert({
                    "id":          alert.id,
                    "sensor_id":   alert.sensor_id,
                    "machine_id":  alert.machine_id,
                    "severity":    alert.severity.value,
                    "fault_type":  alert.fault_type.value,
                    "description": alert.description,
                    "timestamp":   alert.timestamp.isoformat(),
                })
        except Exception as exc:
            logger.warning("Alert evaluation failed for %s: %s", sensor_id, exc)
            result["pipeline_errors"].append(f"alert: {exc}")

    # ------------------------------------------------------------------ #
    # Step 9 — WebSocket broadcast of health update
    # ------------------------------------------------------------------ #
    try:
        await ws_manager.broadcast_health_update({
            "sensor_id":           sensor_id,
            "machine_id":          machine_id,
            "health_score":        result["health_score"],
            "status":              result["status"],
            "anomaly_score":       result["anomaly_score"],
            "failure_probability": result["failure_probability"],
            "trust_status":        result["trust_status"],
            "timestamp":           result["timestamp"],
        })
    except Exception as exc:
        logger.debug("WebSocket broadcast failed: %s", exc)

    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
async def _fetch_window(
    db: AsyncSession,
    sensor_id: str,
    machine_id: str,
) -> np.ndarray:
    """
    Fetch the last WINDOW_SIZE readings for the sensor and return a 2-D
    array of shape (window_size, n_sensors).

    All sensors for the same machine are fetched together so the reliability
    engine can access peer channels.  Missing sensors are filled with NaN.
    """
    arts = model_registry.artifacts

    # Determine sensor columns — use artifacts if available, else fetch from DB
    if arts is not None and any(s.startswith(machine_id.replace('M0', 'M').split('_')[0]) or True
                                 for s in arts.sensor_cols):
        sensor_cols = arts.sensor_cols
    else:
        sensor_cols = None

    # If no artifacts or sensor not in artifacts, build a single-sensor window
    if sensor_cols is None or sensor_id not in sensor_cols:
        # Fetch just this sensor's readings
        q = (
            select(SensorReading)
            .where(SensorReading.sensor_id == sensor_id)
            .order_by(SensorReading.timestamp.desc())
            .limit(WINDOW_SIZE)
        )
        result = await db.execute(q)
        rows   = result.scalars().all()

        if not rows:
            raise ValueError(f"No readings found for sensor {sensor_id}")

        vals = [r.value for r in reversed(rows)]
        if len(vals) < WINDOW_SIZE:
            vals = [float('nan')] * (WINDOW_SIZE - len(vals)) + vals

        window = np.array(vals, dtype=np.float32).reshape(WINDOW_SIZE, 1)
        return window

    # Full multivariate window for known sensors
    q = (
        select(SensorReading)
        .where(SensorReading.machine_id == machine_id)
        .order_by(SensorReading.timestamp.desc())
        .limit(WINDOW_SIZE * len(sensor_cols))
    )
    result = await db.execute(q)
    rows   = result.scalars().all()

    if not rows:
        raise ValueError(f"No readings found for machine {machine_id}")

    from collections import defaultdict
    ts_map: Dict[datetime, Dict[str, float]] = defaultdict(dict)
    for r in rows:
        ts_map[r.timestamp][r.sensor_id] = r.value

    sorted_ts = sorted(ts_map.keys(), reverse=True)[:WINDOW_SIZE]
    sorted_ts.sort()

    window = np.full((len(sorted_ts), len(sensor_cols)), np.nan, dtype=np.float32)
    for i, ts in enumerate(sorted_ts):
        for j, col in enumerate(sensor_cols):
            if col in ts_map[ts]:
                window[i, j] = ts_map[ts][col]

    if len(sorted_ts) < WINDOW_SIZE:
        pad = np.full((WINDOW_SIZE - len(sorted_ts), len(sensor_cols)), np.nan, dtype=np.float32)
        window = np.vstack([pad, window])

    return window  # (WINDOW_SIZE, n_sensors)


def _preprocess_window(window: np.ndarray, sensor_id: str) -> np.ndarray:
    """
    Normalise the raw window using the fitted pipeline artifacts.
    Returns shape (WINDOW_SIZE, n_sensors) as float32.
    For sensors not in artifacts, applies simple min-max normalization.
    """
    arts = model_registry.artifacts

    # Single-sensor window — simple normalization
    if window.shape[1] == 1 or arts is None:
        arr = window.copy().astype(np.float64)
        col = arr[:, 0]
        col = np.where(np.isnan(col), np.nanmedian(col) if not np.all(np.isnan(col)) else 0.0, col)
        lo, hi = col.min(), col.max()
        if hi - lo > 1e-9:
            col = (col - lo) / (hi - lo)
        arr[:, 0] = col
        return arr.astype(np.float32)

    # Impute NaN → median, clip, normalise
    arr = window.copy().astype(np.float64)
    for idx, col in enumerate(arts.sensor_cols):
        col_vals = arr[:, idx]
        last_valid = arts.median_fill.get(col, 0.0)
        for i in range(len(col_vals)):
            if np.isnan(col_vals[i]):
                col_vals[i] = last_valid
            else:
                last_valid = col_vals[i]
        arr[:, idx] = col_vals

        lo = arts.norm_min.get(col, 0.0)
        hi = arts.norm_max.get(col, 1.0)
        rng = hi - lo
        if rng > 1e-9:
            arr[:, idx] = (arr[:, idx] - lo) / rng

    return arr.astype(np.float32)


def _compute_anomaly_score(window_norm: np.ndarray) -> float:
    """
    Run LSTM Autoencoder on the normalised window.
    Returns the mean reconstruction error (MSE) as a float.
    """
    det = model_registry.autoencoder
    if det is None:
        return 0.0
    # score() expects (N, W, S) — add batch dimension
    X = window_norm[np.newaxis, :, :]          # (1, W, S)
    scores = det.score(X)                       # (1,)
    return float(scores[0])


def _extract_sensor_signal(window: np.ndarray, sensor_id: str) -> np.ndarray:
    """
    Extract the 1-D signal for a specific sensor from the raw window.
    Falls back to column 0 if sensor_id is not in the mapping.
    """
    # Single-sensor window (from non-M001 machines)
    if window.shape[1] == 1:
        return window[:, 0]

    arts = model_registry.artifacts
    if arts is None or sensor_id not in arts.sensor_cols:
        return window[:, 0]
    idx = arts.sensor_cols.index(sensor_id)
    return window[:, idx]
