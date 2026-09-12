"""Diagnose ML model loading issues."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import logging
logging.basicConfig(level=logging.WARNING)

from app.services.sensor_pipeline import model_registry, PIPELINE_ARTIFACTS_PATH, ANOMALY_MODEL_PATH, PREDICTION_MODEL_DIR

print("=== Path Check ===")
print(f"Pipeline artifacts : {PIPELINE_ARTIFACTS_PATH} -> exists={PIPELINE_ARTIFACTS_PATH.exists()}")
print(f"Autoencoder model  : {ANOMALY_MODEL_PATH} -> exists={ANOMALY_MODEL_PATH.exists()}")
print(f"Prediction dir     : {PREDICTION_MODEL_DIR} -> exists={PREDICTION_MODEL_DIR.exists()}")
print()

print("=== Loading Models ===")
try:
    from app.ml.preprocessing.pipeline import SensorPreprocessingPipeline
    pipeline = SensorPreprocessingPipeline.load_artifacts(str(PIPELINE_ARTIFACTS_PATH))
    print(f"[OK] Pipeline loaded. Sensors: {pipeline.artifacts.sensor_cols}")
except Exception as e:
    print(f"[FAIL] Pipeline: {e}")

try:
    from app.ml.anomaly_detection.lstm_autoencoder import LSTMAutoencoderDetector
    ae = LSTMAutoencoderDetector.load(str(ANOMALY_MODEL_PATH), device="cpu")
    print(f"[OK] Autoencoder loaded: {ae}")
except Exception as e:
    print(f"[FAIL] Autoencoder: {e}")

try:
    from app.ml.reliability.engine import SensorReliabilityEngine
    rel = SensorReliabilityEngine()
    print(f"[OK] Reliability engine created")
except Exception as e:
    print(f"[FAIL] Reliability engine: {e}")

try:
    from app.ml.prediction.engine import DownstreamPredictor
    pred = DownstreamPredictor.load(str(PREDICTION_MODEL_DIR), device="cpu")
    print(f"[OK] Predictor loaded")
except Exception as e:
    print(f"[FAIL] Predictor: {e}")
