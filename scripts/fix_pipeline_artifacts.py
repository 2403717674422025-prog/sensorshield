"""
Regenerate pipeline_artifacts.pkl so it's pickled under the correct
module path (app.ml.preprocessing.pipeline.PipelineArtifacts)
instead of __main__.PipelineArtifacts.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app.ml.preprocessing.pipeline import SensorPreprocessingPipeline, save_splits

print("SensorShield — Regenerating Pipeline Artifacts")
print("=" * 50)

pipeline = SensorPreprocessingPipeline()
dataset  = pipeline.fit_transform()
pipeline.save_artifacts()
save_splits(dataset)

print("\n[OK] pipeline_artifacts.pkl regenerated with correct module path.")
print(f"     Sensors: {dataset.artifacts.sensor_cols}")
