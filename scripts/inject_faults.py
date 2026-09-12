"""
scripts/inject_faults.py
========================
Offline dataset corruption script — Phase 3.

Reads the clean simulated sensor CSV, applies a representative set of
all 8 fault types at varying severities, and writes three outputs to
data/processed/:
  - corrupted_dataset.csv        : the corrupted time-series
  - fault_labels.csv             : boolean label per sensor per row
  - fault_injection_report.txt   : human-readable summary

Usage (from the repo root):
    python -m scripts.inject_faults          # default settings
    python scripts/inject_faults.py          # also works

The corrupted_dataset.csv is used in Phase 4 to evaluate anomaly detectors
on clean vs. corrupted data.
"""

import os
import sys
import textwrap
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Make sure backend/app is importable
_HERE    = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_HERE)
_BACKEND = os.path.join(_ROOT, "backend")
sys.path.insert(0, _BACKEND)

from app.ml.fault_injection.engine import FaultConfig, FaultEngine
from app.ml.fault_injection.faults import FaultType

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR   = os.path.join(_ROOT, "data")
RAW_CSV    = os.path.join(DATA_DIR, "raw", "sensor_data.csv")
PROC_DIR   = os.path.join(DATA_DIR, "processed")
OUT_CSV    = os.path.join(PROC_DIR, "corrupted_dataset.csv")
LABEL_CSV  = os.path.join(PROC_DIR, "fault_labels.csv")
REPORT_TXT = os.path.join(PROC_DIR, "fault_injection_report.txt")

os.makedirs(PROC_DIR, exist_ok=True)


def load_data() -> pd.DataFrame:
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(
            f"Raw data not found at {RAW_CSV}. "
            "Run the data generation script first (Phase 2)."
        )
    df = pd.read_csv(RAW_CSV, parse_dates=["timestamp"])
    print(f"  Loaded {len(df):,} rows from {RAW_CSV}")
    return df


def build_fault_configs(df: pd.DataFrame) -> list[FaultConfig]:
    """
    Build a representative set of fault configs covering all 8 fault types.
    Faults are spread across the timeline and target different sensors.
    """
    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()
    total_seconds = (ts_max - ts_min).total_seconds()

    # Helper: start at a given fraction of the timeline
    def t(frac: float) -> datetime:
        return ts_min + timedelta(seconds=total_seconds * frac)

    # Detect sensor columns — everything that isn't timestamp / machine_status
    meta_cols = {"timestamp", "machine_status", "machine_id"}
    sensor_cols = [c for c in df.columns if c not in meta_cols]

    if not sensor_cols:
        raise ValueError("No sensor columns found in the dataset.")

    # Assign sensors round-robin across fault types
    def s(idx: int) -> str:
        return sensor_cols[idx % len(sensor_cols)]

    faults = [
        FaultConfig(
            sensor_id="TEMP_01" if "TEMP_01" in sensor_cols else s(0),
            fault_type=FaultType.DRIFT,
            start_time=t(0.05),
            duration_seconds=int(total_seconds * 0.08),
            severity=0.6,
            label="drift_temp_moderate",
        ),
        FaultConfig(
            sensor_id="PRESSURE_01" if "PRESSURE_01" in sensor_cols else s(1),
            fault_type=FaultType.BIAS,
            start_time=t(0.15),
            duration_seconds=int(total_seconds * 0.06),
            severity=0.8,
            label="bias_pressure_high",
        ),
        FaultConfig(
            sensor_id="VIBRATION_01" if "VIBRATION_01" in sensor_cols else s(2),
            fault_type=FaultType.NOISE,
            start_time=t(0.25),
            duration_seconds=int(total_seconds * 0.10),
            severity=0.7,
            label="noise_vibration_high",
        ),
        FaultConfig(
            sensor_id="CURRENT_01" if "CURRENT_01" in sensor_cols else s(3),
            fault_type=FaultType.STUCK,
            start_time=t(0.38),
            duration_seconds=int(total_seconds * 0.05),
            severity=1.0,
            label="stuck_current_full",
        ),
        FaultConfig(
            sensor_id="FLOW_01" if "FLOW_01" in sensor_cols else s(4),
            fault_type=FaultType.MISSING,
            start_time=t(0.50),
            duration_seconds=int(total_seconds * 0.04),
            severity=1.0,
            label="missing_flow",
        ),
        FaultConfig(
            sensor_id="HUMIDITY_01" if "HUMIDITY_01" in sensor_cols else s(5),
            fault_type=FaultType.INTERMITTENT,
            start_time=t(0.60),
            duration_seconds=int(total_seconds * 0.08),
            severity=0.5,
            label="intermittent_humidity_mid",
        ),
        FaultConfig(
            sensor_id="TEMP_01" if "TEMP_01" in sensor_cols else s(0),
            fault_type=FaultType.SPIKE,
            start_time=t(0.72),
            duration_seconds=int(total_seconds * 0.06),
            severity=0.9,
            label="spikes_temp_severe",
        ),
        FaultConfig(
            sensor_id="PRESSURE_01" if "PRESSURE_01" in sensor_cols else s(1),
            fault_type=FaultType.SAMPLING_FAILURE,
            start_time=t(0.85),
            duration_seconds=int(total_seconds * 0.07),
            severity=0.6,
            label="sampling_failure_pressure_mid",
        ),
    ]
    return faults


def print_report(faults: list[FaultConfig], corrupted: pd.DataFrame,
                 labels: pd.DataFrame, save_path: str) -> None:
    lines = [
        "=" * 70,
        "SensorShield Phase 3 -- Fault Injection Report",
        f"Generated : {datetime.now().isoformat()}",
        f"Rows      : {len(corrupted):,}",
        f"Sensors   : {list(labels.columns)}",
        "=" * 70,
        "",
        f"{'Fault Type':<22} {'Sensor':<15} {'Severity':>8}  {'Duration(s)':>12}  {'Label rows':>10}",
        "-" * 75,
    ]
    for f in faults:
        label_col = f.sensor_id
        n_labeled = int(labels[label_col].sum()) if label_col in labels.columns else 0
        lines.append(
            f"{f.fault_type.value:<22} {f.sensor_id:<15} {f.severity:>8.2f}  "
            f"{f.duration_seconds:>12,}  {n_labeled:>10,}"
        )
    lines += [
        "",
        "Total fault-labelled rows (any sensor):",
        f"  {int(labels.any(axis=1).sum()):,} / {len(labels):,} "
        f"({100 * labels.any(axis=1).mean():.1f} %)",
    ]
    report = "\n".join(lines)
    print(report.encode("ascii", "replace").decode("ascii"))
    with open(save_path, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(f"\n  Report saved -> {save_path}")


def main():
    print("\n[Phase 3] Fault Injection Engine -- offline dataset corruption")
    print("-" * 60)

    print("\n1. Loading raw data ...")
    df = load_data()

    print("\n2. Building fault configurations ...")
    faults = build_fault_configs(df)
    print(f"  {len(faults)} faults configured.")

    print("\n3. Applying faults ...")
    engine   = FaultEngine(timestamp_col="timestamp")
    corrupted, labels = engine.apply(df, faults)

    print("\n4. Saving outputs ...")
    corrupted.to_csv(OUT_CSV,   index=False)
    labels.to_csv(LABEL_CSV,    index=False)
    print(f"  corrupted_dataset.csv  -> {OUT_CSV}")
    print(f"  fault_labels.csv       -> {LABEL_CSV}")

    print("\n5. Report:")
    print_report(faults, corrupted, labels, REPORT_TXT)

    print("\n[Phase 3] Done. [OK]")


if __name__ == "__main__":
    main()
