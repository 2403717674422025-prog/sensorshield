"""
scripts/generate_dataset.py
============================
Generates a synthetic pump sensor dataset that mirrors the structure and
statistical properties of the Kaggle Pump Sensor Data (CC0).

Why synthetic generation instead of downloading?
- No Kaggle credentials required to run the project from scratch.
- Fully reproducible (fixed random seed).
- When the real dataset IS available, just drop sensor_data.csv into
  data/raw/ and re-run the preprocessing pipeline — nothing else changes.

Dataset characteristics reproduced:
- 52 sensor columns (sensor_00 to sensor_51)
- machine_status column: NORMAL / BROKEN / RECOVERING
- Timestamps at 1-minute intervals over ~50 days
- Realistic inter-sensor correlations (pump physics)
- Real-world missing value pattern (~2-5% per sensor, not uniform)
- Natural degradation patterns before BROKEN events
- Sensor 15 and Sensor 48 are near-constant (reproduce dataset quirk)

Run from project root:
    python scripts/generate_dataset.py
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT_FILE = os.path.join(OUT_DIR, "sensor_data.csv")

# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------
N_MINUTES = 52_000          # ~36 days of 1-minute readings
START_TS  = datetime(2018, 4, 1, 0, 0, 0)
timestamps = [START_TS + timedelta(minutes=i) for i in range(N_MINUTES)]

N_SENSORS = 52              # sensor_00 … sensor_51

print(f"Generating {N_MINUTES:,} rows × {N_SENSORS} sensors …")

# ---------------------------------------------------------------------------
# Step 1: Build machine status column
# Pump alternates between NORMAL, BROKEN, RECOVERING in realistic blocks
# ---------------------------------------------------------------------------
status = ["NORMAL"] * N_MINUTES

# Inject 5 failure events at random positions (not too early, not too late)
failure_starts = rng.integers(5000, N_MINUTES - 3000, size=5)
for fs in sorted(failure_starts):
    # Broken window: 60-180 minutes
    broken_len = int(rng.integers(60, 180))
    # Recovering window: 120-360 minutes
    recover_len = int(rng.integers(120, 360))
    for i in range(fs, min(fs + broken_len, N_MINUTES)):
        status[i] = "BROKEN"
    for i in range(fs + broken_len, min(fs + broken_len + recover_len, N_MINUTES)):
        status[i] = "RECOVERING"

# ---------------------------------------------------------------------------
# Step 2: Generate sensor signals
# Group sensors by physical role to create realistic correlations
# ---------------------------------------------------------------------------
t = np.arange(N_MINUTES, dtype=float)
status_arr = np.array(status)

# Helper: smooth degradation signal before/during BROKEN
def degradation_signal(status_arr, lookahead=200):
    """Returns a 0-1 array. 1=healthy, 0=fully broken. Smoothly ramps down."""
    deg = np.ones(N_MINUTES)
    for i in range(N_MINUTES):
        if status_arr[i] == "BROKEN":
            deg[i] = 0.0
        elif status_arr[i] == "RECOVERING":
            deg[i] = 0.4
    # Smooth backward: ramp down in the 200 mins before each broken event
    for i in range(N_MINUTES - 1, -1, -1):
        if deg[i] == 0.0:
            for j in range(max(0, i - lookahead), i):
                ramp = (j - (i - lookahead)) / lookahead
                deg[j] = min(deg[j], ramp)
    return deg

deg = degradation_signal(status_arr)

# Base periodic component (pump cycle noise)
cycle = np.sin(2 * np.pi * t / 1440)  # daily cycle

# ---- Sensor groups ----
# Group A (sensors 00-07): pressure-related
#   Normal ~3-5 bar, drops during fault
def group_A(idx):
    base = 4.0 + 0.5 * cycle + rng.normal(0, 0.08, N_MINUTES)
    fault_drop = (1 - deg) * rng.uniform(1.5, 2.5)
    sig = base - fault_drop
    sig += rng.normal(0, 0.01 * (idx + 1), N_MINUTES)  # per-sensor variation
    return sig.clip(0, 10)

# Group B (sensors 08-15): flow-related
#   Normal ~100-150 L/min, drops during fault
def group_B(idx):
    base = 125 + 15 * cycle + rng.normal(0, 2.0, N_MINUTES)
    fault_drop = (1 - deg) * rng.uniform(40, 80)
    sig = base - fault_drop
    sig += rng.normal(0, 0.5 * (idx - 7), N_MINUTES)
    if idx == 15:
        # Sensor 15 is near-constant (dataset quirk)
        sig = np.full(N_MINUTES, 0.0) + rng.normal(0, 0.001, N_MINUTES)
    return sig.clip(0, 300)

# Group C (sensors 16-23): vibration-related
#   Normal ~0.5-2 mm/s, spikes during fault
def group_C(idx):
    base = 1.2 + 0.3 * cycle + rng.normal(0, 0.05, N_MINUTES)
    fault_spike = (1 - deg) * rng.uniform(1.0, 3.0)
    sig = base + fault_spike
    sig += rng.normal(0, 0.02 * (idx - 15), N_MINUTES)
    return sig.clip(0, 20)

# Group D (sensors 24-31): temperature-related
#   Normal ~60-80 °C, rises during fault
def group_D(idx):
    base = 70 + 5 * cycle + rng.normal(0, 0.8, N_MINUTES)
    fault_rise = (1 - deg) * rng.uniform(5, 15)
    sig = base + fault_rise
    sig += rng.normal(0, 0.3 * (idx - 23), N_MINUTES)
    return sig.clip(20, 150)

# Group E (sensors 32-39): current-related
#   Normal ~8-12 A, changes during fault
def group_E(idx):
    base = 10 + 1.5 * cycle + rng.normal(0, 0.2, N_MINUTES)
    fault_delta = (1 - deg) * rng.uniform(-3, 4)
    sig = base + fault_delta
    sig += rng.normal(0, 0.1 * (idx - 31), N_MINUTES)
    return sig.clip(0, 30)

# Group F (sensors 40-47): efficiency / derived metrics
#   Normal ~0.6-0.9 (ratio), drops during fault
def group_F(idx):
    base = 0.75 + 0.05 * cycle + rng.normal(0, 0.01, N_MINUTES)
    fault_drop = (1 - deg) * rng.uniform(0.1, 0.3)
    sig = base - fault_drop
    sig += rng.normal(0, 0.005 * (idx - 39), N_MINUTES)
    return sig.clip(0, 1)

# Group G (sensors 48-51): near-constant or secondary
def group_G(idx):
    if idx == 48:
        # Near-constant (dataset quirk — mirrors sensor_15)
        return np.full(N_MINUTES, 0.0) + rng.normal(0, 0.001, N_MINUTES)
    base = 50 + 5 * cycle + rng.normal(0, 1.0, N_MINUTES)
    fault_delta = (1 - deg) * rng.uniform(-10, 10)
    sig = base + fault_delta
    return sig.clip(0, 200)

# Build sensor matrix
sensor_data = {}
for i in range(N_SENSORS):
    name = f"sensor_{i:02d}"
    if   i <= 7:  sensor_data[name] = group_A(i)
    elif i <= 15: sensor_data[name] = group_B(i)
    elif i <= 23: sensor_data[name] = group_C(i)
    elif i <= 31: sensor_data[name] = group_D(i)
    elif i <= 39: sensor_data[name] = group_E(i)
    elif i <= 47: sensor_data[name] = group_F(i)
    else:         sensor_data[name] = group_G(i)

# ---------------------------------------------------------------------------
# Step 3: Inject realistic missing values (not uniform — bursty)
# ---------------------------------------------------------------------------
for name, sig in sensor_data.items():
    # ~2-5% missing rate per sensor, in bursts of 1-20 readings
    target_missing = rng.uniform(0.02, 0.05)
    n_missing = int(N_MINUTES * target_missing)
    n_bursts = rng.integers(20, 80)
    burst_starts = rng.integers(0, N_MINUTES, size=n_bursts)
    for bs in burst_starts:
        burst_len = int(rng.integers(1, max(2, n_missing // n_bursts * 2)))
        sig[bs:bs + burst_len] = np.nan
    sensor_data[name] = sig

# ---------------------------------------------------------------------------
# Step 4: Build DataFrame
# ---------------------------------------------------------------------------
df = pd.DataFrame(sensor_data)
df.insert(0, "timestamp", timestamps)
df["machine_status"] = status

# ---------------------------------------------------------------------------
# Step 5: Save
# ---------------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
df.to_csv(OUT_FILE, index=False)

n_normal   = (df["machine_status"] == "NORMAL").sum()
n_broken   = (df["machine_status"] == "BROKEN").sum()
n_recover  = (df["machine_status"] == "RECOVERING").sum()
n_missing  = df[list(sensor_data.keys())].isna().sum().sum()
missing_pct = 100 * n_missing / (N_MINUTES * N_SENSORS)

print(f"\nDataset saved → {OUT_FILE}")
print(f"  Rows         : {len(df):,}")
print(f"  Sensors      : {N_SENSORS}")
print(f"  NORMAL       : {n_normal:,} ({100*n_normal/N_MINUTES:.1f}%)")
print(f"  BROKEN       : {n_broken:,} ({100*n_broken/N_MINUTES:.1f}%)")
print(f"  RECOVERING   : {n_recover:,} ({100*n_recover/N_MINUTES:.1f}%)")
print(f"  Missing vals : {n_missing:,} ({missing_pct:.2f}% of all readings)")
print(f"\nNext step: python scripts/run_eda.py")
