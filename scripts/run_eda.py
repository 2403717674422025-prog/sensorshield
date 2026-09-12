"""
scripts/run_eda.py
==================
Phase 2 — Exploratory Data Analysis

Produces a structured EDA report saved to data/processed/eda_report.txt
and a set of plots saved to data/processed/plots/.

What we analyse:
1. Dataset shape and dtypes
2. Machine status distribution
3. Missing value analysis per sensor
4. Sensor value statistics (mean, std, min, max, skew, kurtosis)
5. Sensor group correlations
6. Temporal analysis: rolling statistics, variance over time
7. Pre-fault behaviour: do sensors change before BROKEN events?
8. Near-constant sensor detection
9. Sensor selection for the 6-sensor machine model

Run from project root:
    python scripts/run_eda.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display required)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = os.path.join(os.path.dirname(__file__), "..")
RAW_FILE  = os.path.join(ROOT, "data", "raw", "sensor_data.csv")
OUT_DIR   = os.path.join(ROOT, "data", "processed")
PLOT_DIR  = os.path.join(OUT_DIR, "plots")
REPORT    = os.path.join(OUT_DIR, "eda_report.txt")

os.makedirs(PLOT_DIR, exist_ok=True)

report_lines = []

def log(msg=""):
    print(msg)
    report_lines.append(msg)

# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
log("=" * 70)
log("SENSORSHIELD — EDA REPORT")
log("=" * 70)

df = pd.read_csv(RAW_FILE, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
N = len(df)
S = len(sensor_cols)

log(f"\n[1] DATASET OVERVIEW")
log(f"    Rows          : {N:,}")
log(f"    Sensors       : {S}")
log(f"    Time start    : {df['timestamp'].min()}")
log(f"    Time end      : {df['timestamp'].max()}")
log(f"    Duration      : {(df['timestamp'].max() - df['timestamp'].min()).days} days")

# Check for duplicate timestamps
n_dupes = df.duplicated("timestamp").sum()
log(f"    Duplicate ts  : {n_dupes}")

# Sampling interval
intervals = df["timestamp"].diff().dropna().dt.total_seconds()
log(f"    Sampling (sec): median={intervals.median():.0f}  "
    f"min={intervals.min():.0f}  max={intervals.max():.0f}")

# ---------------------------------------------------------------------------
# 2. Machine status distribution
# ---------------------------------------------------------------------------
log(f"\n[2] MACHINE STATUS DISTRIBUTION")
status_counts = df["machine_status"].value_counts()
for s, c in status_counts.items():
    log(f"    {s:<15}: {c:>7,}  ({100*c/N:.2f}%)")

# Plot status over time
fig, ax = plt.subplots(figsize=(16, 3))
status_map = {"NORMAL": 0, "RECOVERING": 1, "BROKEN": 2}
status_num = df["machine_status"].map(status_map)
ax.fill_between(df["timestamp"], status_num, alpha=0.6, color="steelblue")
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(["NORMAL", "RECOVERING", "BROKEN"])
ax.set_title("Machine Status Over Time")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "01_machine_status_timeline.png"), dpi=100)
plt.close(fig)

# ---------------------------------------------------------------------------
# 3. Missing value analysis
# ---------------------------------------------------------------------------
log(f"\n[3] MISSING VALUE ANALYSIS")
missing = df[sensor_cols].isna()
missing_pct = 100 * missing.mean()
total_missing = missing.sum().sum()
total_cells = N * S
log(f"    Total missing cells : {total_missing:,} / {total_cells:,}  "
    f"({100*total_missing/total_cells:.2f}%)")

# Sensors with highest missing rates
log(f"\n    Top 10 sensors by missing rate:")
top_missing = missing_pct.sort_values(ascending=False).head(10)
for sensor, pct in top_missing.items():
    log(f"      {sensor}: {pct:.2f}%")

# Plot missing rate per sensor
fig, ax = plt.subplots(figsize=(18, 4))
missing_pct.sort_values(ascending=False).plot(kind="bar", ax=ax, color="coral")
ax.set_title("Missing Value Rate per Sensor (%)")
ax.set_ylabel("Missing %")
ax.set_xlabel("Sensor")
ax.tick_params(axis="x", labelsize=7, rotation=90)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "02_missing_values.png"), dpi=100)
plt.close(fig)

# Burst detection: consecutive missing count
def max_consecutive_missing(series):
    count = max_c = 0
    for v in series.isna():
        if v:
            count += 1
            max_c = max(max_c, count)
        else:
            count = 0
    return max_c

sample_sensors = sensor_cols[:10]
log(f"\n    Max consecutive missing (first 10 sensors):")
for s in sample_sensors:
    mc = max_consecutive_missing(df[s])
    log(f"      {s}: {mc} consecutive")

# ---------------------------------------------------------------------------
# 4. Sensor statistics
# ---------------------------------------------------------------------------
log(f"\n[4] SENSOR STATISTICS")
stats_df = df[sensor_cols].describe().T
stats_df["skew"]     = df[sensor_cols].skew()
stats_df["kurtosis"] = df[sensor_cols].kurtosis()
stats_df["cv"]       = stats_df["std"] / stats_df["mean"].abs()  # coefficient of variation
stats_df["missing%"] = missing_pct

log(f"\n    Sensor value ranges (min/mean/max/cv):")
log(f"    {'Sensor':<15} {'min':>8} {'mean':>10} {'max':>8} {'cv':>6} {'missing%':>9}")
log(f"    {'-'*60}")
for sensor in sensor_cols:
    row = stats_df.loc[sensor]
    log(f"    {sensor:<15} {row['min']:>8.3f} {row['mean']:>10.3f} "
        f"{row['max']:>8.3f} {row['cv']:>6.3f} {row['missing%']:>8.2f}%")

# Near-constant sensor detection (cv < 0.01)
near_const = stats_df[stats_df["cv"] < 0.01].index.tolist()
log(f"\n    Near-constant sensors (CV < 0.01): {near_const}")

# ---------------------------------------------------------------------------
# 5. Correlation analysis
# ---------------------------------------------------------------------------
log(f"\n[5] SENSOR CORRELATION ANALYSIS")

df_filled = df[sensor_cols].fillna(df[sensor_cols].median())
corr = df_filled.corr()

# High correlations
upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
high_corr_pairs = (
    upper_tri.stack()
    .reset_index()
    .rename(columns={"level_0": "s1", "level_1": "s2", 0: "r"})
)
high_corr_pairs = high_corr_pairs[high_corr_pairs["r"].abs() > 0.8].sort_values("r", ascending=False)
log(f"\n    Top 15 highly correlated pairs (|r| > 0.8):")
for _, row in high_corr_pairs.head(15).iterrows():
    log(f"      {row['s1']:15} ↔ {row['s2']:15}  r={row['r']:+.3f}")

# Correlation heatmap (full)
fig, ax = plt.subplots(figsize=(20, 18))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            ax=ax, xticklabels=True, yticklabels=True, linewidths=0.3)
ax.set_title("Sensor Correlation Matrix (lower triangle)")
ax.tick_params(labelsize=7)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "03_correlation_heatmap.png"), dpi=100)
plt.close(fig)

# ---------------------------------------------------------------------------
# 6. Temporal analysis: sensor variance by machine state
# ---------------------------------------------------------------------------
log(f"\n[6] SENSOR BEHAVIOUR BY MACHINE STATE")

df_filled["machine_status"] = df["machine_status"].values
state_stats = df_filled.groupby("machine_status")[sensor_cols].mean()

log(f"\n    Mean value per sensor per state (first 10 sensors):")
log(f"    {'Sensor':<15} {'NORMAL':>10} {'BROKEN':>10} {'RECOVERING':>11} {'Δ(BROKEN-NORMAL)':>18}")
log(f"    {'-'*65}")
for s in sensor_cols[:10]:
    n_val = state_stats.loc["NORMAL", s] if "NORMAL" in state_stats.index else 0
    b_val = state_stats.loc["BROKEN", s] if "BROKEN" in state_stats.index else 0
    r_val = state_stats.loc["RECOVERING", s] if "RECOVERING" in state_stats.index else 0
    delta = b_val - n_val
    log(f"    {s:<15} {n_val:>10.3f} {b_val:>10.3f} {r_val:>11.3f} {delta:>+18.3f}")

# ---------------------------------------------------------------------------
# 7. Pre-fault sensor behaviour
# ---------------------------------------------------------------------------
log(f"\n[7] PRE-FAULT SENSOR BEHAVIOUR (60-minute window before BROKEN)")

# Find first BROKEN event start
broken_mask = df["machine_status"] == "BROKEN"
broken_indices = df.index[broken_mask].tolist()

if broken_indices:
    first_fault = broken_indices[0]
    window_start = max(0, first_fault - 60)
    pre_fault_df = df.iloc[window_start:first_fault]
    normal_df = df[df["machine_status"] == "NORMAL"].sample(
        min(60, (df["machine_status"] == "NORMAL").sum()), random_state=42
    )

    log(f"\n    Pre-fault window: rows {window_start}–{first_fault}")
    log(f"    Sensors with significant pre-fault change (|t-stat| > 2):")

    for s in sensor_cols[:20]:
        pre_vals = pre_fault_df[s].dropna()
        norm_vals = normal_df[s].dropna()
        if len(pre_vals) < 5 or len(norm_vals) < 5:
            continue
        t_stat, p_val = stats.ttest_ind(pre_vals, norm_vals, equal_var=False)
        if abs(t_stat) > 2:
            log(f"      {s}: t={t_stat:+.2f}  p={p_val:.4f}")

# ---------------------------------------------------------------------------
# 8. Sensor selection for 6-sensor machine model
# ---------------------------------------------------------------------------
log(f"\n[8] SENSOR SELECTION FOR 6-SENSOR MACHINE MODEL")
log("""
    Selection criteria:
    - One representative from each physical group (pressure, flow, vibration,
      temperature, current, efficiency)
    - Low near-constant risk (CV > 0.01)
    - Good fault discrimination (high |Δ BROKEN - NORMAL|)
    - Low missing rate (< 5%)
    - Not highly redundant with already-selected sensors

    Selected sensors:
""")

# Compute discrimination score = |mean_broken - mean_normal| / std_normal
disc_scores = {}
for s in sensor_cols:
    if s in near_const:
        disc_scores[s] = 0
        continue
    norm_vals = df.loc[df["machine_status"] == "NORMAL", s].dropna()
    brok_vals = df.loc[df["machine_status"] == "BROKEN", s].dropna()
    if len(brok_vals) < 10:
        disc_scores[s] = 0
        continue
    disc_scores[s] = abs(brok_vals.mean() - norm_vals.mean()) / (norm_vals.std() + 1e-9)

disc_series = pd.Series(disc_scores).sort_values(ascending=False)

# Pick best representative from each group (ensuring diversity)
group_ranges = {
    "pressure":    range(0, 8),
    "flow":        range(8, 16),
    "vibration":   range(16, 24),
    "temperature": range(24, 32),
    "current":     range(32, 40),
    "efficiency":  range(40, 48),
}

selected = {}
for group, idx_range in group_ranges.items():
    candidates = [f"sensor_{i:02d}" for i in idx_range if f"sensor_{i:02d}" not in near_const]
    # Sort by discrimination score
    candidates_sorted = sorted(candidates, key=lambda x: disc_scores.get(x, 0), reverse=True)
    best = candidates_sorted[0]
    miss_rate = missing_pct[best]
    selected[group] = {
        "sensor_id": best,
        "disc_score": disc_scores[best],
        "missing_pct": miss_rate,
        "mean_normal": df.loc[df["machine_status"] == "NORMAL", best].mean(),
        "mean_broken": df.loc[df["machine_status"] == "BROKEN", best].mean(),
    }

SELECTED_SENSORS = {}
log(f"    {'Group':<14} {'Sensor':<12} {'DiscScore':>10} {'Missing%':>9} {'Normal':>9} {'Broken':>9}")
log(f"    {'-'*65}")
for group, info in selected.items():
    log(f"    {group:<14} {info['sensor_id']:<12} {info['disc_score']:>10.3f} "
        f"{info['missing_pct']:>8.2f}% {info['mean_normal']:>9.3f} {info['mean_broken']:>9.3f}")
    SELECTED_SENSORS[group] = info["sensor_id"]

# These 6 sensors map to our machine model:
SENSOR_MAPPING = {
    "TEMP_01":     SELECTED_SENSORS["temperature"],
    "PRESSURE_01": SELECTED_SENSORS["pressure"],
    "VIBRATION_01":SELECTED_SENSORS["vibration"],
    "CURRENT_01":  SELECTED_SENSORS["current"],
    "FLOW_01":     SELECTED_SENSORS["flow"],
    "HUMIDITY_01": SELECTED_SENSORS["efficiency"],   # closest available
}
log(f"\n    Machine model sensor mapping:")
for machine_id, dataset_col in SENSOR_MAPPING.items():
    log(f"      {machine_id} → {dataset_col}")

# Save mapping for use by preprocessing pipeline
import json
mapping_path = os.path.join(OUT_DIR, "sensor_mapping.json")
with open(mapping_path, "w") as f:
    json.dump(SENSOR_MAPPING, f, indent=2)
log(f"\n    Sensor mapping saved → {mapping_path}")

# ---------------------------------------------------------------------------
# 9. Time series plot for selected sensors
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(len(SELECTED_SENSORS), 1, figsize=(18, 3 * len(SELECTED_SENSORS)),
                          sharex=True)
colors = {"NORMAL": "steelblue", "BROKEN": "firebrick", "RECOVERING": "darkorange"}

for ax, (group, col) in zip(axes, SELECTED_SENSORS.items()):
    for status_val, color in colors.items():
        mask = df["machine_status"] == status_val
        ax.scatter(df.loc[mask, "timestamp"], df.loc[mask, col],
                   c=color, s=0.5, alpha=0.4, label=status_val)
    ax.set_ylabel(f"{group}\n({col})", fontsize=8)
    ax.set_title(f"Selected sensor: {col} ({group})", fontsize=9)

axes[0].legend(markerscale=10, loc="upper right", fontsize=8)
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
fig.suptitle("Selected Sensors — Raw Signal by Machine State", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "04_selected_sensors_timeseries.png"), dpi=100)
plt.close(fig)

# ---------------------------------------------------------------------------
# 10. Distribution plots for selected sensors
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, (group, col) in zip(axes.flat, SELECTED_SENSORS.items()):
    for status_val, color in colors.items():
        vals = df.loc[df["machine_status"] == status_val, col].dropna()
        ax.hist(vals, bins=60, alpha=0.5, color=color, label=status_val, density=True)
    ax.set_title(f"{col} ({group})", fontsize=9)
    ax.set_xlabel("Value")
    ax.legend(fontsize=7)

fig.suptitle("Selected Sensors — Distribution by Machine State")
fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "05_selected_sensors_distributions.png"), dpi=100)
plt.close(fig)

# ---------------------------------------------------------------------------
# Save report
# ---------------------------------------------------------------------------
with open(REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

log(f"\n{'='*70}")
log(f"EDA COMPLETE")
log(f"  Report : {REPORT}")
log(f"  Plots  : {PLOT_DIR}")
log(f"{'='*70}")
log(f"\nNext step: python scripts/run_preprocessing.py")
