"""
ml/fault_injection/faults.py
=============================
Stateless, vectorised implementations of all 8 fault types.

Every function takes a *copy* of a numpy array (the signal segment that
falls inside the fault window) and returns a corrupted version of the
same shape.  None of them mutate the original array.

Fault types
-----------
1.  drift             — slow linear ramp added to the signal
2.  bias              — constant offset added to the signal
3.  noise             — additive Gaussian noise scaled by severity
4.  stuck             — signal frozen at the value at fault-start
5.  missing           — values replaced by NaN (sensor dropout)
6.  intermittent      — random NaN bursts (packet-loss style dropout)
7.  spike             — isolated large impulse spikes
8.  sampling_failure  — readings replaced by NaN at a configurable rate
                         (models a sensor that fails to deliver samples)

Parameter contract
------------------
All functions share the signature:
    func(signal: np.ndarray, severity: float, rng: np.random.Generator,
         **kwargs) -> np.ndarray

  severity : float in [0, 1]  — 0 = barely perceptible, 1 = extreme fault
  rng      : NumPy random generator (seeded by FaultEngine for reproducibility)
  **kwargs : fault-specific tuning knobs (all have sensible defaults)
"""

from __future__ import annotations

import enum
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# FaultType enum
# ---------------------------------------------------------------------------
class FaultType(str, enum.Enum):
    """Mirrors ``app.models.alert.FaultType`` — kept separate to avoid
    importing SQLAlchemy from pure-Python ML code."""
    DRIFT              = "DRIFT"
    BIAS               = "BIAS"
    NOISE              = "NOISE"
    STUCK              = "STUCK"
    MISSING            = "MISSING"
    INTERMITTENT       = "INTERMITTENT"
    SPIKE              = "SPIKE"
    SAMPLING_FAILURE   = "SAMPLING_FAILURE"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _signal_scale(signal: np.ndarray) -> float:
    """Return the std of the signal (or 1.0 if constant / very short)."""
    std = float(np.nanstd(signal))
    return std if std > 1e-9 else 1.0


def _signal_span(signal: np.ndarray) -> float:
    """Return the range (max - min) of the signal (or 1.0 if constant)."""
    span = float(np.nanmax(signal) - np.nanmin(signal))
    return span if span > 1e-9 else 1.0


# ---------------------------------------------------------------------------
# 1. Drift — slow linear ramp
# ---------------------------------------------------------------------------
def apply_drift(
    signal: np.ndarray,
    severity: float,
    rng: np.random.Generator,
    *,
    max_drift_factor: float = 2.0,
) -> np.ndarray:
    """
    Add a monotonically increasing (or decreasing) ramp to the signal.

    The ramp endpoint is ``severity * max_drift_factor * signal_std``.
    A random sign (+ / -) is drawn so drift can go either direction.

    Parameters
    ----------
    max_drift_factor : float
        Multiplier on signal std to set the total drift magnitude.
        Severity=1 produces a drift equal to max_drift_factor * std.
    """
    signal = signal.copy().astype(float)
    n = len(signal)
    if n == 0:
        return signal

    scale = _signal_scale(signal)
    total_drift = severity * max_drift_factor * scale
    direction   = rng.choice([-1.0, 1.0])
    ramp        = np.linspace(0.0, total_drift * direction, n)
    return signal + ramp


# ---------------------------------------------------------------------------
# 2. Bias — constant additive offset
# ---------------------------------------------------------------------------
def apply_bias(
    signal: np.ndarray,
    severity: float,
    rng: np.random.Generator,
    *,
    max_bias_factor: float = 3.0,
) -> np.ndarray:
    """
    Add a constant offset to the entire signal segment.

    Offset = ``severity * max_bias_factor * signal_std``.
    Direction is drawn randomly.

    Parameters
    ----------
    max_bias_factor : float
        At severity=1 the bias is max_bias_factor × std.
    """
    signal = signal.copy().astype(float)
    if len(signal) == 0:
        return signal

    scale     = _signal_scale(signal)
    magnitude = severity * max_bias_factor * scale
    direction = rng.choice([-1.0, 1.0])
    return signal + direction * magnitude


# ---------------------------------------------------------------------------
# 3. Noise — additive Gaussian noise
# ---------------------------------------------------------------------------
def apply_noise(
    signal: np.ndarray,
    severity: float,
    rng: np.random.Generator,
    *,
    max_noise_factor: float = 2.0,
) -> np.ndarray:
    """
    Add zero-mean Gaussian noise whose std scales with severity.

    Noise std = ``severity * max_noise_factor * signal_std``.

    Parameters
    ----------
    max_noise_factor : float
        At severity=1 the noise std equals max_noise_factor × signal_std.
    """
    signal = signal.copy().astype(float)
    if len(signal) == 0:
        return signal

    scale    = _signal_scale(signal)
    noise_std = severity * max_noise_factor * scale
    noise     = rng.normal(loc=0.0, scale=noise_std, size=len(signal))
    return signal + noise


# ---------------------------------------------------------------------------
# 4. Stuck — sensor frozen at start value
# ---------------------------------------------------------------------------
def apply_stuck(
    signal: np.ndarray,
    severity: float,
    rng: np.random.Generator,
    *,
    jitter_factor: float = 0.01,
) -> np.ndarray:
    """
    Replace the signal with the value at the start of the fault window.

    A tiny amount of jitter proportional to ``(1 - severity)`` is added
    to simulate a near-stuck sensor.  At severity=1 jitter is zero.

    Parameters
    ----------
    jitter_factor : float
        Controls the residual jitter amplitude at low severity.
    """
    signal = signal.copy().astype(float)
    n = len(signal)
    if n == 0:
        return signal

    stuck_value = float(np.nanmean(signal[:max(1, n // 10)]))
    scale       = _signal_scale(signal)
    jitter_std  = (1.0 - severity) * jitter_factor * scale
    jitter      = rng.normal(loc=0.0, scale=max(jitter_std, 1e-12), size=n)
    return np.full(n, stuck_value, dtype=float) + jitter


# ---------------------------------------------------------------------------
# 5. Missing — complete dropout (all NaN)
# ---------------------------------------------------------------------------
def apply_missing(
    signal: np.ndarray,
    severity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Replace all values in the fault window with NaN.

    ``severity`` is deliberately ignored here because the fault definition
    is *total* absence of the signal.  A partial-missing scenario is covered
    by ``apply_intermittent``.
    """
    return np.full(len(signal), np.nan, dtype=float)


# ---------------------------------------------------------------------------
# 6. Intermittent — random burst dropout
# ---------------------------------------------------------------------------
def apply_intermittent(
    signal: np.ndarray,
    severity: float,
    rng: np.random.Generator,
    *,
    burst_length: int = 5,
) -> np.ndarray:
    """
    Randomly set contiguous bursts of values to NaN.

    The fraction of the signal lost scales linearly with severity:
    at severity=0.2 ~20 % is NaN; at severity=1.0 ~100 %.

    Parameters
    ----------
    burst_length : int
        Approximate number of consecutive NaN samples per dropout event.
    """
    signal = signal.copy().astype(float)
    n      = len(signal)
    if n == 0 or burst_length <= 0:
        return signal

    target_nan = int(round(severity * n))
    n_bursts   = max(1, target_nan // burst_length)

    for _ in range(n_bursts):
        start = int(rng.integers(0, n))
        end   = min(n, start + burst_length)
        signal[start:end] = np.nan

    return signal


# ---------------------------------------------------------------------------
# 7. Spike — isolated impulse spikes
# ---------------------------------------------------------------------------
def apply_spike(
    signal: np.ndarray,
    severity: float,
    rng: np.random.Generator,
    *,
    max_spike_factor: float = 5.0,
    spike_rate: float = 0.05,
) -> np.ndarray:
    """
    Inject isolated large-amplitude spikes at random positions.

    Spike amplitude = ``severity * max_spike_factor * signal_std``.
    Each spike occupies a single sample.

    Parameters
    ----------
    max_spike_factor : float
        At severity=1 spike amplitude = max_spike_factor × std.
    spike_rate : float
        Fraction of samples that receive a spike.  Minimum 1 spike always.
    """
    signal = signal.copy().astype(float)
    n = len(signal)
    if n == 0:
        return signal

    scale           = _signal_scale(signal)
    spike_amplitude = severity * max_spike_factor * scale
    n_spikes        = max(1, int(round(spike_rate * n)))

    spike_positions = rng.choice(n, size=n_spikes, replace=False)
    for pos in spike_positions:
        direction        = rng.choice([-1.0, 1.0])
        signal[pos]     += direction * spike_amplitude

    return signal


# ---------------------------------------------------------------------------
# 8. Sampling failure — random sample-level NaN at a configurable rate
# ---------------------------------------------------------------------------
def apply_sampling_failure(
    signal: np.ndarray,
    severity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Randomly drop individual samples (not bursts) to NaN.

    This models a sensor that fails to transmit individual readings
    (e.g., packet loss, ADC timeout).

    Drop rate = ``severity``.  At severity=1.0 every sample is dropped.
    """
    signal       = signal.copy().astype(float)
    n            = len(signal)
    if n == 0:
        return signal

    drop_mask    = rng.random(n) < severity
    signal[drop_mask] = np.nan
    return signal


# ---------------------------------------------------------------------------
# Dispatch table — maps FaultType → function
# ---------------------------------------------------------------------------
FAULT_DISPATCH = {
    FaultType.DRIFT:             apply_drift,
    FaultType.BIAS:              apply_bias,
    FaultType.NOISE:             apply_noise,
    FaultType.STUCK:             apply_stuck,
    FaultType.MISSING:           apply_missing,
    FaultType.INTERMITTENT:      apply_intermittent,
    FaultType.SPIKE:             apply_spike,
    FaultType.SAMPLING_FAILURE:  apply_sampling_failure,
}
