"""
ml/fault_injection/engine.py
==============================
FaultEngine — applies one or many FaultConfig objects to a pandas DataFrame
containing time-series sensor readings.

Design principles
-----------------
* The engine works on *copies* — the original DataFrame is never mutated.
* Faults are bounded by (start_time, start_time + duration).
* Multiple faults on the same sensor compose sequentially in the order
  they are applied.
* The engine is stateless between calls; state is carried entirely by the
  FaultConfig objects passed to ``apply()``.
* Seeds are derived deterministically from the fault parameters, making
  runs reproducible without a global random state.

Expected DataFrame format
-------------------------
The DataFrame must have:
  - a ``timestamp`` column (datetime-like, used to locate the fault window)
  - one column per sensor (column name == sensor_id)
  - any number of other columns (passed through unchanged)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone, UTC
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .faults import FAULT_DISPATCH, FaultType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FaultConfig — parameters for a single fault event
# ---------------------------------------------------------------------------
@dataclass
class FaultConfig:
    """
    All parameters that describe one fault injection event.

    Attributes
    ----------
    sensor_id : str
        Column name in the DataFrame this fault targets.
    fault_type : FaultType
        Which corruption to apply.
    start_time : datetime
        When the fault begins.
    duration_seconds : int
        How long the fault lasts (seconds).  Must be >= 1.
    severity : float
        Fault intensity in [0.0, 1.0].  0 = barely perceptible; 1 = extreme.
    fault_kwargs : dict
        Optional keyword arguments forwarded to the fault function
        (e.g., ``burst_length`` for intermittent, ``spike_rate`` for spike).
    label : str
        Human-readable label written to the ``fault_label`` column in the
        corrupted DataFrame (useful for ML evaluation).
    """
    sensor_id:        str
    fault_type:       FaultType
    start_time:       datetime
    duration_seconds: int
    severity:         float
    fault_kwargs:     Dict = field(default_factory=dict)
    label:            str  = ""

    def __post_init__(self):
        if not (0.0 <= self.severity <= 1.0):
            raise ValueError(
                f"severity must be in [0.0, 1.0], got {self.severity}"
            )
        if self.duration_seconds < 1:
            raise ValueError(
                f"duration_seconds must be >= 1, got {self.duration_seconds}"
            )
        if not self.label:
            self.label = f"{self.fault_type.value}@{self.sensor_id}"

    @property
    def end_time(self) -> datetime:
        return self.start_time + timedelta(seconds=self.duration_seconds)

    def seed(self) -> int:
        """
        Deterministic seed derived from fault parameters.
        Two identical FaultConfigs always produce the same random sequence.
        """
        key = (
            f"{self.sensor_id}|{self.fault_type}|"
            f"{self.start_time.isoformat()}|{self.duration_seconds}|"
            f"{self.severity:.6f}"
        )
        digest = hashlib.md5(key.encode()).hexdigest()[:8]
        return int(digest, 16) % (2 ** 31)


# ---------------------------------------------------------------------------
# FaultEngine
# ---------------------------------------------------------------------------
class FaultEngine:
    """
    Applies a sequence of FaultConfig objects to a sensor DataFrame.

    Usage
    -----
    >>> engine = FaultEngine()
    >>> corrupted_df, label_df = engine.apply(clean_df, faults)

    Parameters
    ----------
    timestamp_col : str
        Name of the timestamp column (default: ``"timestamp"``).
    """

    def __init__(self, timestamp_col: str = "timestamp") -> None:
        self.timestamp_col = timestamp_col

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def apply(
        self,
        df: pd.DataFrame,
        faults: Sequence[FaultConfig],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply all faults to the DataFrame and return:
          1. ``corrupted_df``  — DataFrame with corrupted sensor values
          2. ``label_df``      — Boolean DataFrame (same shape as sensor cols)
                                  True wherever a fault was active

        The original ``df`` is NOT modified.

        Parameters
        ----------
        df     : Input DataFrame (must contain ``timestamp_col``).
        faults : Sequence of FaultConfig describing each injection.

        Returns
        -------
        (corrupted_df, label_df)
        """
        if self.timestamp_col not in df.columns:
            raise ValueError(
                f"DataFrame must contain column '{self.timestamp_col}'. "
                f"Available: {list(df.columns)}"
            )

        corrupted = df.copy()
        # Ensure timestamps are timezone-naive or all tz-aware for comparison
        ts_series = pd.to_datetime(corrupted[self.timestamp_col])
        corrupted[self.timestamp_col] = ts_series

        # Label DataFrame — booleans, same sensor columns
        sensor_cols = [f.sensor_id for f in faults]
        sensor_cols_unique = list(dict.fromkeys(sensor_cols))  # preserve order
        # Only track labels for columns that actually exist
        existing_sensor_cols = [c for c in sensor_cols_unique if c in corrupted.columns]
        label_df = pd.DataFrame(
            False,
            index=corrupted.index,
            columns=existing_sensor_cols,
        )

        for fault in faults:
            corrupted, label_df = self._apply_one(
                corrupted, label_df, fault
            )

        return corrupted, label_df

    def apply_single(
        self,
        df: pd.DataFrame,
        fault: FaultConfig,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Convenience wrapper: apply a single fault."""
        return self.apply(df, [fault])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _apply_one(
        self,
        df: pd.DataFrame,
        label_df: pd.DataFrame,
        fault: FaultConfig,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply one FaultConfig in-place (on the copy)."""

        if fault.sensor_id not in df.columns:
            logger.warning(
                "FaultEngine: sensor_id '%s' not found in DataFrame columns %s — skipping",
                fault.sensor_id,
                list(df.columns),
            )
            return df, label_df

        if fault.fault_type not in FAULT_DISPATCH:
            raise ValueError(
                f"Unknown fault type: {fault.fault_type}. "
                f"Supported: {list(FAULT_DISPATCH.keys())}"
            )

        # Build fault window mask
        ts = df[self.timestamp_col]
        start = self._to_naive(fault.start_time)
        end   = self._to_naive(fault.end_time)

        ts_naive = ts.apply(self._to_naive)
        mask     = (ts_naive >= start) & (ts_naive < end)

        n_affected = mask.sum()
        if n_affected == 0:
            logger.debug(
                "FaultEngine: fault '%s' on '%s' produced 0 matching rows "
                "(start=%s, end=%s) — check timestamp alignment",
                fault.fault_type, fault.sensor_id, start, end,
            )
            return df, label_df

        # Extract the signal segment to corrupt
        signal_segment = df.loc[mask, fault.sensor_id].to_numpy(dtype=float)

        # Deterministic RNG seeded per fault
        rng = np.random.default_rng(fault.seed())

        # Dispatch to the fault function
        fn = FAULT_DISPATCH[fault.fault_type]
        corrupted_segment = fn(
            signal_segment,
            severity=fault.severity,
            rng=rng,
            **fault.fault_kwargs,
        )

        df.loc[mask, fault.sensor_id] = corrupted_segment

        # Record labels
        if fault.sensor_id in label_df.columns:
            label_df.loc[mask, fault.sensor_id] = True

        logger.debug(
            "FaultEngine: applied %s to '%s' over %d rows (severity=%.2f)",
            fault.fault_type, fault.sensor_id, n_affected, fault.severity,
        )
        return df, label_df

    @staticmethod
    def _to_naive(dt) -> datetime:
        """Strip timezone info for safe comparison."""
        if isinstance(dt, (pd.Timestamp,)):
            return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt.to_pydatetime()
        if hasattr(dt, "tzinfo") and dt.tzinfo is not None:
            return dt.replace(tzinfo=None)
        return dt


# ---------------------------------------------------------------------------
# Convenience: build a FaultConfig from the API request schema
# ---------------------------------------------------------------------------
def fault_config_from_request(
    sensor_id:          str,
    fault_type_str:     str,
    severity:           float,
    duration_seconds:   int,
    start_offset_seconds: int = 0,
    reference_time:     Optional[datetime] = None,
) -> FaultConfig:
    """
    Translate API request fields into a FaultConfig.

    Parameters
    ----------
    reference_time : datetime, optional
        Base time for computing ``start_time``.  Defaults to ``datetime.utcnow()``.
    """
    if reference_time is None:
        reference_time = datetime.now(UTC).replace(tzinfo=None)

    start_time = reference_time + timedelta(seconds=start_offset_seconds)

    # Normalise fault type string — try both the "SAMPLING_FAILURE" form and
    # the legacy "sampling failure" (from FaultType enum in alert.py that may
    # not have SAMPLING_FAILURE).
    fault_type_str = fault_type_str.upper().replace(" ", "_")
    try:
        ft = FaultType(fault_type_str)
    except ValueError:
        valid = [e.value for e in FaultType]
        raise ValueError(
            f"Unknown fault_type '{fault_type_str}'. Valid values: {valid}"
        )

    return FaultConfig(
        sensor_id        = sensor_id,
        fault_type       = ft,
        start_time       = start_time,
        duration_seconds = duration_seconds,
        severity         = severity,
    )
