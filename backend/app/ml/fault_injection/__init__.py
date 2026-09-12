"""
ml/fault_injection/__init__.py
================================
Fault Injection Engine — Phase 3.

Public surface:
    FaultEngine        — applies faults to a DataFrame of sensor readings
    FaultConfig        — parameters for a single fault injection
    FaultType          — enum of all supported fault types (mirrors alert model)
    apply_fault        — stateless helper: apply one fault to a numpy array
"""

from .engine import FaultConfig, FaultEngine
from .faults import (
    FaultType,
    apply_bias,
    apply_drift,
    apply_intermittent,
    apply_missing,
    apply_noise,
    apply_sampling_failure,
    apply_spike,
    apply_stuck,
)

__all__ = [
    "FaultConfig",
    "FaultEngine",
    "FaultType",
    "apply_bias",
    "apply_drift",
    "apply_intermittent",
    "apply_missing",
    "apply_noise",
    "apply_sampling_failure",
    "apply_spike",
    "apply_stuck",
]
