"""
models/__init__.py
==================
Import all models here so Alembic's autogenerate can discover them,
and so a single import of this package guarantees all tables are registered
with the SQLAlchemy metadata before create_all() or migration runs.
"""

from app.models.machine import Machine, MachineStatus
from app.models.sensor import Sensor, SensorStatus, SensorType
from app.models.reading import SensorReading
from app.models.health import SensorHealth
from app.models.prediction import Prediction, TrustStatus
from app.models.alert import Alert, AlertSeverity, AlertStatus, FaultType
from app.models.reconstruction import ReconstructedReading

__all__ = [
    "Machine",
    "MachineStatus",
    "Sensor",
    "SensorStatus",
    "SensorType",
    "SensorReading",
    "SensorHealth",
    "Prediction",
    "TrustStatus",
    "Alert",
    "AlertSeverity",
    "AlertStatus",
    "FaultType",
    "ReconstructedReading",
]
