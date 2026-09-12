"""
models/sensor.py
================
Represents a single sensor attached to a machine.
Sensor type determines what validation rules and thresholds apply.
"""

import enum
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class SensorType(str, enum.Enum):
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    VIBRATION = "vibration"
    CURRENT = "current"
    FLOW = "flow"
    HUMIDITY = "humidity"
    VOLTAGE = "voltage"
    SPEED = "speed"
    UNKNOWN = "unknown"


class SensorStatus(str, enum.Enum):
    HEALTHY = "HEALTHY"
    DRIFTING = "DRIFTING"
    NOISY = "NOISY"
    STUCK = "STUCK"
    INTERMITTENT = "INTERMITTENT"
    MISSING = "MISSING"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


class Sensor(Base):
    __tablename__ = "sensors"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    machine_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("machines.id", ondelete="CASCADE"), nullable=False
    )
    sensor_type: Mapped[SensorType] = mapped_column(
        Enum(SensorType, values_callable=lambda x: [e.value for e in x]), nullable=False
    )
    name: Mapped[str] = mapped_column(String(100), nullable=True)
    unit: Mapped[str] = mapped_column(String(20), nullable=False)

    # Valid range for this sensor — used by the data validator
    min_valid_value: Mapped[float] = mapped_column(Float, nullable=True)
    max_valid_value: Mapped[float] = mapped_column(Float, nullable=True)

    status: Mapped[SensorStatus] = mapped_column(
        Enum(SensorStatus), default=SensorStatus.HEALTHY, nullable=False
    )
    installation_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    machine: Mapped["Machine"] = relationship("Machine", back_populates="sensors")  # noqa: F821
    readings: Mapped[list["SensorReading"]] = relationship(  # noqa: F821
        "SensorReading", back_populates="sensor", cascade="all, delete-orphan"
    )
    health_records: Mapped[list["SensorHealth"]] = relationship(  # noqa: F821
        "SensorHealth", back_populates="sensor", cascade="all, delete-orphan"
    )
    alerts: Mapped[list["Alert"]] = relationship(  # noqa: F821
        "Alert", back_populates="sensor", cascade="all, delete-orphan"
    )
    reconstructed_readings: Mapped[list["ReconstructedReading"]] = relationship(  # noqa: F821
        "ReconstructedReading", back_populates="sensor", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<Sensor id={self.id} type={self.sensor_type} "
            f"machine={self.machine_id} status={self.status}>"
        )
