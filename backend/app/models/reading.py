"""
models/reading.py
=================
Raw sensor readings as they arrive from the ingestion pipeline.
This is the source of truth — never modified after insertion.
All downstream processing (anomaly scores, health scores) references
readings by sensor_id + timestamp, not by modifying this table.
"""

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class SensorReading(Base):
    __tablename__ = "sensor_readings"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    sensor_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("sensors.id", ondelete="CASCADE"), nullable=False
    )
    machine_id: Mapped[str] = mapped_column(String(50), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    value: Mapped[float] = mapped_column(Float, nullable=False)

    # Data quality flags set by the validator (not by ML models)
    is_valid: Mapped[bool] = mapped_column(default=True, nullable=False)
    quality_flag: Mapped[str] = mapped_column(
        String(50), nullable=True
    )  # e.g. "RANGE_VIOLATION", "DUPLICATE", "STALE"

    # Relationships
    sensor: Mapped["Sensor"] = relationship("Sensor", back_populates="readings")  # noqa: F821

    # Composite index on (sensor_id, timestamp) — the most common query pattern
    __table_args__ = (
        Index("ix_sensor_readings_sensor_ts", "sensor_id", "timestamp"),
        Index("ix_sensor_readings_machine_ts", "machine_id", "timestamp"),
    )

    def __repr__(self) -> str:
        return (
            f"<SensorReading sensor={self.sensor_id} "
            f"ts={self.timestamp} value={self.value}>"
        )
