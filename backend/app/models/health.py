"""
models/health.py
================
Sensor health record produced by the Reliability Engine on each evaluation cycle.
Stores the composite health score plus individual sub-scores so the dashboard
can show users *why* a sensor is degraded, not just *that* it is.
"""

from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.sensor import SensorStatus


class SensorHealth(Base):
    __tablename__ = "sensor_health"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    sensor_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("sensors.id", ondelete="CASCADE"), nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    # Composite health score 0–100 (100 = perfectly healthy)
    health_score: Mapped[float] = mapped_column(Float, nullable=False)

    # Sub-scores from individual detectors (all 0–100)
    anomaly_score: Mapped[float] = mapped_column(Float, nullable=True)
    drift_score: Mapped[float] = mapped_column(Float, nullable=True)
    noise_score: Mapped[float] = mapped_column(Float, nullable=True)
    missing_data_score: Mapped[float] = mapped_column(Float, nullable=True)
    consistency_score: Mapped[float] = mapped_column(Float, nullable=True)
    reconstruction_error: Mapped[float] = mapped_column(Float, nullable=True)

    # Inferred sensor state and confidence
    status: Mapped[SensorStatus] = mapped_column(
        Enum(SensorStatus), nullable=False
    )
    status_confidence: Mapped[float] = mapped_column(
        Float, nullable=True
    )  # 0.0–1.0

    # Human-readable reason (populated by the explainability layer)
    reason: Mapped[str] = mapped_column(Text, nullable=True)

    # Relationships
    sensor: Mapped["Sensor"] = relationship(  # noqa: F821
        "Sensor", back_populates="health_records"
    )

    __table_args__ = (
        Index("ix_sensor_health_sensor_ts", "sensor_id", "timestamp"),
    )

    def __repr__(self) -> str:
        return (
            f"<SensorHealth sensor={self.sensor_id} "
            f"ts={self.timestamp} score={self.health_score} status={self.status}>"
        )
