"""
models/reconstruction.py
========================
Stores reconstructed sensor values alongside observed values.
When a sensor is missing or corrupted, the Signal Reconstructor estimates
what the value *should* be. This table lets the dashboard overlay the
reconstructed signal on top of the raw signal.
"""

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class ReconstructedReading(Base):
    __tablename__ = "reconstructed_readings"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    sensor_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("sensors.id", ondelete="CASCADE"), nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    # The raw observed value (may be None if the reading was missing entirely)
    observed_value: Mapped[float] = mapped_column(Float, nullable=True)

    # The value estimated by the reconstruction model
    reconstructed_value: Mapped[float] = mapped_column(Float, nullable=False)

    # Confidence in the reconstruction (0.0–1.0)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    # Which reconstruction method produced this estimate
    method: Mapped[str] = mapped_column(
        String(50), nullable=True
    )  # e.g. "lstm", "linear_regression", "neighbor_interpolation"

    # Relationships
    sensor: Mapped["Sensor"] = relationship(  # noqa: F821
        "Sensor", back_populates="reconstructed_readings"
    )

    __table_args__ = (
        Index("ix_reconstructed_sensor_ts", "sensor_id", "timestamp"),
    )

    def __repr__(self) -> str:
        return (
            f"<ReconstructedReading sensor={self.sensor_id} "
            f"ts={self.timestamp} observed={self.observed_value} "
            f"reconstructed={self.reconstructed_value:.3f} conf={self.confidence:.2f}>"
        )
