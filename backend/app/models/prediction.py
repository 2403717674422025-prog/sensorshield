"""
models/prediction.py
====================
Downstream ML prediction for a machine at a point in time.
Stores both the raw model output AND the trust layer output so we can
retrospectively analyze how often trust labels aligned with outcomes.
"""

import enum
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class TrustStatus(str, enum.Enum):
    TRUSTED = "TRUSTED"
    CAUTION = "CAUTION"
    UNTRUSTED = "UNTRUSTED"
    UNKNOWN = "UNKNOWN"


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    machine_id: Mapped[str] = mapped_column(
        String(50), ForeignKey("machines.id", ondelete="CASCADE"), nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    # Downstream model output
    failure_probability: Mapped[float] = mapped_column(Float, nullable=False)
    model_uncertainty: Mapped[float] = mapped_column(
        Float, nullable=True
    )  # std of MC-Dropout samples or ensemble variance

    # Average sensor reliability at prediction time (0–100)
    avg_sensor_reliability: Mapped[float] = mapped_column(Float, nullable=True)
    min_sensor_reliability: Mapped[float] = mapped_column(Float, nullable=True)

    # Trust Engine output
    trust_score: Mapped[float] = mapped_column(Float, nullable=True)  # 0–100
    trust_status: Mapped[TrustStatus] = mapped_column(
        Enum(TrustStatus), default=TrustStatus.UNKNOWN, nullable=False
    )
    trust_reason: Mapped[str] = mapped_column(Text, nullable=True)

    # Relationships
    machine: Mapped["Machine"] = relationship(  # noqa: F821
        "Machine", back_populates="predictions"
    )

    __table_args__ = (
        Index("ix_predictions_machine_ts", "machine_id", "timestamp"),
    )

    def __repr__(self) -> str:
        return (
            f"<Prediction machine={self.machine_id} "
            f"ts={self.timestamp} prob={self.failure_probability:.3f} "
            f"trust={self.trust_status}>"
        )
