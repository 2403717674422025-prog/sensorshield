"""
models/machine.py
=================
Represents a physical or simulated industrial machine.
One machine contains multiple sensors.
"""

import enum
from datetime import datetime

from sqlalchemy import DateTime, Enum, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class MachineStatus(str, enum.Enum):
    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    DEGRADED = "DEGRADED"
    MAINTENANCE = "MAINTENANCE"
    UNKNOWN = "UNKNOWN"


class Machine(Base):
    __tablename__ = "machines"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    location: Mapped[str] = mapped_column(String(200), nullable=True)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    status: Mapped[MachineStatus] = mapped_column(
        Enum(MachineStatus), default=MachineStatus.ONLINE, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    sensors: Mapped[list["Sensor"]] = relationship(  # noqa: F821
        "Sensor", back_populates="machine", cascade="all, delete-orphan"
    )
    predictions: Mapped[list["Prediction"]] = relationship(  # noqa: F821
        "Prediction", back_populates="machine", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Machine id={self.id} name={self.name} status={self.status}>"
