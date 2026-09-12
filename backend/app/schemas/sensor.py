"""
schemas/sensor.py
=================
Pydantic schemas for sensor and machine API requests/responses.
Schemas are separate from ORM models intentionally:
- ORM models own database structure
- Schemas own API contract (what goes in, what comes out)
This separation means we can evolve the DB schema without breaking API
consumers and vice versa.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from app.models.sensor import SensorStatus, SensorType
from app.models.machine import MachineStatus


# ---------------------------------------------------------------------------
# Machine schemas
# ---------------------------------------------------------------------------
class MachineBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    location: Optional[str] = Field(None, max_length=200)
    description: Optional[str] = None


class MachineCreate(MachineBase):
    id: str = Field(..., min_length=1, max_length=50, pattern=r"^[A-Za-z0-9_\-]+$")


class MachineResponse(MachineBase):
    id: str
    status: MachineStatus
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Sensor schemas
# ---------------------------------------------------------------------------
class SensorBase(BaseModel):
    sensor_type: SensorType
    unit: str = Field(..., min_length=1, max_length=20)
    name: Optional[str] = Field(None, max_length=100)
    min_valid_value: Optional[float] = None
    max_valid_value: Optional[float] = None

    @field_validator("min_valid_value", "max_valid_value", mode="before")
    @classmethod
    def validate_range(cls, v):
        return v  # Further cross-field validation done at the model level


class SensorCreate(SensorBase):
    id: str = Field(..., min_length=1, max_length=50, pattern=r"^[A-Za-z0-9_\-]+$")
    machine_id: str = Field(..., min_length=1, max_length=50)


class SensorResponse(SensorBase):
    id: str
    machine_id: str
    status: SensorStatus
    installation_date: Optional[datetime]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Sensor reading schemas
# ---------------------------------------------------------------------------
class SensorReadingCreate(BaseModel):
    """
    Schema for a single incoming reading.
    This is the shape of the JSON that ingestion accepts via REST or MQTT.
    """
    timestamp: datetime
    machine_id: str = Field(..., max_length=50)
    sensor_id: str = Field(..., max_length=50)
    sensor_type: SensorType
    value: float
    unit: str = Field(..., max_length=20)


class SensorReadingResponse(BaseModel):
    id: int
    sensor_id: str
    machine_id: str
    timestamp: datetime
    value: float
    is_valid: bool
    quality_flag: Optional[str]

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Sensor health schemas
# ---------------------------------------------------------------------------
class SensorHealthResponse(BaseModel):
    id: int
    sensor_id: str
    timestamp: datetime
    health_score: float = Field(..., ge=0, le=100)
    anomaly_score: Optional[float]
    drift_score: Optional[float]
    noise_score: Optional[float]
    missing_data_score: Optional[float]
    consistency_score: Optional[float]
    reconstruction_error: Optional[float]
    status: SensorStatus
    status_confidence: Optional[float]
    reason: Optional[str]

    model_config = {"from_attributes": True}
