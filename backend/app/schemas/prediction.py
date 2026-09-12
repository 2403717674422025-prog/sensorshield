"""
schemas/prediction.py
=====================
Pydantic schemas for predictions and trust engine responses.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from app.models.prediction import TrustStatus


class PredictionResponse(BaseModel):
    id: int
    machine_id: str
    timestamp: datetime
    failure_probability: float = Field(..., ge=0.0, le=1.0)
    model_uncertainty: Optional[float]
    avg_sensor_reliability: Optional[float]
    min_sensor_reliability: Optional[float]
    trust_score: Optional[float] = Field(None, ge=0, le=100)
    trust_status: TrustStatus
    trust_reason: Optional[str]

    model_config = {"from_attributes": True}
