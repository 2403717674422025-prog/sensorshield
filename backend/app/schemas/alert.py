"""
schemas/alert.py
================
Pydantic schemas for alert API responses and fault injection requests.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from app.models.alert import AlertSeverity, AlertStatus, FaultType


class AlertResponse(BaseModel):
    id: int
    sensor_id: Optional[str]
    machine_id: Optional[str]
    timestamp: datetime
    severity: AlertSeverity
    fault_type: FaultType
    status: AlertStatus
    description: str
    impact: Optional[str]
    recommended_action: Optional[str]
    health_score_at_alert: Optional[float]
    trust_score_at_alert: Optional[float]
    resolved_at: Optional[datetime]

    model_config = {"from_attributes": True}


class FaultInjectionRequest(BaseModel):
    """
    Schema for the fault injection API endpoint.
    Allows the dashboard to trigger controlled faults for demonstration.
    """
    sensor_id: str = Field(..., max_length=50)
    fault_type: FaultType
    severity: float = Field(..., ge=0.0, le=1.0, description="0=minimal, 1=severe")
    duration_seconds: int = Field(
        ..., ge=1, le=3600, description="How long the fault lasts"
    )
    # Optional: override start time (defaults to now)
    start_offset_seconds: int = Field(
        0, ge=0, description="Seconds from now when fault begins"
    )


class FaultInjectionResponse(BaseModel):
    success: bool
    message: str
    sensor_id: str
    fault_type: FaultType
    severity: float
    duration_seconds: int
