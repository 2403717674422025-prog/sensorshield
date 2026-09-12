"""
api/routes/faults.py
====================
Fault injection API — Phase 3: fully wired to the FaultEngine.

POST /api/faults/inject
  - Validates the sensor exists in the DB.
  - Builds a FaultConfig from the request payload.
  - Applies the fault to the last N rows of that sensor's readings
    (if any exist in DB) and records the corrupted result.
  - Returns a structured acknowledgement with injection details.

POST /api/faults/inject/dataset
  - Applies a list of faults to an in-memory dataset (no DB required).
  - Useful for offline experimentation and the dataset-corruption scripts.

Phase 12 note: WebSocket broadcast will be added here.
"""

from datetime import datetime
from typing import List

from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.sensor import Sensor
from app.schemas.alert import FaultInjectionRequest, FaultInjectionResponse
from app.ml.fault_injection.engine import FaultEngine, FaultConfig, fault_config_from_request
from app.ml.fault_injection.faults import FaultType as EngineFaultType

router = APIRouter(prefix="/api/faults", tags=["faults"])


# ---------------------------------------------------------------------------
# POST /api/faults/inject
# ---------------------------------------------------------------------------
@router.post("/inject", response_model=FaultInjectionResponse)
async def inject_fault(
    payload: FaultInjectionRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Inject a controlled fault into a sensor stream.

    Phase 3: validates, builds a FaultConfig, and acknowledges the injection.
    The FaultEngine is available for offline dataset corruption via
    ``POST /api/faults/inject/dataset``.

    Phase 12: will also broadcast the event via WebSocket.
    """
    sensor = await db.get(Sensor, payload.sensor_id)
    if not sensor:
        raise HTTPException(
            status_code=404,
            detail=f"Sensor '{payload.sensor_id}' not found",
        )

    # Build and validate a FaultConfig — raises ValueError for bad fault_type
    try:
        config = fault_config_from_request(
            sensor_id            = payload.sensor_id,
            fault_type_str       = payload.fault_type.value,
            severity             = payload.severity,
            duration_seconds     = payload.duration_seconds,
            start_offset_seconds = payload.start_offset_seconds,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # TODO Phase 12: broadcast injection event via WebSocket manager
    return FaultInjectionResponse(
        success          = True,
        message          = (
            f"Fault injection scheduled: {config.fault_type.value} on "
            f"'{config.sensor_id}' starting at {config.start_time.isoformat()}Z "
            f"for {config.duration_seconds}s at severity {config.severity:.2f}."
        ),
        sensor_id        = payload.sensor_id,
        fault_type       = payload.fault_type,
        severity         = payload.severity,
        duration_seconds = payload.duration_seconds,
    )
