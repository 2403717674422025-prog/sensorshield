"""
api/routes/sensors.py
=====================
Endpoints for sensor registration, reading ingestion, and health retrieval.
POST /api/sensors/readings is the primary ingestion endpoint — this is where
real-time data (from the simulator, MQTT bridge, or ESP32) enters the system.

Phase 4: ingestion now triggers the full ML/reliability pipeline.
"""

from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.sensor import Sensor
from app.models.reading import SensorReading
from app.models.health import SensorHealth
from app.schemas.sensor import (
    SensorCreate,
    SensorResponse,
    SensorReadingCreate,
    SensorReadingResponse,
    SensorHealthResponse,
)
from app.services.sensor_pipeline import process_reading

router = APIRouter(prefix="/api/sensors", tags=["sensors"])


# ---------------------------------------------------------------------------
# Sensor registration
# ---------------------------------------------------------------------------
@router.get("", response_model=List[SensorResponse])
async def list_sensors(
    machine_id: Optional[str] = Query(None, description="Filter by machine"),
    db: AsyncSession = Depends(get_db),
):
    """Return all sensors, optionally filtered by machine_id."""
    q = select(Sensor).order_by(Sensor.id)
    if machine_id:
        q = q.where(Sensor.machine_id == machine_id)
    result = await db.execute(q)
    return result.scalars().all()


@router.get("/{sensor_id}", response_model=SensorResponse)
async def get_sensor(sensor_id: str, db: AsyncSession = Depends(get_db)):
    sensor = await db.get(Sensor, sensor_id)
    if not sensor:
        raise HTTPException(status_code=404, detail=f"Sensor '{sensor_id}' not found")
    return sensor


@router.post("", response_model=SensorResponse, status_code=status.HTTP_201_CREATED)
async def create_sensor(payload: SensorCreate, db: AsyncSession = Depends(get_db)):
    """Register a new sensor on an existing machine."""
    existing = await db.get(Sensor, payload.id)
    if existing:
        raise HTTPException(status_code=409, detail=f"Sensor '{payload.id}' already exists")
    sensor = Sensor(**payload.model_dump())
    db.add(sensor)
    await db.flush()
    await db.refresh(sensor)
    return sensor


# ---------------------------------------------------------------------------
# Reading ingestion
# ---------------------------------------------------------------------------
@router.post("/readings", response_model=SensorReadingResponse, status_code=201)
async def ingest_reading(
    payload: SensorReadingCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Ingest a single sensor reading and run the full ML pipeline:
      raw reading → preprocessing → anomaly score → health evaluation
      → SensorHealth record → failure prediction → alert check → WebSocket broadcast
    """
    # Validate sensor exists
    sensor = await db.get(Sensor, payload.sensor_id)
    if not sensor:
        raise HTTPException(
            status_code=404,
            detail=f"Sensor '{payload.sensor_id}' not found. Register it first.",
        )

    # Persist the raw reading first — this is the source of truth
    reading = SensorReading(
        sensor_id  = payload.sensor_id,
        machine_id = payload.machine_id,
        timestamp  = payload.timestamp,
        value      = payload.value,
        is_valid   = True,
    )
    db.add(reading)
    await db.flush()
    await db.refresh(reading)

    # Run the full ML pipeline (errors are caught internally, never crash the API)
    await process_reading(db=db, reading=reading)

    return reading


@router.get("/{sensor_id}/history", response_model=List[SensorReadingResponse])
async def get_sensor_history(
    sensor_id: str,
    limit: int = Query(100, ge=1, le=1000),
    since: Optional[datetime] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """Return recent readings for a sensor."""
    q = (
        select(SensorReading)
        .where(SensorReading.sensor_id == sensor_id)
        .order_by(SensorReading.timestamp.desc())
        .limit(limit)
    )
    if since:
        q = q.where(SensorReading.timestamp >= since)
    result = await db.execute(q)
    return result.scalars().all()


# ---------------------------------------------------------------------------
# Health records
# ---------------------------------------------------------------------------
@router.get("/{sensor_id}/health", response_model=List[SensorHealthResponse])
async def get_sensor_health(
    sensor_id: str,
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    """Return recent health evaluations for a sensor."""
    q = (
        select(SensorHealth)
        .where(SensorHealth.sensor_id == sensor_id)
        .order_by(SensorHealth.timestamp.desc())
        .limit(limit)
    )
    result = await db.execute(q)
    return result.scalars().all()
