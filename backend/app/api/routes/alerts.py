"""
api/routes/alerts.py
====================
Endpoints for querying and acknowledging alerts.
Alert generation happens in the Alert Engine (Phase 9).
These endpoints are read-only except for status updates.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.alert import Alert, AlertStatus
from app.schemas.alert import AlertResponse

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.get("", response_model=List[AlertResponse])
async def list_alerts(
    status: Optional[AlertStatus] = Query(None),
    machine_id: Optional[str] = Query(None),
    sensor_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    """
    Return alerts with optional filters.
    Default: active alerts only, most recent first.
    """
    q = select(Alert).order_by(Alert.timestamp.desc()).limit(limit)

    if status:
        q = q.where(Alert.status == status)
    else:
        q = q.where(Alert.status == AlertStatus.ACTIVE)

    if machine_id:
        q = q.where(Alert.machine_id == machine_id)
    if sensor_id:
        q = q.where(Alert.sensor_id == sensor_id)

    result = await db.execute(q)
    return result.scalars().all()


@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert(alert_id: int, db: AsyncSession = Depends(get_db)):
    alert = await db.get(Alert, alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    return alert


@router.patch("/{alert_id}/acknowledge", response_model=AlertResponse)
async def acknowledge_alert(alert_id: int, db: AsyncSession = Depends(get_db)):
    """Mark an alert as acknowledged."""
    alert = await db.get(Alert, alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    alert.status = AlertStatus.ACKNOWLEDGED
    await db.flush()
    await db.refresh(alert)
    return alert
