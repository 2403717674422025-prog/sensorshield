"""
api/routes/predictions.py
=========================
Endpoints for querying downstream ML predictions.
Prediction generation happens in the Predictive Model + Trust Engine (Phase 9).
These endpoints are read-only — predictions are never created via API directly.
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.prediction import Prediction
from app.schemas.prediction import PredictionResponse

router = APIRouter(prefix="/api/predictions", tags=["predictions"])


@router.get("/{machine_id}", response_model=List[PredictionResponse])
async def get_predictions(
    machine_id: str,
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    """Return recent predictions for a machine, newest first."""
    q = (
        select(Prediction)
        .where(Prediction.machine_id == machine_id)
        .order_by(Prediction.timestamp.desc())
        .limit(limit)
    )
    result = await db.execute(q)
    predictions = result.scalars().all()
    if not predictions:
        raise HTTPException(
            status_code=404,
            detail=f"No predictions found for machine '{machine_id}'",
        )
    return predictions


@router.get("/{machine_id}/latest", response_model=PredictionResponse)
async def get_latest_prediction(machine_id: str, db: AsyncSession = Depends(get_db)):
    """Return only the most recent prediction for a machine."""
    q = (
        select(Prediction)
        .where(Prediction.machine_id == machine_id)
        .order_by(Prediction.timestamp.desc())
        .limit(1)
    )
    result = await db.execute(q)
    prediction = result.scalar_one_or_none()
    if not prediction:
        raise HTTPException(
            status_code=404,
            detail=f"No predictions found for machine '{machine_id}'",
        )
    return prediction
