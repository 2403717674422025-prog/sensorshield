"""
api/routes/machines.py
======================
CRUD endpoints for machines.
Phase 1: create and list only.
Full machine detail (with sensors, latest prediction, health) will be
added when the ML pipeline is connected in Phase 10.
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.machine import Machine
from app.schemas.sensor import MachineCreate, MachineResponse

router = APIRouter(prefix="/api/machines", tags=["machines"])


@router.get("", response_model=List[MachineResponse])
async def list_machines(db: AsyncSession = Depends(get_db)):
    """Return all registered machines."""
    result = await db.execute(select(Machine).order_by(Machine.id))
    return result.scalars().all()


@router.get("/{machine_id}", response_model=MachineResponse)
async def get_machine(machine_id: str, db: AsyncSession = Depends(get_db)):
    """Return a single machine by ID."""
    machine = await db.get(Machine, machine_id)
    if not machine:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Machine '{machine_id}' not found",
        )
    return machine


@router.post("", response_model=MachineResponse, status_code=status.HTTP_201_CREATED)
async def create_machine(payload: MachineCreate, db: AsyncSession = Depends(get_db)):
    """Register a new machine."""
    existing = await db.get(Machine, payload.id)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Machine '{payload.id}' already exists",
        )
    machine = Machine(**payload.model_dump())
    db.add(machine)
    await db.flush()
    await db.refresh(machine)
    return machine
