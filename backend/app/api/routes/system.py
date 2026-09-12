"""
api/routes/system.py
====================
System-level endpoints used for health checks, monitoring, and readiness probes.
These run before any ML logic is loaded, so they must be fast and never fail
due to a missing model file.
"""

import platform
import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.database import check_database_connection, get_db

router = APIRouter(prefix="/api/system", tags=["system"])

# Track when the process started so we can report uptime
_START_TIME = time.time()


@router.get("/health")
async def health_check(
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    """
    Health check endpoint.
    Returns 200 if the API is reachable and the database is connected.
    Used by Docker health checks, load balancers, and the dashboard status widget.
    """
    db_ok = await check_database_connection()
    uptime_seconds = int(time.time() - _START_TIME)

    return {
        "status": "healthy" if db_ok else "degraded",
        "database": "connected" if db_ok else "disconnected",
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENV,
        "uptime_seconds": uptime_seconds,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
    }


@router.get("/ready")
async def readiness_check():
    """
    Kubernetes/Docker readiness probe.
    Returns 200 once the application has started.
    (We keep this separate from /health so liveness and readiness can diverge.)
    """
    return {"ready": True, "timestamp": datetime.now(timezone.utc).isoformat()}
