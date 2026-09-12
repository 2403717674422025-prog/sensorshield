"""
api/dependencies.py
===================
Shared FastAPI dependencies used across multiple routes.
Keeping them in one place avoids circular imports and makes
testing easy (you can override any dependency in tests).
"""

from fastapi import Depends, Header, HTTPException, status

from app.core.config import Settings, get_settings


async def verify_internal_key(
    x_internal_key: str = Header(None),
    settings: Settings = Depends(get_settings),
):
    """
    Simple API key check for internal/admin endpoints.
    This is NOT a substitute for proper auth — it's a lightweight guard
    for the Phase 1 prototype. Full JWT auth will be added in Phase 10.

    To use:
        @router.post("/admin/...", dependencies=[Depends(verify_internal_key)])
    """
    if settings.APP_ENV == "production":
        if not x_internal_key or x_internal_key != settings.SECRET_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid internal key",
            )
