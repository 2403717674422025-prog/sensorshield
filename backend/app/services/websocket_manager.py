"""
services/websocket_manager.py
==============================
WebSocket connection manager.

Supports multiple simultaneous frontend clients.
Broadcasts health events and alerts independently of the ML layer.
Client disconnects are handled safely — a broken connection never
crashes the broadcast loop.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages a pool of active WebSocket connections.

    Usage
    -----
    manager = ConnectionManager()          # singleton created below

    # In a WebSocket route:
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()   # keep alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)

    # From any service / route:
    await manager.broadcast({"type": "health_update", "data": {...}})
    """

    def __init__(self) -> None:
        self._active: List[WebSocket] = []

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------
    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._active.append(websocket)
        logger.info(
            "WebSocket client connected. Active connections: %d",
            len(self._active),
        )

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self._active:
            self._active.remove(websocket)
        logger.info(
            "WebSocket client disconnected. Active connections: %d",
            len(self._active),
        )

    # ------------------------------------------------------------------
    # Broadcasting
    # ------------------------------------------------------------------
    async def broadcast(self, message: Dict[str, Any]) -> None:
        """
        Send a JSON message to every connected client.
        Stale/broken connections are removed silently.
        """
        dead: List[WebSocket] = []
        for ws in list(self._active):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)

        for ws in dead:
            self.disconnect(ws)

        if dead:
            logger.debug("Removed %d dead WebSocket connections.", len(dead))

    async def broadcast_health_update(self, health_dict: Dict[str, Any]) -> None:
        """Convenience wrapper — wraps a health evaluation in a typed envelope."""
        await self.broadcast({"type": "health_update", "data": health_dict})

    async def broadcast_alert(self, alert_dict: Dict[str, Any]) -> None:
        """Convenience wrapper — wraps an alert in a typed envelope."""
        await self.broadcast({"type": "alert", "data": alert_dict})

    async def broadcast_prediction(self, prediction_dict: Dict[str, Any]) -> None:
        """Convenience wrapper — wraps a prediction update."""
        await self.broadcast({"type": "prediction_update", "data": prediction_dict})

    @property
    def connection_count(self) -> int:
        return len(self._active)


# ---------------------------------------------------------------------------
# Module-level singleton — imported by routes and services
# ---------------------------------------------------------------------------
ws_manager = ConnectionManager()
