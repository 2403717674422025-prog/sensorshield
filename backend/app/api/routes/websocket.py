"""
api/routes/websocket.py
========================
WebSocket endpoint for real-time sensor health and alert streaming.

Clients connect to /ws/sensors and receive JSON messages of two types:
  { "type": "health_update", "data": { ... } }
  { "type": "alert",         "data": { ... } }
  { "type": "prediction_update", "data": { ... } }

The client should send any text (e.g. "ping") to keep the connection alive.
"""

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.websocket_manager import ws_manager

router = APIRouter(tags=["websocket"])
logger = logging.getLogger(__name__)


@router.websocket("/ws/sensors")
async def websocket_sensor_stream(websocket: WebSocket):
    """
    Real-time sensor health and alert stream.
    Connect with any WebSocket client at ws://host:8000/ws/sensors
    """
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep the connection alive; client can send "ping" or any text
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as exc:
        logger.debug("WebSocket connection closed with error: %s", exc)
        ws_manager.disconnect(websocket)
