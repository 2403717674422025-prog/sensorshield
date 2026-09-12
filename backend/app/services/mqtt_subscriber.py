"""
services/mqtt_subscriber.py
============================
MQTT subscriber — Phase 8.

Connects to the MQTT broker using the existing config, subscribes to
sensor topics, and feeds every message through the SAME processing
service used by the REST endpoint (process_reading).

Topic convention:
  {MQTT_TOPIC_PREFIX}/readings/{machine_id}/{sensor_id}

Payload JSON:
  {
    "timestamp": "2026-09-07T10:00:00Z",  // ISO-8601
    "value": 72.4,
    "unit": "C",
    "sensor_type": "temperature"
  }

Architecture:
  REST  ──┐
           ├──► process_reading() ──► ML ──► Health ──► Alert ──► WS
  MQTT  ──┘

Start from main.py lifespan with: asyncio.create_task(start_mqtt_subscriber())
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional

import paho.mqtt.client as mqtt

from app.core.config import settings
from app.core.database import get_db_context
from app.models.reading import SensorReading
from app.models.sensor import Sensor

logger = logging.getLogger(__name__)

_mqtt_client: Optional[mqtt.Client] = None


def _on_connect(client: mqtt.Client, userdata, flags, rc, properties=None):
    if rc == 0:
        topic = f"{settings.MQTT_TOPIC_PREFIX}/readings/#"
        client.subscribe(topic, qos=1)
        logger.info("MQTT connected — subscribed to %s", topic)
    else:
        logger.error("MQTT connection failed with code %d", rc)


def _on_message(client: mqtt.Client, userdata, msg: mqtt.MQTTMessage):
    """
    Called on the paho network thread.  We schedule the async processing
    on the main event loop so it can use AsyncSession safely.
    """
    loop: asyncio.AbstractEventLoop = userdata["loop"]
    asyncio.run_coroutine_threadsafe(_handle_message(msg), loop)


async def _handle_message(msg: mqtt.MQTTMessage) -> None:
    """Parse MQTT message and push it through the sensor processing pipeline."""
    try:
        # Topic: sensorshield/readings/{machine_id}/{sensor_id}
        parts = msg.topic.split("/")
        if len(parts) < 4:
            logger.warning("Unexpected MQTT topic format: %s", msg.topic)
            return

        machine_id = parts[-2]
        sensor_id  = parts[-1]

        payload = json.loads(msg.payload.decode("utf-8"))

        ts_raw  = payload.get("timestamp")
        value   = float(payload["value"])
        unit    = payload.get("unit", "unknown")

        timestamp = (
            datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            if ts_raw
            else datetime.now(timezone.utc)
        )

    except Exception as exc:
        logger.warning("MQTT message parse error on topic %s: %s", msg.topic, exc)
        return

    # Push through the same pipeline as REST
    try:
        from app.services.sensor_pipeline import process_reading  # local import avoids circular

        async with get_db_context() as db:
            # Validate sensor exists
            sensor = await db.get(Sensor, sensor_id)
            if not sensor:
                logger.debug("MQTT: sensor %s not registered — skipping.", sensor_id)
                return

            reading = SensorReading(
                sensor_id  = sensor_id,
                machine_id = machine_id,
                timestamp  = timestamp,
                value      = value,
                is_valid   = True,
            )
            db.add(reading)
            await db.flush()
            await db.refresh(reading)

            await process_reading(db=db, reading=reading)

    except Exception as exc:
        logger.error("MQTT pipeline error for %s/%s: %s", machine_id, sensor_id, exc)


async def start_mqtt_subscriber() -> None:
    """
    Connect to the MQTT broker and start the background network loop.
    Runs indefinitely in an asyncio task — reconnects on failure.
    """
    global _mqtt_client
    loop = asyncio.get_running_loop()

    client = mqtt.Client(
        client_id      = "sensorshield-backend",
        protocol       = mqtt.MQTTv5,
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
    )
    client.on_connect = _on_connect
    client.on_message = _on_message
    client.user_data_set({"loop": loop})

    try:
        client.connect(settings.MQTT_HOST, settings.MQTT_PORT, keepalive=60)
    except Exception as exc:
        logger.warning("MQTT broker not reachable at %s:%d — %s", settings.MQTT_HOST, settings.MQTT_PORT, exc)
        return

    client.loop_start()
    _mqtt_client = client
    logger.info("MQTT subscriber started (broker=%s:%d)", settings.MQTT_HOST, settings.MQTT_PORT)


def stop_mqtt_subscriber() -> None:
    global _mqtt_client
    if _mqtt_client:
        _mqtt_client.loop_stop()
        _mqtt_client.disconnect()
        _mqtt_client = None
        logger.info("MQTT subscriber stopped.")
