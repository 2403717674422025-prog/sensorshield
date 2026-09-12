"""
tests/integration/test_sensor_ingestion.py
===========================================
Integration tests for the sensor ingestion pipeline.

These tests use the in-memory SQLite test database and the ASGI test client.
The ML pipeline is tested for graceful degradation — it may or may not have
models loaded, but the API must always return 201 with a valid response.
"""

import pytest
from datetime import datetime, timezone


MACHINE_PAYLOAD = {"id": "M001", "name": "Test Pump", "location": "Bay 1"}
SENSOR_PAYLOAD  = {
    "id":          "TEMP_01",
    "machine_id":  "M001",
    "sensor_type": "temperature",
    "unit":        "C",
    "name":        "Body Temperature",
    "min_valid_value": 0.0,
    "max_valid_value": 150.0,
}


async def _seed(client):
    """Create machine + sensor as preconditions."""
    await client.post("/api/machines", json=MACHINE_PAYLOAD)
    await client.post("/api/sensors",  json=SENSOR_PAYLOAD)


@pytest.mark.asyncio
async def test_reading_ingestion_returns_201(client):
    """A valid reading must be accepted and persisted."""
    await _seed(client)
    payload = {
        "sensor_id":   "TEMP_01",
        "machine_id":  "M001",
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "value":       72.4,
        "unit":        "C",
        "sensor_type": "temperature",
    }
    r = await client.post("/api/sensors/readings", json=payload)
    assert r.status_code == 201
    data = r.json()
    assert data["sensor_id"] == "TEMP_01"
    assert data["value"]     == pytest.approx(72.4)
    assert data["is_valid"]  is True


@pytest.mark.asyncio
async def test_reading_for_unknown_sensor_returns_404(client):
    """Posting a reading for a sensor that doesn't exist must return 404."""
    payload = {
        "sensor_id":   "GHOST_SENSOR",
        "machine_id":  "M001",
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "value":       10.0,
        "unit":        "C",
        "sensor_type": "temperature",
    }
    r = await client.post("/api/sensors/readings", json=payload)
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_multiple_readings_stored(client):
    """Multiple sequential readings should all be stored."""
    await _seed(client)
    for v in [70.0, 71.5, 73.2, 68.9]:
        payload = {
            "sensor_id":   "TEMP_01",
            "machine_id":  "M001",
            "timestamp":   datetime.now(timezone.utc).isoformat(),
            "value":       v,
            "unit":        "C",
            "sensor_type": "temperature",
        }
        r = await client.post("/api/sensors/readings", json=payload)
        assert r.status_code == 201

    history = await client.get("/api/sensors/TEMP_01/history?limit=10")
    assert history.status_code == 200
    assert len(history.json()) == 4


@pytest.mark.asyncio
async def test_sensor_history_returns_readings(client):
    """GET /sensors/{id}/history must return stored readings."""
    await _seed(client)
    payload = {
        "sensor_id":   "TEMP_01",
        "machine_id":  "M001",
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "value":       55.0,
        "unit":        "C",
        "sensor_type": "temperature",
    }
    await client.post("/api/sensors/readings", json=payload)

    r = await client.get("/api/sensors/TEMP_01/history?limit=5")
    assert r.status_code == 200
    readings = r.json()
    assert len(readings) >= 1
    assert readings[0]["sensor_id"] == "TEMP_01"


@pytest.mark.asyncio
async def test_alerts_endpoint_accessible(client):
    """GET /api/alerts should return a list (possibly empty)."""
    r = await client.get("/api/alerts")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


@pytest.mark.asyncio
async def test_acknowledge_nonexistent_alert_returns_404(client):
    r = await client.patch("/api/alerts/99999/acknowledge")
    assert r.status_code == 404
