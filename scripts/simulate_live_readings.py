"""
scripts/simulate_live_readings.py
==================================
Simulates live sensor readings for all 4 machines by POSTing to
/api/sensors/readings every few seconds.

Usage:
    python scripts/simulate_live_readings.py
"""

import asyncio
import random
import httpx
from datetime import datetime, timezone

API_BASE = "http://localhost:8000"
INTERVAL = 3  # seconds between batches

# ---------------------------------------------------------------------------
# Machine + sensor profiles
# ---------------------------------------------------------------------------
MACHINES = {
    "M001": {
        "name": "Pump Alpha",
        "sensors": [
            {"id": "TEMP_01",      "sensor_type": "temperature", "min": 40.0,   "max": 90.0,   "unit": "C"},
            {"id": "PRESSURE_01",  "sensor_type": "pressure",    "min": 2.0,    "max": 8.0,    "unit": "bar"},
            {"id": "VIBRATION_01", "sensor_type": "vibration",   "min": 0.5,    "max": 15.0,   "unit": "mm/s"},
            {"id": "CURRENT_01",   "sensor_type": "current",     "min": 5.0,    "max": 25.0,   "unit": "A"},
            {"id": "FLOW_01",      "sensor_type": "flow",        "min": 50.0,   "max": 150.0,  "unit": "L/min"},
            {"id": "HUMIDITY_01",  "sensor_type": "humidity",    "min": 20.0,   "max": 70.0,   "unit": "%RH"},
        ],
    },
    "M002": {
        "name": "CNC Beta",
        "sensors": [
            {"id": "TEMP_02",      "sensor_type": "temperature", "min": 20.0,   "max": 90.0,   "unit": "C"},
            {"id": "VIBRATION_02", "sensor_type": "vibration",   "min": 0.5,    "max": 20.0,   "unit": "mm/s"},
            {"id": "CURRENT_02",   "sensor_type": "current",     "min": 5.0,    "max": 40.0,   "unit": "A"},
            {"id": "SPEED_02",     "sensor_type": "speed",       "min": 1000.0, "max": 8000.0, "unit": "RPM"},
            {"id": "VOLTAGE_02",   "sensor_type": "voltage",     "min": 210.0,  "max": 420.0,  "unit": "V"},
            {"id": "HUMIDITY_02",  "sensor_type": "humidity",    "min": 10.0,   "max": 60.0,   "unit": "%RH"},
        ],
    },
    "M003": {
        "name": "Conveyor Gamma",
        "sensors": [
            {"id": "SPEED_03",     "sensor_type": "speed",       "min": 0.5,    "max": 4.0,    "unit": "m/s"},
            {"id": "CURRENT_03",   "sensor_type": "current",     "min": 2.0,    "max": 18.0,   "unit": "A"},
            {"id": "VIBRATION_03", "sensor_type": "vibration",   "min": 0.2,    "max": 20.0,   "unit": "mm/s"},
            {"id": "TEMP_03",      "sensor_type": "temperature", "min": 15.0,   "max": 80.0,   "unit": "C"},
            {"id": "VOLTAGE_03",   "sensor_type": "voltage",     "min": 200.0,  "max": 240.0,  "unit": "V"},
        ],
    },
    "M004": {
        "name": "Compressor Delta",
        "sensors": [
            {"id": "PRESSURE_04",  "sensor_type": "pressure",    "min": 4.0,    "max": 12.0,   "unit": "bar"},
            {"id": "TEMP_04",      "sensor_type": "temperature", "min": 30.0,   "max": 150.0,  "unit": "C"},
            {"id": "VIBRATION_04", "sensor_type": "vibration",   "min": 1.0,    "max": 30.0,   "unit": "mm/s"},
            {"id": "CURRENT_04",   "sensor_type": "current",     "min": 5.0,    "max": 35.0,   "unit": "A"},
            {"id": "FLOW_04",      "sensor_type": "flow",        "min": 10.0,   "max": 80.0,   "unit": "CFM"},
            {"id": "HUMIDITY_04",  "sensor_type": "humidity",    "min": 15.0,   "max": 85.0,   "unit": "%RH"},
        ],
    },
}


def generate_value(sensor: dict, tick: int, machine_id: str) -> float:
    """Generate a stable sensor value with small random noise — no drift."""
    mid   = (sensor["min"] + sensor["max"]) / 2
    span  = (sensor["max"] - sensor["min"])
    # Small random walk — stays near midpoint, no trend
    noise = random.gauss(0, span * 0.015)
    value = mid + noise
    return round(max(sensor["min"], min(sensor["max"], value)), 4)


async def post_reading(
    client:     httpx.AsyncClient,
    sensor:     dict,
    machine_id: str,
    value:      float,
    timestamp:  str,
):
    payload = {
        "sensor_id":   sensor["id"],
        "machine_id":  machine_id,
        "sensor_type": sensor["sensor_type"],
        "timestamp":   timestamp,
        "value":       value,
        "unit":        sensor["unit"],
        "is_valid":    True,
    }
    try:
        resp = await client.post(
            f"{API_BASE}/api/sensors/readings",
            json=payload,
            timeout=5.0,
        )
        if resp.status_code in (200, 201):
            return True
        else:
            print(f"  ✗  {sensor['id']:15s} → HTTP {resp.status_code}: {resp.text[:60]}")
            return False
    except Exception as e:
        print(f"  ✗  {sensor['id']:15s} → {e}")
        return False


async def run():
    total_sensors = sum(len(m["sensors"]) for m in MACHINES.values())
    print(f"SensorShield Live Simulator")
    print(f"Machines : {len(MACHINES)} ({', '.join(MACHINES.keys())})")
    print(f"Sensors  : {total_sensors} total")
    print(f"Interval : {INTERVAL}s between batches")
    print(f"API      : {API_BASE}")
    print(f"Press Ctrl+C to stop.\n")

    tick = 0
    async with httpx.AsyncClient() as client:
        while True:
            shared_ts = datetime.now(timezone.utc).isoformat()
            ok_count  = 0
            tasks     = []

            for machine_id, machine in MACHINES.items():
                for sensor in machine["sensors"]:
                    value = generate_value(sensor, tick, machine_id)
                    tasks.append(post_reading(client, sensor, machine_id, value, shared_ts))

            results  = await asyncio.gather(*tasks)
            ok_count = sum(1 for r in results if r)

            print(f"[{datetime.now().strftime('%H:%M:%S')}] Batch {tick + 1:4d} — {ok_count}/{total_sensors} readings posted")
            tick += 1
            await asyncio.sleep(INTERVAL)


if __name__ == "__main__":
    asyncio.run(run())
