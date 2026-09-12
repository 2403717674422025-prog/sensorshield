"""
scripts/seed_database.py
========================
Seeds the database with 4 industrial machines and their sensors.

Machines:
  M001 — Industrial Pump Unit Alpha        (water pump, 6 sensors)
  M002 — CNC Milling Machine Beta          (CNC machine, 6 sensors)
  M003 — Conveyor Belt System Gamma        (conveyor, 5 sensors)
  M004 — Air Compressor Unit Delta         (compressor, 6 sensors)

Usage (from repo root):
    python scripts/seed_database.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from app.core.config import settings
from app.models.machine import Machine, MachineStatus
from app.models.sensor import Sensor, SensorType, SensorStatus

# ---------------------------------------------------------------------------
# Machine definitions
# ---------------------------------------------------------------------------
MACHINES = [
    {
        "id":          "M001",
        "name":        "Industrial Pump Unit Alpha",
        "location":    "Plant Floor A — Bay 3",
        "description": "Primary water pump. Feeds the main cooling circuit.",
        "status":      MachineStatus.ONLINE,
    },
    {
        "id":          "M002",
        "name":        "CNC Milling Machine Beta",
        "location":    "Machining Floor B — Bay 1",
        "description": "High-precision CNC mill for metal components.",
        "status":      MachineStatus.ONLINE,
    },
    {
        "id":          "M003",
        "name":        "Conveyor Belt System Gamma",
        "location":    "Assembly Line C — Section 2",
        "description": "Main conveyor belt for parts transport between stations.",
        "status":      MachineStatus.ONLINE,
    },
    {
        "id":          "M004",
        "name":        "Air Compressor Unit Delta",
        "location":    "Utilities Room D",
        "description": "Central air compressor supplying pneumatic tools.",
        "status":      MachineStatus.ONLINE,
    },
]

# ---------------------------------------------------------------------------
# Sensor definitions per machine
# ---------------------------------------------------------------------------
SENSORS = [
    # ── M001: Industrial Pump ──────────────────────────────────────────
    {"id": "TEMP_01",      "machine_id": "M001", "sensor_type": SensorType.TEMPERATURE, "name": "Pump Body Temperature",   "unit": "C",    "min_valid_value": -10.0, "max_valid_value": 150.0, "status": SensorStatus.HEALTHY},
    {"id": "PRESSURE_01",  "machine_id": "M001", "sensor_type": SensorType.PRESSURE,    "name": "Inlet Pressure",          "unit": "bar",  "min_valid_value": 0.0,   "max_valid_value": 10.0,  "status": SensorStatus.HEALTHY},
    {"id": "VIBRATION_01", "machine_id": "M001", "sensor_type": SensorType.VIBRATION,   "name": "Bearing Vibration",       "unit": "mm/s", "min_valid_value": 0.0,   "max_valid_value": 50.0,  "status": SensorStatus.HEALTHY},
    {"id": "CURRENT_01",   "machine_id": "M001", "sensor_type": SensorType.CURRENT,     "name": "Motor Current Draw",      "unit": "A",    "min_valid_value": 0.0,   "max_valid_value": 30.0,  "status": SensorStatus.HEALTHY},
    {"id": "FLOW_01",      "machine_id": "M001", "sensor_type": SensorType.FLOW,        "name": "Flow Rate",               "unit": "L/min","min_valid_value": 0.0,   "max_valid_value": 200.0, "status": SensorStatus.HEALTHY},
    {"id": "HUMIDITY_01",  "machine_id": "M001", "sensor_type": SensorType.HUMIDITY,    "name": "Enclosure Humidity",      "unit": "%RH",  "min_valid_value": 0.0,   "max_valid_value": 100.0, "status": SensorStatus.HEALTHY},

    # ── M002: CNC Milling Machine ──────────────────────────────────────
    {"id": "TEMP_02",      "machine_id": "M002", "sensor_type": SensorType.TEMPERATURE, "name": "Spindle Temperature",     "unit": "C",    "min_valid_value": 10.0,  "max_valid_value": 120.0, "status": SensorStatus.HEALTHY},
    {"id": "VIBRATION_02", "machine_id": "M002", "sensor_type": SensorType.VIBRATION,   "name": "Spindle Vibration",       "unit": "mm/s", "min_valid_value": 0.0,   "max_valid_value": 30.0,  "status": SensorStatus.HEALTHY},
    {"id": "CURRENT_02",   "machine_id": "M002", "sensor_type": SensorType.CURRENT,     "name": "Spindle Motor Current",   "unit": "A",    "min_valid_value": 0.0,   "max_valid_value": 50.0,  "status": SensorStatus.HEALTHY},
    {"id": "SPEED_02",     "machine_id": "M002", "sensor_type": SensorType.SPEED,       "name": "Spindle Speed",           "unit": "RPM",  "min_valid_value": 0.0,   "max_valid_value": 10000.0,"status": SensorStatus.HEALTHY},
    {"id": "VOLTAGE_02",   "machine_id": "M002", "sensor_type": SensorType.VOLTAGE,     "name": "Drive Voltage",           "unit": "V",    "min_valid_value": 200.0, "max_valid_value": 480.0, "status": SensorStatus.HEALTHY},
    {"id": "HUMIDITY_02",  "machine_id": "M002", "sensor_type": SensorType.HUMIDITY,    "name": "Cabinet Humidity",        "unit": "%RH",  "min_valid_value": 0.0,   "max_valid_value": 80.0,  "status": SensorStatus.HEALTHY},

    # ── M003: Conveyor Belt ────────────────────────────────────────────
    {"id": "SPEED_03",     "machine_id": "M003", "sensor_type": SensorType.SPEED,       "name": "Belt Speed",              "unit": "m/s",  "min_valid_value": 0.0,   "max_valid_value": 5.0,   "status": SensorStatus.HEALTHY},
    {"id": "CURRENT_03",   "machine_id": "M003", "sensor_type": SensorType.CURRENT,     "name": "Drive Motor Current",     "unit": "A",    "min_valid_value": 0.0,   "max_valid_value": 20.0,  "status": SensorStatus.HEALTHY},
    {"id": "VIBRATION_03", "machine_id": "M003", "sensor_type": SensorType.VIBRATION,   "name": "Roller Vibration",        "unit": "mm/s", "min_valid_value": 0.0,   "max_valid_value": 25.0,  "status": SensorStatus.HEALTHY},
    {"id": "TEMP_03",      "machine_id": "M003", "sensor_type": SensorType.TEMPERATURE, "name": "Motor Temperature",       "unit": "C",    "min_valid_value": 5.0,   "max_valid_value": 100.0, "status": SensorStatus.HEALTHY},
    {"id": "VOLTAGE_03",   "machine_id": "M003", "sensor_type": SensorType.VOLTAGE,     "name": "Drive Voltage",           "unit": "V",    "min_valid_value": 190.0, "max_valid_value": 250.0, "status": SensorStatus.HEALTHY},

    # ── M004: Air Compressor ───────────────────────────────────────────
    {"id": "PRESSURE_04",  "machine_id": "M004", "sensor_type": SensorType.PRESSURE,    "name": "Tank Pressure",           "unit": "bar",  "min_valid_value": 0.0,   "max_valid_value": 15.0,  "status": SensorStatus.HEALTHY},
    {"id": "TEMP_04",      "machine_id": "M004", "sensor_type": SensorType.TEMPERATURE, "name": "Compressor Temperature",  "unit": "C",    "min_valid_value": 5.0,   "max_valid_value": 180.0, "status": SensorStatus.HEALTHY},
    {"id": "VIBRATION_04", "machine_id": "M004", "sensor_type": SensorType.VIBRATION,   "name": "Compressor Vibration",    "unit": "mm/s", "min_valid_value": 0.0,   "max_valid_value": 40.0,  "status": SensorStatus.HEALTHY},
    {"id": "CURRENT_04",   "machine_id": "M004", "sensor_type": SensorType.CURRENT,     "name": "Motor Current",           "unit": "A",    "min_valid_value": 0.0,   "max_valid_value": 40.0,  "status": SensorStatus.HEALTHY},
    {"id": "FLOW_04",      "machine_id": "M004", "sensor_type": SensorType.FLOW,        "name": "Air Flow Output",         "unit": "CFM",  "min_valid_value": 0.0,   "max_valid_value": 100.0, "status": SensorStatus.HEALTHY},
    {"id": "HUMIDITY_04",  "machine_id": "M004", "sensor_type": SensorType.HUMIDITY,    "name": "Intake Air Humidity",     "unit": "%RH",  "min_valid_value": 0.0,   "max_valid_value": 95.0,  "status": SensorStatus.HEALTHY},
]


async def seed():
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

    async with AsyncSessionLocal() as session:
        seeded_machines = 0
        seeded_sensors  = 0

        for machine_data in MACHINES:
            existing = await session.get(Machine, machine_data["id"])
            if existing:
                print(f"[SKIP] Machine {machine_data['id']} already exists.")
                continue
            machine = Machine(**machine_data)
            session.add(machine)
            print(f"[ADD]  Machine: {machine_data['id']} — {machine_data['name']}")
            seeded_machines += 1

        for sensor_data in SENSORS:
            existing = await session.get(Sensor, sensor_data["id"])
            if existing:
                print(f"[SKIP] Sensor {sensor_data['id']} already exists.")
                continue
            sensor = Sensor(**sensor_data)
            session.add(sensor)
            print(f"[ADD]  Sensor:  {sensor_data['id']} ({sensor_data['sensor_type'].value}) — {sensor_data['name']}")
            seeded_sensors += 1

        await session.commit()

        print(f"\n[OK] Seed complete.")
        print(f"     Machines added : {seeded_machines}")
        print(f"     Sensors added  : {seeded_sensors}")
        print(f"\n     Total machines : {len(MACHINES)}")
        print(f"     Total sensors  : {len(SENSORS)}")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(seed())
