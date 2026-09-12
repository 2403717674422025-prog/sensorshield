"""
services/alert_service.py
==========================
Alert generation service (Phase 5).

Receives a SensorHealthEvaluation, applies threshold rules, creates Alert
records in the database, and resolves stale alerts when the sensor recovers.

Default thresholds (override via settings if needed):
  health_score < 30  -> CRITICAL
  health_score < 50  -> WARNING
  health_score >= 70 -> sensor considered recovered, resolve open alerts
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.alert import Alert, AlertSeverity, AlertStatus, FaultType
from app.models.sensor import SensorStatus
from app.ml.reliability.engine import SensorHealthEvaluation
from app.services.email_service import send_alert_email

logger = logging.getLogger(__name__)

# Thresholds
CRITICAL_THRESHOLD = 30.0
WARNING_THRESHOLD  = 50.0
RECOVERY_THRESHOLD = 70.0


def _fault_type_from_status(status: SensorStatus) -> FaultType:
    """Map a SensorStatus to the most appropriate FaultType for the alert."""
    mapping = {
        SensorStatus.DRIFTING:     FaultType.DRIFT,
        SensorStatus.NOISY:        FaultType.NOISE,
        SensorStatus.STUCK:        FaultType.STUCK,
        SensorStatus.MISSING:      FaultType.MISSING,
        SensorStatus.INTERMITTENT: FaultType.INTERMITTENT,
        SensorStatus.FAILED:       FaultType.UNKNOWN,
    }
    return mapping.get(status, FaultType.LOW_RELIABILITY)


def _severity_from_score(health_score: float) -> Optional[AlertSeverity]:
    """Return alert severity based on health score, or None if healthy."""
    if health_score < CRITICAL_THRESHOLD:
        return AlertSeverity.CRITICAL
    if health_score < WARNING_THRESHOLD:
        return AlertSeverity.WARNING
    return None


def _build_impact(status: SensorStatus, sensor_id: str) -> str:
    impacts = {
        SensorStatus.DRIFTING:     f"Downstream ML predictions using {sensor_id} may be systematically biased.",
        SensorStatus.NOISY:        f"High noise on {sensor_id} increases model uncertainty and reduces prediction confidence.",
        SensorStatus.STUCK:        f"{sensor_id} is frozen — all readings are identical. ML models will fail silently.",
        SensorStatus.MISSING:      f"{sensor_id} data is unavailable. Downstream predictions are operating blind on this channel.",
        SensorStatus.INTERMITTENT: f"Intermittent dropouts on {sensor_id} cause inconsistent model inputs.",
        SensorStatus.FAILED:       f"{sensor_id} has failed. Remove it from active ML features immediately.",
    }
    return impacts.get(status, f"Sensor {sensor_id} reliability is degraded. Prediction trust is reduced.")


def _build_action(status: SensorStatus) -> str:
    actions = {
        SensorStatus.DRIFTING:     "Inspect sensor calibration. Recalibrate or replace if drift persists beyond 2 cycles.",
        SensorStatus.NOISY:        "Check sensor wiring and grounding. Verify no nearby EMI sources. Replace sensor if noise floor is too high.",
        SensorStatus.STUCK:        "Sensor output is frozen. Check power supply, connector, and ADC. Replace sensor if unresponsive.",
        SensorStatus.MISSING:      "Sensor not reporting. Verify network/MQTT connection, power, and sensor firmware.",
        SensorStatus.INTERMITTENT: "Inspect cable and connector for intermittent contact. Check for packet loss on the data bus.",
        SensorStatus.FAILED:       "Sensor has failed. Take offline immediately and schedule replacement.",
    }
    return actions.get(status, "Inspect sensor hardware and review recent maintenance logs.")


async def evaluate_and_create_alert(
    db: AsyncSession,
    evaluation: SensorHealthEvaluation,
    machine_id: str,
    trust_score: Optional[float] = None,
) -> Optional[Alert]:
    """
    Core alert logic:
    1. If health is degraded → check for an existing ACTIVE alert for this sensor.
    2. If no active alert exists → create a new one (avoid duplicate spam).
    3. If health has recovered → resolve any open alerts.

    Returns the newly created Alert, or None if no action was taken.
    """
    sensor_id    = evaluation.sensor_id
    health_score = evaluation.health_score
    status       = evaluation.status

    # ------------------------------------------------------------------ #
    # 1. Recovery: resolve open alerts when sensor is healthy again
    # ------------------------------------------------------------------ #
    if health_score >= RECOVERY_THRESHOLD and status == SensorStatus.HEALTHY:
        resolved = await _resolve_open_alerts(db, sensor_id)
        if resolved:
            logger.info("Sensor %s recovered (score=%.1f) — resolved %d alert(s).", sensor_id, health_score, resolved)
        return None

    # ------------------------------------------------------------------ #
    # 2. Determine severity
    # ------------------------------------------------------------------ #
    severity = _severity_from_score(health_score)
    if severity is None:
        # Between 50–70: degrading but not alert-worthy yet
        return None

    # ------------------------------------------------------------------ #
    # 3. De-duplicate: skip if an identical active alert already exists
    # ------------------------------------------------------------------ #
    existing = await _get_active_alert(db, sensor_id)
    if existing is not None:
        # Upgrade severity if the situation worsened
        if severity == AlertSeverity.CRITICAL and existing.severity == AlertSeverity.WARNING:
            existing.severity = AlertSeverity.CRITICAL
            existing.description = evaluation.reason
            existing.health_score_at_alert = health_score
            logger.info("Upgraded alert %d for sensor %s to CRITICAL.", existing.id, sensor_id)
        return None  # Alert already open, no duplicate

    # ------------------------------------------------------------------ #
    # 4. Create new alert
    # ------------------------------------------------------------------ #
    fault_type = _fault_type_from_status(status)
    alert = Alert(
        sensor_id              = sensor_id,
        machine_id             = machine_id,
        timestamp              = datetime.now(timezone.utc),
        severity               = severity,
        fault_type             = fault_type,
        status                 = AlertStatus.ACTIVE,
        description            = evaluation.reason,
        impact                 = _build_impact(status, sensor_id),
        recommended_action     = _build_action(status),
        health_score_at_alert  = health_score,
        trust_score_at_alert   = trust_score,
    )
    db.add(alert)
    await db.flush()
    await db.refresh(alert)

    logger.warning(
        "ALERT created: sensor=%s severity=%s fault=%s score=%.1f",
        sensor_id, severity.value, fault_type.value, health_score,
    )

    # Send email notification for WARNING and CRITICAL alerts
    import asyncio
    asyncio.create_task(send_alert_email(
        sensor_id   = sensor_id,
        machine_id  = machine_id,
        severity    = severity.value,
        fault_type  = fault_type.value,
        health_score= health_score,
        description = evaluation.reason or "",
        impact      = _build_impact(status, sensor_id),
        action      = _build_action(status),
        timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
    ))

    return alert


async def _get_active_alert(db: AsyncSession, sensor_id: str) -> Optional[Alert]:
    """Return the most recent ACTIVE alert for a sensor, if any."""
    q = (
        select(Alert)
        .where(Alert.sensor_id == sensor_id)
        .where(Alert.status == AlertStatus.ACTIVE)
        .order_by(Alert.timestamp.desc())
        .limit(1)
    )
    result = await db.execute(q)
    return result.scalar_one_or_none()


async def _resolve_open_alerts(db: AsyncSession, sensor_id: str) -> int:
    """Resolve all ACTIVE/ACKNOWLEDGED alerts for a sensor. Returns count resolved."""
    q = (
        select(Alert)
        .where(Alert.sensor_id == sensor_id)
        .where(Alert.status.in_([AlertStatus.ACTIVE, AlertStatus.ACKNOWLEDGED]))
    )
    result = await db.execute(q)
    alerts = result.scalars().all()
    now = datetime.now(timezone.utc)
    for alert in alerts:
        alert.status      = AlertStatus.RESOLVED
        alert.resolved_at = now
    return len(alerts)
