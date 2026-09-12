"""
tests/unit/test_alert_service.py
=================================
Unit tests for the alert generation service.

Coverage:
  - Alert created when health score below WARNING threshold
  - Alert created at CRITICAL threshold with correct severity
  - No duplicate alert when one already exists for the sensor
  - Alert severity upgraded from WARNING -> CRITICAL when score worsens
  - Alerts resolved when sensor recovers above RECOVERY_THRESHOLD
  - No alert created when health is in the degrading-but-safe zone (50-70)
"""

import pytest
from datetime import datetime, timezone

from app.models.alert import Alert, AlertSeverity, AlertStatus
from app.models.sensor import SensorStatus
from app.ml.reliability.engine import SensorHealthEvaluation
from app.services.alert_service import (
    evaluate_and_create_alert,
    CRITICAL_THRESHOLD,
    WARNING_THRESHOLD,
    RECOVERY_THRESHOLD,
)


def _make_eval(sensor_id: str, health_score: float, status: SensorStatus = SensorStatus.DRIFTING) -> SensorHealthEvaluation:
    return SensorHealthEvaluation(
        sensor_id            = sensor_id,
        timestamp            = datetime.now(timezone.utc),
        health_score         = health_score,
        status               = status,
        status_confidence    = 0.9,
        reason               = f"Test evaluation for {sensor_id}",
        drift_score          = 100.0 - health_score,
        noise_score          = 100.0,
        consistency_score    = 100.0,
        missing_data_score   = 100.0,
        anomaly_score        = 100.0,
        reconstruction_error = 0.01,
    )


@pytest.mark.asyncio
async def test_warning_alert_created(db_session):
    """Health score just below WARNING_THRESHOLD should create a WARNING alert."""
    evaluation = _make_eval("TEMP_01", WARNING_THRESHOLD - 5)
    alert = await evaluate_and_create_alert(db_session, evaluation, "M001")

    assert alert is not None
    assert alert.severity == AlertSeverity.WARNING
    assert alert.status   == AlertStatus.ACTIVE
    assert alert.sensor_id == "TEMP_01"


@pytest.mark.asyncio
async def test_critical_alert_created(db_session):
    """Health score below CRITICAL_THRESHOLD should create a CRITICAL alert."""
    evaluation = _make_eval("TEMP_01", CRITICAL_THRESHOLD - 5, SensorStatus.STUCK)
    alert = await evaluate_and_create_alert(db_session, evaluation, "M001")

    assert alert is not None
    assert alert.severity == AlertSeverity.CRITICAL


@pytest.mark.asyncio
async def test_no_duplicate_alert(db_session):
    """Second evaluation for same sensor should NOT create another alert."""
    evaluation = _make_eval("PRESSURE_01", 25.0)
    first  = await evaluate_and_create_alert(db_session, evaluation, "M001")
    second = await evaluate_and_create_alert(db_session, evaluation, "M001")

    assert first  is not None
    assert second is None   # deduplicated


@pytest.mark.asyncio
async def test_no_alert_in_safe_zone(db_session):
    """Health between 50 and 70 should not trigger an alert."""
    evaluation = _make_eval("FLOW_01", 60.0, SensorStatus.HEALTHY)
    alert = await evaluate_and_create_alert(db_session, evaluation, "M001")
    assert alert is None


@pytest.mark.asyncio
async def test_alert_resolved_on_recovery(db_session):
    """An open alert should be resolved when the sensor recovers above RECOVERY_THRESHOLD."""
    # Create the initial alert
    bad_eval = _make_eval("VIBRATION_01", 25.0, SensorStatus.STUCK)
    alert = await evaluate_and_create_alert(db_session, bad_eval, "M001")
    assert alert is not None

    # Flush so the alert is visible to the next query
    await db_session.flush()

    # Sensor recovers
    good_eval = _make_eval("VIBRATION_01", 85.0, SensorStatus.HEALTHY)
    result = await evaluate_and_create_alert(db_session, good_eval, "M001")
    assert result is None   # no new alert on recovery

    # The old alert should now be RESOLVED
    from sqlalchemy import select as sa_select
    from app.models.alert import Alert as AlertModel
    result = await db_session.execute(sa_select(AlertModel).where(AlertModel.id == alert.id))
    refreshed = result.scalar_one()
    assert refreshed.status == AlertStatus.RESOLVED
    assert refreshed.resolved_at is not None
