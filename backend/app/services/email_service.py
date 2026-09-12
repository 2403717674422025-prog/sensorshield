"""
services/email_service.py
==========================
Email notification service for SensorShield.

Sends email alerts when a sensor goes CRITICAL or WARNING.
Uses Python's built-in smtplib — no extra dependencies needed.

Supports:
  - Gmail (with App Password)
  - Outlook / Hotmail
  - Any SMTP server

Configuration (add to .env):
  EMAIL_ENABLED=true
  EMAIL_HOST=smtp.gmail.com
  EMAIL_PORT=587
  EMAIL_USER=your@gmail.com
  EMAIL_PASSWORD=your-app-password
  EMAIL_TO=recipient@example.com
"""

from __future__ import annotations

import logging
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


def _get_email_config() -> dict:
    """Read email config from settings/env."""
    import os
    return {
        "enabled":  os.getenv("EMAIL_ENABLED", "false").lower() == "true",
        "host":     os.getenv("EMAIL_HOST", "smtp.gmail.com"),
        "port":     int(os.getenv("EMAIL_PORT", "587")),
        "user":     os.getenv("EMAIL_USER", ""),
        "password": os.getenv("EMAIL_PASSWORD", ""),
        "to":       os.getenv("EMAIL_TO", ""),
    }


def _build_html(
    sensor_id: str,
    machine_id: str,
    severity: str,
    fault_type: str,
    health_score: float,
    description: str,
    impact: str,
    action: str,
    timestamp: str,
) -> str:
    """Build a styled HTML email body."""
    color = "#ff4d6a" if severity == "CRITICAL" else "#f0b429"
    icon  = "🔴" if severity == "CRITICAL" else "⚠️"

    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f4f6f9; margin: 0; padding: 20px; }}
    .container {{ max-width: 600px; margin: 0 auto; background: #fff; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.1); }}
    .header {{ background: linear-gradient(135deg, #080c14, #1a2640); padding: 28px 32px; }}
    .header-title {{ color: #fff; font-size: 20px; font-weight: 800; margin: 0; }}
    .header-sub {{ color: #6b82a8; font-size: 13px; margin-top: 4px; }}
    .badge {{ display: inline-block; background: {color}22; color: {color}; border: 1px solid {color}44; border-radius: 20px; padding: 4px 12px; font-size: 12px; font-weight: 700; margin-top: 12px; }}
    .body {{ padding: 28px 32px; }}
    .metric-row {{ display: flex; gap: 16px; margin-bottom: 20px; }}
    .metric {{ flex: 1; background: #f8f9fb; border-radius: 10px; padding: 14px 16px; text-align: center; }}
    .metric-label {{ font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }}
    .metric-value {{ font-size: 22px; font-weight: 800; color: #1a1a2e; }}
    .section {{ background: #f8f9fb; border-radius: 10px; padding: 16px; margin-bottom: 14px; }}
    .section-label {{ font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; font-weight: 600; }}
    .section-text {{ font-size: 13px; color: #333; line-height: 1.6; }}
    .action-box {{ background: #edf7f0; border: 1px solid #b7e4c7; border-radius: 10px; padding: 16px; margin-bottom: 14px; }}
    .action-label {{ font-size: 11px; color: #2d6a4f; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; font-weight: 700; }}
    .action-text {{ font-size: 13px; color: #2d6a4f; line-height: 1.6; }}
    .footer {{ background: #f8f9fb; padding: 16px 32px; text-align: center; font-size: 12px; color: #999; }}
    .divider {{ height: 1px; background: #eee; margin: 20px 0; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div style="display:flex; align-items:center; gap:10px;">
        <span style="font-size:28px;">🛡️</span>
        <div>
          <div class="header-title">SensorShield Alert</div>
          <div class="header-sub">Industrial Sensor Monitoring System</div>
        </div>
      </div>
      <div class="badge">{icon} {severity} ALERT — {fault_type}</div>
    </div>

    <div class="body">
      <div class="metric-row">
        <div class="metric">
          <div class="metric-label">Sensor</div>
          <div class="metric-value" style="font-size:16px;">{sensor_id}</div>
        </div>
        <div class="metric">
          <div class="metric-label">Machine</div>
          <div class="metric-value" style="font-size:16px;">{machine_id}</div>
        </div>
        <div class="metric">
          <div class="metric-label">Health Score</div>
          <div class="metric-value" style="color:{color};">{health_score:.1f}</div>
        </div>
      </div>

      <div class="section">
        <div class="section-label">📋 Description</div>
        <div class="section-text">{description}</div>
      </div>

      <div class="section">
        <div class="section-label">⚡ Impact</div>
        <div class="section-text">{impact}</div>
      </div>

      <div class="action-box">
        <div class="action-label">✅ Recommended Action</div>
        <div class="action-text">{action}</div>
      </div>

      <div class="divider"></div>
      <div style="font-size:12px; color:#999;">
        🕐 Alert triggered at: <strong>{timestamp}</strong>
      </div>
    </div>

    <div class="footer">
      This is an automated alert from SensorShield monitoring system.<br>
      Do not reply to this email.
    </div>
  </div>
</body>
</html>
"""


async def send_alert_email(
    sensor_id: str,
    machine_id: str,
    severity: str,
    fault_type: str,
    health_score: float,
    description: str,
    impact: str = "",
    action: str = "",
    timestamp: Optional[str] = None,
) -> bool:
    """
    Send an email alert for a sensor event.
    Returns True if sent successfully, False otherwise.
    Failures are logged but never raise — email issues must not crash the pipeline.
    """
    cfg = _get_email_config()

    if not cfg["enabled"]:
        logger.debug("Email notifications disabled (EMAIL_ENABLED=false)")
        return False

    if not cfg["user"] or not cfg["password"] or not cfg["to"]:
        logger.warning("Email config incomplete — set EMAIL_USER, EMAIL_PASSWORD, EMAIL_TO in .env")
        return False

    ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    subject = f"[SensorShield] {severity} Alert — {sensor_id} ({fault_type})"

    html_body = _build_html(
        sensor_id=sensor_id,
        machine_id=machine_id,
        severity=severity,
        fault_type=fault_type,
        health_score=health_score,
        description=description,
        impact=impact,
        action=action,
        timestamp=ts,
    )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"SensorShield <{cfg['user']}>"
    msg["To"]      = cfg["to"]
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(cfg["host"], cfg["port"], timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(cfg["user"], cfg["password"])
            server.sendmail(cfg["user"], cfg["to"], msg.as_string())
        logger.info(
            "Alert email sent: sensor=%s severity=%s → %s",
            sensor_id, severity, cfg["to"]
        )
        return True
    except Exception as exc:
        logger.error("Failed to send alert email: %s", exc)
        return False
