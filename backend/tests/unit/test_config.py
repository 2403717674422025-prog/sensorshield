"""
tests/unit/test_config.py
=========================
Verify that the settings object loads correctly.
These tests run without a database or network.
"""

from app.core.config import settings


def test_settings_load():
    """Settings object must be constructable without errors."""
    assert settings is not None


def test_app_version_format():
    """Version must follow major.minor.patch."""
    parts = settings.APP_VERSION.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)


def test_cors_origins_list():
    """CORS origins must be parsed into a list."""
    origins = settings.cors_origins_list
    assert isinstance(origins, list)
    assert len(origins) >= 1


def test_threshold_values():
    """Trust thresholds must be logically ordered."""
    assert 0 < settings.TRUST_LOW_THRESHOLD < settings.TRUST_HIGH_THRESHOLD < 100
