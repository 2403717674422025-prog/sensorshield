"""
tests/integration/test_system.py
=================================
Integration tests for the system health endpoint.
These use the ASGI test client so no real network is needed,
but they do exercise the full request → DB → response path.
"""

import pytest


@pytest.mark.asyncio
async def test_health_check_returns_200(client):
    """Health check must always return 200 in test environment."""
    response = await client.get("/api/system/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_check_response_structure(client):
    """Health check response must contain required fields."""
    response = await client.get("/api/system/health")
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "environment" in data
    assert "uptime_seconds" in data
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_readiness_check(client):
    """Readiness probe must return ready=True."""
    response = await client.get("/api/system/ready")
    assert response.status_code == 200
    assert response.json()["ready"] is True


@pytest.mark.asyncio
async def test_machine_list_empty(client):
    """Fresh DB should return empty machine list, not an error."""
    response = await client.get("/api/machines")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_create_and_retrieve_machine(client):
    """Create a machine then retrieve it by ID."""
    payload = {
        "id": "M001",
        "name": "Test Pump",
        "location": "Test Bay",
    }
    create_resp = await client.post("/api/machines", json=payload)
    assert create_resp.status_code == 201
    assert create_resp.json()["id"] == "M001"

    get_resp = await client.get("/api/machines/M001")
    assert get_resp.status_code == 200
    assert get_resp.json()["name"] == "Test Pump"


@pytest.mark.asyncio
async def test_create_machine_duplicate_returns_409(client):
    """Inserting the same machine ID twice must return 409."""
    payload = {"id": "M002", "name": "Duplicate Test"}
    await client.post("/api/machines", json=payload)
    resp = await client.post("/api/machines", json=payload)
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_get_nonexistent_machine_returns_404(client):
    """Requesting a machine that doesn't exist must return 404."""
    resp = await client.get("/api/machines/GHOST_999")
    assert resp.status_code == 404
