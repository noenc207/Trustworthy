"""
API Tests for health check endpoints.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.mark.api
class TestHealthEndpoints:
    """Tests for /health and /ready endpoints."""

    @pytest.fixture(scope="class")
    def client(self):
        from src.api.main import create_app
        app = create_app()
        with TestClient(app) as c:
            yield c

    def test_health_returns_200(self, client):
        """GET /health should return HTTP 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self, client):
        """Health response should include status, version, environment."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "version" in data
        assert "environment" in data

    def test_readiness_returns_200(self, client):
        """GET /ready should return HTTP 200."""
        response = client.get("/ready")
        assert response.status_code == 200
