from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import app
from app.security import SecurityMiddleware, SlidingWindowLimiter, validate_profile_id


def test_security_headers_and_request_id_are_present():
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert response.headers.get("x-request-id")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert "microphone=(self)" in response.headers["permissions-policy"]


def test_security_middleware_rejects_large_declared_body():
    api = FastAPI()

    @api.post("/secure")
    async def secure_endpoint():
        return {"ok": True}

    api.add_middleware(
        SecurityMiddleware,
        max_body_bytes=10,
        rate_limit_enabled=False,
        api_auth_enabled=False,
    )
    with TestClient(api) as client:
        response = client.post("/secure", content=b"01234567890")
    assert response.status_code == 413
    assert response.json()["detail"] == "Request body is too large for CEREBRO."
    assert response.headers.get("x-request-id")


def test_security_middleware_supports_optional_api_key():
    api = FastAPI()

    @api.get("/secure")
    async def secure_endpoint():
        return {"ok": True}

    api.add_middleware(
        SecurityMiddleware,
        rate_limit_enabled=False,
        api_auth_enabled=True,
        api_key="super-secret",
    )
    with TestClient(api) as client:
        assert client.get("/secure").status_code == 401
        assert client.get("/secure", headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.get("/secure", headers={"X-API-Key": "super-secret"}).status_code == 200
        assert client.options("/secure").status_code != 401


def test_security_middleware_reports_missing_server_api_key_as_configuration_error():
    api = FastAPI()

    @api.get("/secure")
    async def secure_endpoint():
        return {"ok": True}

    api.add_middleware(
        SecurityMiddleware,
        rate_limit_enabled=False,
        api_auth_enabled=True,
        api_key=None,
    )
    with TestClient(api) as client:
        response = client.get("/secure")
    assert response.status_code == 503
    assert "no server API key" in response.json()["detail"]


def test_sliding_window_limiter_enforces_limit_and_reset():
    limiter = SlidingWindowLimiter(limit=2, window_seconds=60)
    assert limiter.allow("client", now=100.0)
    assert limiter.allow("client", now=101.0)
    assert not limiter.allow("client", now=102.0)
    assert limiter.allow("other", now=102.0)
    limiter.reset()
    assert limiter.allow("client", now=103.0)


def test_profile_id_validation_rejects_path_traversal():
    try:
        validate_profile_id("../secret")
    except ValueError as exc:
        assert "Profile ID" in str(exc)
    else:
        raise AssertionError("Expected unsafe profile ID to be rejected")
