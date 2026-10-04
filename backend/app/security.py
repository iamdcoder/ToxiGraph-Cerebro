from __future__ import annotations

import hmac
import time
import uuid
from collections import defaultdict, deque
from threading import Lock
from typing import Awaitable, Callable

from fastapi import Request, WebSocket
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class SecurityPolicyError(ValueError):
    """Raised when a configured security policy cannot be satisfied."""


class SlidingWindowLimiter:
    """Small in-process sliding-window rate limiter.

    This is intentionally process-local. Multi-worker deployments should place
    a shared rate limiter at the reverse proxy or gateway in front of CEREBRO.
    """

    def __init__(self, limit: int = 120, window_seconds: float = 60.0) -> None:
        if limit <= 0 or window_seconds <= 0:
            raise ValueError("Rate limiter limit and window must be positive.")
        self.limit = int(limit)
        self.window_seconds = float(window_seconds)
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        timestamp = time.monotonic() if now is None else float(now)
        cutoff = timestamp - self.window_seconds
        with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.limit:
                return False
            events.append(timestamp)
            if len(self._events) > 2048:
                stale_keys = [name for name, values in self._events.items() if not values or values[-1] <= cutoff]
                for name in stale_keys[:512]:
                    self._events.pop(name, None)
            return True

    def reset(self) -> None:
        with self._lock:
            self._events.clear()


def client_ip(scope: Scope) -> str:
    client = scope.get("client")
    if isinstance(client, (list, tuple)) and client:
        return str(client[0])
    return "unknown"


def header_value(scope: Scope, name: str) -> str | None:
    target = name.lower().encode("latin-1")
    for key, value in scope.get("headers", []):
        if key.lower() == target:
            return value.decode("latin-1", errors="ignore")
    return None


def origin_allowed(origin: str | None, allowed_origins: list[str]) -> bool:
    if not origin:
        return True
    return origin.rstrip("/") in {item.rstrip("/") for item in allowed_origins if item.strip()}


def validate_profile_id(profile_id: str) -> str:
    value = str(profile_id).strip()
    if not value or len(value) > 64:
        raise SecurityPolicyError("Profile ID must be 1–64 characters.")
    if not all(char.isalnum() or char in "-_" for char in value):
        raise SecurityPolicyError("Profile ID may contain only letters, numbers, '-' or '_'.")
    return value


def api_key_matches(provided: str | None, expected: str | None) -> bool:
    if not expected:
        return False
    if not provided:
        return False
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


class ConnectionLimiter:
    def __init__(self, max_connections_per_key: int = 3) -> None:
        if max_connections_per_key <= 0:
            raise ValueError("Maximum connections per key must be positive.")
        self.max_connections_per_key = int(max_connections_per_key)
        self._active: dict[str, int] = defaultdict(int)
        self._lock = Lock()

    def acquire(self, key: str) -> bool:
        with self._lock:
            current = self._active[key]
            if current >= self.max_connections_per_key:
                return False
            self._active[key] = current + 1
            return True

    def release(self, key: str) -> None:
        with self._lock:
            current = self._active.get(key, 0)
            if current <= 1:
                self._active.pop(key, None)
            else:
                self._active[key] = current - 1

    def reset(self) -> None:
        with self._lock:
            self._active.clear()


GLOBAL_WS_CONNECTION_LIMITER = ConnectionLimiter(max_connections_per_key=3)


def websocket_policy_ok(websocket: WebSocket, allowed_origins: list[str]) -> bool:
    return origin_allowed(websocket.headers.get("origin"), allowed_origins)


class SecurityMiddleware:
    """Apply request IDs, size limits, HTTP rate limits and security headers."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        enabled: bool = True,
        max_body_bytes: int = 20 * 1024 * 1024,
        rate_limit_enabled: bool = True,
        rate_limit_requests: int = 120,
        rate_limit_window_seconds: float = 60.0,
        api_auth_enabled: bool = False,
        api_key: str | None = None,
        exempt_paths: tuple[str, ...] = ("/", "/api/v1/health/", "/api/v1/health"),
        hsts_enabled: bool = False,
    ) -> None:
        self.app = app
        self.enabled = enabled
        self.max_body_bytes = int(max_body_bytes)
        self.rate_limit_enabled = rate_limit_enabled
        self.api_auth_enabled = api_auth_enabled
        self.api_key = api_key
        self.exempt_paths = tuple(exempt_paths)
        self.hsts_enabled = hsts_enabled
        self.limiter = SlidingWindowLimiter(rate_limit_requests, rate_limit_window_seconds)
        self._handler: Callable[[Scope, Receive, Send], Awaitable[None]] = app

    @staticmethod
    async def _send_json(send: Send, status: int, detail: str, request_id: str) -> None:
        import json

        body = json.dumps({"detail": detail, "request_id": request_id}).encode("utf-8")
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
            (b"cache-control", b"no-store"),
            (b"x-request-id", request_id.encode("ascii")),
            (b"x-content-type-options", b"nosniff"),
            (b"x-frame-options", b"DENY"),
            (b"referrer-policy", b"no-referrer"),
            (b"permissions-policy", b"microphone=(self), camera=(), geolocation=()"),
        ]
        if status == 429:
            headers.append((b"retry-after", b"1"))
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not self.enabled or scope.get("type") != "http":
            return await self.app(scope, receive, send)

        request_id = self._request_id(scope)
        path = str(scope.get("path", ""))
        ip = client_ip(scope)
        method = str(scope.get("method", "GET"))
        content_length = header_value(scope, "content-length")

        if content_length:
            try:
                declared = int(content_length)
            except ValueError:
                await self._send_json(send, 400, "Invalid Content-Length header.", request_id)
                return
            if declared < 0 or declared > self.max_body_bytes:
                await self._send_json(send, 413, "Request body is too large for CEREBRO.", request_id)
                return

        if self.api_auth_enabled and method != "OPTIONS" and not self._is_exempt(path):
            if not self.api_key:
                await self._send_json(send, 503, "API authentication is enabled but no server API key is configured.", request_id)
                return
            provided = header_value(scope, "x-api-key")
            if not api_key_matches(provided, self.api_key):
                await self._send_json(send, 401, "Valid API authentication is required.", request_id)
                return

        if self.rate_limit_enabled and method != "OPTIONS":
            key = f"{ip}:{method}:{path}"
            if not self.limiter.allow(key):
                await self._send_json(send, 429, "Rate limit exceeded. Retry shortly.", request_id)
                return

        async def send_with_headers(message: Message) -> None:
            if message.get("type") == "http.response.start":
                headers = list(message.get("headers", []))
                normalized = {key.lower() for key, _ in headers}
                self._append_header(headers, normalized, b"x-request-id", request_id.encode("ascii"))
                self._append_header(headers, normalized, b"x-content-type-options", b"nosniff")
                self._append_header(headers, normalized, b"x-frame-options", b"DENY")
                self._append_header(headers, normalized, b"referrer-policy", b"no-referrer")
                self._append_header(
                    headers,
                    normalized,
                    b"permissions-policy",
                    b"microphone=(self), camera=(), geolocation=()",
                )
                if path.startswith("/api/"):
                    self._append_header(headers, normalized, b"cache-control", b"no-store")
                if self.hsts_enabled:
                    self._append_header(
                        headers,
                        normalized,
                        b"strict-transport-security",
                        b"max-age=31536000; includeSubDomains",
                    )
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_headers)

    @staticmethod
    def _append_header(headers: list[tuple[bytes, bytes]], normalized: set[bytes], key: bytes, value: bytes) -> None:
        lowered = key.lower()
        if lowered not in normalized:
            headers.append((lowered, value))
            normalized.add(lowered)

    @staticmethod
    def _request_id(scope: Scope) -> str:
        supplied = header_value(scope, "x-request-id")
        if supplied and len(supplied) <= 64 and all(char.isalnum() or char in "-_" for char in supplied):
            return supplied
        return uuid.uuid4().hex

    def _is_exempt(self, path: str) -> bool:
        return path in self.exempt_paths
