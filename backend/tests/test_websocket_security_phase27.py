from __future__ import annotations

import json

import numpy as np
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.config import settings
from app.main import app
from app.routers import live as live_router
from app.routers import live_multi_speaker as live_multi_router
from app.speaker_diarization.model import SpeakerTurn


def pcm_chunk(duration=0.3, sample_rate=16000):
    count = int(duration * sample_rate)
    t = np.arange(count, dtype=np.float32) / sample_rate
    signal = 0.05 * np.sin(2 * np.pi * 220.0 * t)
    return (signal * 32767).astype("<i2").tobytes()


def test_single_live_websocket_rejects_disallowed_origin(monkeypatch):
    monkeypatch.setattr(settings, "LIVE_STREAM_ENABLED", True)
    monkeypatch.setattr(settings, "ALLOWED_ORIGINS", ["http://allowed.test"])
    with TestClient(app) as client:
        try:
            with client.websocket_connect("/api/v1/audio/live/ws", headers={"origin": "http://evil.test"}):
                raise AssertionError("Expected disallowed origin to be rejected")
        except WebSocketDisconnect as exc:
            assert exc.code == 1008


def test_single_live_websocket_requires_token_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "LIVE_STREAM_ENABLED", True)
    monkeypatch.setattr(settings, "WEBSOCKET_AUTH_ENABLED", True)
    monkeypatch.setattr(settings, "WEBSOCKET_AUTH_TOKEN", "secret-token")
    monkeypatch.setattr(live_router, "get_live_runtime", lambda: object())
    with TestClient(app) as client:
        with client.websocket_connect("/api/v1/audio/live/ws") as websocket:
            websocket.send_text(json.dumps({"type": "start", "sample_rate": 16000, "token": "wrong"}))
            payload = websocket.receive_json()
            assert payload["code"] == "unauthorized"


def test_single_live_websocket_accepts_correct_token_with_fixture(monkeypatch):
    from app.live.engine import LiveInferenceResult
    from app.fusion.constants import CANONICAL_LABELS

    class FakeRuntime:
        def analyze_window(self, session, samples, start_seconds, end_seconds, *, latency_ms):
            probabilities = {label: 0.1 / (len(CANONICAL_LABELS) - 1) for label in CANONICAL_LABELS}
            probabilities["neutral"] = 0.9
            return LiveInferenceResult(
                sequence=session.next_sequence(),
                window_start_seconds=start_seconds,
                window_end_seconds=end_seconds,
                emotion="neutral",
                confidence=0.9,
                probabilities=probabilities,
                valence=0.5,
                arousal=0.3,
                dominance=0.5,
                state="calm",
                state_transition=None,
                speech_ratio=1.0,
                quality_score=0.9,
                quality_verdict="good",
                latency_ms=1.0,
                model_metadata={"fixture": True},
            )

    monkeypatch.setattr(settings, "LIVE_STREAM_ENABLED", True)
    monkeypatch.setattr(settings, "WEBSOCKET_AUTH_ENABLED", True)
    monkeypatch.setattr(settings, "WEBSOCKET_AUTH_TOKEN", "secret-token")
    monkeypatch.setattr(live_router, "get_live_runtime", lambda: FakeRuntime())
    monkeypatch.setattr(settings, "ALLOWED_ORIGINS", [])
    with TestClient(app) as client:
        with client.websocket_connect("/api/v1/audio/live/ws") as websocket:
            websocket.send_text(json.dumps({"type": "start", "sample_rate": 16000, "token": "secret-token"}))
            ready = websocket.receive_json()
            assert ready["type"] == "ready"
            websocket.send_bytes(pcm_chunk(3.1))
            analysis = websocket.receive_json()
            assert analysis["type"] == "analysis"
            websocket.send_text(json.dumps({"type": "stop"}))
            complete = websocket.receive_json()
            assert complete["type"] == "complete"


def test_multi_live_websocket_rejects_disallowed_origin(monkeypatch):
    monkeypatch.setattr(settings, "LIVE_MULTI_SPEAKER_ENABLED", True)
    monkeypatch.setattr(settings, "ALLOWED_ORIGINS", ["http://allowed.test"])
    with TestClient(app) as client:
        try:
            with client.websocket_connect("/api/v1/audio/live-multi/ws", headers={"origin": "http://evil.test"}):
                raise AssertionError("Expected disallowed origin to be rejected")
        except WebSocketDisconnect as exc:
            assert exc.code == 1008
