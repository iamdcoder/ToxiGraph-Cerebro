from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

import app.routers.voice_change as router_module
from app.main import app
from app.longitudinal_emotion.service import LongitudinalEmotionService


def payload(i: int, *, pitch=200, energy=0.05) -> dict:
    return {
        "session_id": f"s{i}",
        "recorded_at": (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)).isoformat(),
        "duration_seconds": 5,
        "emotion": "neutral",
        "confidence": 0.8,
        "quality_score": 0.9,
        "acoustic": {
            "pitch_mean_hz": pitch,
            "pitch_std_hz": 20,
            "energy_mean": energy,
            "estimated_syllable_rate_sps": 2.5,
            "estimated_pause_ratio": 0.15,
            "spectral_centroid_mean_hz": 1800,
        },
        "affect": {"valence": 0.5, "arousal": 0.5, "dominance": 0.5},
    }


def test_status_route():
    with TestClient(app) as client:
        response = client.get("/api/v1/audio/voice-change/status")
    assert response.status_code == 200
    assert response.json()["service"] == "voice_change_detection"


def test_current_change_route(tmp_path, monkeypatch):
    history_service = LongitudinalEmotionService(tmp_path)
    for i in range(4):
        history_service.append("browser-1", payload(i))
    monkeypatch.setattr(router_module, "_history_service", history_service)
    with TestClient(app) as client:
        response = client.post("/api/v1/audio/voice-change/browser-1/change", json=payload(10, pitch=340, energy=0.2))
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["report"]["available"] is True
    assert body["report"]["level"] in {"minor_change", "significant_change", "unusual"}


def test_latest_change_route_requires_saved_session(tmp_path, monkeypatch):
    history_service = LongitudinalEmotionService(tmp_path)
    monkeypatch.setattr(router_module, "_history_service", history_service)
    with TestClient(app) as client:
        response = client.get("/api/v1/audio/voice-change/browser-1/change")
    assert response.status_code == 404


def test_latest_change_route_returns_saved_comparison(tmp_path, monkeypatch):
    history_service = LongitudinalEmotionService(tmp_path)
    for i in range(5):
        history_service.append("browser-1", payload(i))
    history_service.append("browser-1", payload(9, pitch=340, energy=0.2))
    monkeypatch.setattr(router_module, "_history_service", history_service)
    with TestClient(app) as client:
        response = client.get("/api/v1/audio/voice-change/browser-1/change")
    assert response.status_code == 200
    report = response.json()["report"]
    assert report["current_session_id"] == "s9"
    assert report["historical_session_count"] == 5
    assert report["available"] is True
