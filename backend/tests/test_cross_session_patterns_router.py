from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

import app.routers.cross_session_patterns as router_module
from app.main import app
from app.longitudinal_emotion.service import LongitudinalEmotionService


def payload(index: int, emotion: str = "neutral", *, valence: float = 0.5, arousal: float = 0.5, dominance: float = 0.5) -> dict:
    return {
        "session_id": f"pattern-{index}",
        "recorded_at": (datetime(2026, 2, 1, tzinfo=timezone.utc) + timedelta(days=index)).isoformat(),
        "duration_seconds": 5,
        "emotion": emotion,
        "confidence": 0.8,
        "quality_score": 0.9,
        "baseline_deviation": 0.2,
        "acoustic": {
            "pitch_mean_hz": 200 + index * 4,
            "pitch_std_hz": 20,
            "energy_mean": 0.05 + index * 0.01,
            "estimated_syllable_rate_sps": 2.5 + index * 0.05,
            "estimated_pause_ratio": 0.15,
            "spectral_centroid_mean_hz": 1800 + index * 30,
        },
        "affect": {"valence": valence, "arousal": arousal, "dominance": dominance},
    }


def test_pattern_status_route():
    with TestClient(app) as client:
        response = client.get("/api/v1/audio/history-patterns/status")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "cross_session_pattern_engine"
    assert body["minimum_sessions"] == 4


def test_pattern_route_returns_insufficient_history(tmp_path, monkeypatch):
    service = LongitudinalEmotionService(tmp_path)
    for i in range(3):
        service.append("browser-patterns", payload(i))
    monkeypatch.setattr(router_module, "_history", service)
    with TestClient(app) as client:
        response = client.get("/api/v1/audio/history-patterns/browser-patterns")
    assert response.status_code == 200
    report = response.json()["report"]
    assert report["available"] is False
    assert report["session_count"] == 3


def test_pattern_route_returns_discovered_patterns(tmp_path, monkeypatch):
    service = LongitudinalEmotionService(tmp_path)
    emotions = ["neutral", "angry", "neutral", "angry", "neutral", "angry"]
    for i, emotion in enumerate(emotions):
        service.append("browser-patterns", payload(i, emotion, valence=0.3 if emotion == "angry" else 0.6, arousal=0.8 if emotion == "angry" else 0.4))
    monkeypatch.setattr(router_module, "_history", service)
    with TestClient(app) as client:
        response = client.get("/api/v1/audio/history-patterns/browser-patterns")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["report"]["available"] is True
    assert body["report"]["patterns"]
    assert body["report"]["repeated_transitions"]
