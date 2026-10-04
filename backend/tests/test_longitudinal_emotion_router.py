from __future__ import annotations

from fastapi.testclient import TestClient

import app.routers.longitudinal_emotion as router_module
from app.main import app
from app.longitudinal_emotion.service import LongitudinalEmotionService


def payload(index: int = 0) -> dict:
    return {
        "session_id": f"router-{index}",
        "recorded_at": f"2026-02-{index + 1:02d}T12:00:00Z",
        "duration_seconds": 5,
        "emotion": "neutral",
        "confidence": 0.8,
        "quality_score": 0.9,
        "baseline_deviation": 0.2,
        "acoustic": {
            "pitch_mean_hz": 210,
            "energy_mean": 0.05,
            "estimated_syllable_rate_sps": 2.5,
        },
        "affect": {"valence": 0.5, "arousal": 0.4, "dominance": 0.6},
    }


def test_history_status_route():
    with TestClient(app) as client:
        response = client.get("/api/v1/audio/history/status")
    assert response.status_code == 200
    body = response.json()
    assert body["stores_audio"] is False
    assert body["max_sessions"] == 100


def test_history_save_and_get_route(tmp_path, monkeypatch):
    monkeypatch.setattr(router_module, "_service", LongitudinalEmotionService(tmp_path))
    with TestClient(app) as client:
        save = client.post("/api/v1/audio/history/browser-1/sessions", json=payload())
        assert save.status_code == 200
        assert save.json()["success"] is True

        get = client.get("/api/v1/audio/history/browser-1")
        assert get.status_code == 200
        body = get.json()
        assert body["session_count"] == 1
        assert body["sessions"][0]["session_id"] == "router-0"


def test_history_delete_route(tmp_path, monkeypatch):
    monkeypatch.setattr(router_module, "_service", LongitudinalEmotionService(tmp_path))
    with TestClient(app) as client:
        client.post("/api/v1/audio/history/browser-1/sessions", json=payload())
        delete = client.delete("/api/v1/audio/history/browser-1")
        assert delete.status_code == 200
        assert delete.json()["success"] is True
        missing = client.delete("/api/v1/audio/history/browser-1")
        assert missing.status_code == 404
