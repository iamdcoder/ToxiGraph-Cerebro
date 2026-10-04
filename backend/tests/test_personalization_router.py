from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers import personalization
from app.personalization.service import PersonalizedCalibrationService


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(personalization, "_service", PersonalizedCalibrationService(tmp_path))
    return TestClient(app)


def base_payload(label: str, index: int) -> dict:
    labels = ["angry", "happy", "neutral", "sad", "fear", "disgust", "surprised"]
    probabilities = {emotion: 0.04 for emotion in labels}
    probabilities[label] = 0.76
    return {
        "session_id": f"router-{index}",
        "label": label,
        "quality_score": 0.9,
        "fused_probabilities": probabilities,
        "acoustic": {
            "pitch_mean_hz": 180 + index,
            "pitch_std_hz": 30,
            "energy_mean": 0.08,
            "energy_std": 0.03,
            "estimated_syllable_rate_sps": 3.4,
            "estimated_pause_ratio": 0.22,
        },
    }


def test_status_route(client):
    response = client.get("/api/v1/audio/personalization/status")
    assert response.status_code == 200
    assert response.json()["minimum_feedback"] == 8


def test_feedback_route_activates_after_diverse_feedback(client):
    for index, label in enumerate(["neutral", "happy", "angry"] * 4):
        response = client.post(f"/api/v1/audio/personalization/speaker1/feedback", json=base_payload(label, index))
        assert response.status_code == 200
    payload = response.json()
    assert payload["status"]["model_ready"] is True


def test_predict_route_returns_global_fallback_before_training(client):
    response = client.post(
        "/api/v1/audio/personalization/speaker1/predict",
        json={
            "fused_probabilities": {"angry": 0.8, "neutral": 0.1, "happy": 0.1},
            "acoustic": {
                "pitch_mean_hz": 180,
                "pitch_std_hz": 30,
                "energy_mean": 0.08,
                "energy_std": 0.03,
                "estimated_syllable_rate_sps": 3.4,
                "estimated_pause_ratio": 0.22,
            },
        },
    )
    assert response.status_code == 200
    assert response.json()["prediction"]["model_ready"] is False
    assert response.json()["prediction"]["global_emotion"] == "angry"


def test_delete_route_is_idempotent(client):
    response = client.delete("/api/v1/audio/personalization/speaker1")
    assert response.status_code == 200
    assert response.json()["success"] is True
