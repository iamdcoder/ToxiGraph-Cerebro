from dataclasses import dataclass

import numpy as np
import pytest

from app.affect_dynamics.engine import AffectDynamicsAnalyzer, AffectDynamicsConfig, AffectDynamicsError
from app.dimensional_emotion.model import DimensionalEmotionMetadata


@dataclass
class FakePrediction:
    valence: float
    arousal: float
    dominance: float

    @property
    def values(self):
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
        }


class FakeDimensionalModel:
    def __init__(self, states):
        self.states = list(states)
        self.index = 0

    def predict(self, samples, sample_rate):
        state = self.states[min(self.index, len(self.states) - 1)]
        self.index += 1
        return FakePrediction(*state)


def speech(seconds=5.0, sample_rate=16000):
    t = np.linspace(0, seconds, int(seconds * sample_rate), endpoint=False)
    return (0.08 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_config_rejects_invalid_thresholds():
    with pytest.raises(AffectDynamicsError):
        AffectDynamicsAnalyzer(AffectDynamicsConfig(event_delta_threshold=0))


def test_empty_audio_rejected():
    with pytest.raises(AffectDynamicsError):
        AffectDynamicsAnalyzer().analyze(np.zeros(0, dtype=np.float32), 16000, dimensional_model=FakeDimensionalModel([(0.5, 0.5, 0.5)]))


def test_silence_produces_no_speech_points():
    result = AffectDynamicsAnalyzer().analyze(
        np.zeros(5 * 16000, dtype=np.float32),
        16000,
        dimensional_model=FakeDimensionalModel([(0.5, 0.5, 0.5)]),
    )
    assert result.speech_coverage == 0
    assert all(not point.is_speech for point in result.points)
    assert result.dominant_quadrant is None


def test_constant_affect_has_zero_net_change_and_high_stability():
    result = AffectDynamicsAnalyzer().analyze(
        speech(),
        16000,
        dimensional_model=FakeDimensionalModel([(0.5, 0.5, 0.5)]),
    )
    assert result.start_state == result.end_state
    assert result.net_change == {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
    assert result.volatility == 0.0
    assert result.stability_score == 1.0
    assert result.events == ()


def test_rising_arousal_and_falling_valence_create_events():
    states = [
        (0.70, 0.40, 0.50),
        (0.58, 0.60, 0.55),
        (0.45, 0.78, 0.58),
        (0.32, 0.90, 0.62),
    ]
    result = AffectDynamicsAnalyzer().analyze(
        speech(6.0), 16000, dimensional_model=FakeDimensionalModel(states)
    )
    event_types = {event.event_type for event in result.events}
    dimensions = {(event.event_type, event.dimension) for event in result.events}
    assert "dimension_rise" in event_types
    assert "dimension_fall" in event_types
    assert ("dimension_rise", "arousal") in dimensions
    assert ("dimension_fall", "valence") in dimensions
    assert result.end_state["arousal"] > result.start_state["arousal"]
    assert result.end_state["valence"] < result.start_state["valence"]


def test_large_trajectory_shift_creates_affect_shift_event():
    states = [
        (0.75, 0.25, 0.50),
        (0.25, 0.85, 0.70),
        (0.20, 0.90, 0.75),
    ]
    result = AffectDynamicsAnalyzer().analyze(
        speech(5.0), 16000, dimensional_model=FakeDimensionalModel(states)
    )
    assert any(event.event_type == "affect_shift" for event in result.events)
    assert result.largest_shift > 0.12


def test_quadrant_transition_is_detected():
    states = [
        (0.80, 0.20, 0.50),
        (0.20, 0.85, 0.50),
        (0.15, 0.90, 0.55),
    ]
    result = AffectDynamicsAnalyzer().analyze(
        speech(5.0), 16000, dimensional_model=FakeDimensionalModel(states)
    )
    assert any(event.event_type == "quadrant_transition" for event in result.events)


def test_peak_and_low_points_are_reported():
    states = [
        (0.60, 0.30, 0.50),
        (0.40, 0.80, 0.55),
        (0.25, 0.70, 0.60),
    ]
    result = AffectDynamicsAnalyzer().analyze(
        speech(5.0), 16000, dimensional_model=FakeDimensionalModel(states)
    )
    assert result.peak_arousal is not None
    assert result.peak_arousal_at_seconds is not None
    assert result.lowest_valence is not None
    assert result.lowest_valence_at_seconds is not None


def test_smoothed_values_stay_inside_unit_interval():
    result = AffectDynamicsAnalyzer().analyze(
        speech(5.0),
        16000,
        dimensional_model=FakeDimensionalModel([(0.2, 0.9, 0.1), (0.9, 0.1, 0.9)]),
    )
    for point in result.points:
        if point.smoothed:
            assert all(0 <= value <= 1 for value in point.smoothed.values())


def test_api_status(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.routers import affect_dynamics

    monkeypatch.setattr(
        affect_dynamics,
        "get_status",
        lambda: {"available": True, "enabled": True, "model_id": "fake"},
    )
    response = TestClient(app).get("/api/v1/audio/emotion/affect/dynamics/status")
    assert response.status_code == 200
    assert response.json()["feature"] == "affect_dynamics"


def test_api_success(monkeypatch):
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    from app.main import app
    from app.routers import affect_dynamics

    processed = SimpleNamespace(
        samples=speech(4.0),
        sample_rate=16000,
        channels=1,
        original_sample_rate=16000,
        original_channels=1,
        original_sample_width_bytes=2,
        duration_seconds=4.0,
        rms=0.05,
        peak=0.08,
        is_silent=False,
    )

    class FakeModel:
        def predict(self, samples, sample_rate):
            return FakePrediction(0.55, 0.65, 0.50)

    monkeypatch.setattr(affect_dynamics.AudioProcessor, "process", lambda self, _: processed)
    monkeypatch.setattr(affect_dynamics, "get_model", lambda: FakeModel())
    monkeypatch.setattr(affect_dynamics.settings, "DIMENSIONAL_EMOTION_ENABLED", True)

    response = TestClient(app).post(
        "/api/v1/audio/emotion/affect/dynamics",
        files={"file": ("sample.wav", b"RIFF", "audio/wav")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["points"]
    assert payload["dominant_quadrant"] in {
        "positive / activated",
        "positive / calm",
        "mixed / transitional",
        "negative / activated",
        "negative / calm",
    }
