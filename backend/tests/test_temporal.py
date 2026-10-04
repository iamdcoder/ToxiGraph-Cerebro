from __future__ import annotations

import io
import wave
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from app.fusion.calibration import FusionCalibrationArtifact
from app.fusion.constants import CANONICAL_LABELS
from app.fusion.engine import FusionEngine
from app.main import app
from app.temporal.engine import TemporalAnalysisError, TemporalConfig, TemporalEmotionAnalyzer


client = TestClient(app)


def make_wav_from_samples(samples: np.ndarray, sample_rate: int = 16_000) -> bytes:
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


def make_segmented_signal(sample_rate: int = 16_000) -> np.ndarray:
    t = np.arange(9 * sample_rate, dtype=np.float32) / sample_rate
    first = 0.08 * np.sin(2 * np.pi * 220 * t[: 3 * sample_rate])
    silence = np.zeros(3 * sample_rate, dtype=np.float32)
    third = 0.45 * np.sin(2 * np.pi * 220 * t[: 3 * sample_rate])
    return np.concatenate([first, silence, third]).astype(np.float32)


def make_engine() -> FusionEngine:
    artifact = FusionCalibrationArtifact(
        version="temporal-fixture",
        labels=tuple(CANONICAL_LABELS),
        classical_temperature=1.0,
        deep_temperature=1.0,
        classical_weight=0.5,
        deep_weight=0.5,
        validation_metrics={},
        metadata={"fixture": True},
    )
    return FusionEngine(artifact)


class FakeClassical:
    def predict(self, vector):
        energy = float(np.mean(np.abs(vector)))
        target = "angry" if energy > 0.15 else "neutral"
        other = 1.0 - 0.85
        probabilities = {label: other / (len(CANONICAL_LABELS) - 1) for label in CANONICAL_LABELS}
        probabilities[target] = 0.85
        return {
            "emotion": target,
            "confidence": 0.85,
            "probabilities": probabilities,
            "model_name": "fixture-classical",
            "feature_version": "acoustic-v1",
            "training_metadata": {"fixture": True},
        }


class FakeDeep:
    def predict(self, samples, sample_rate):
        energy = float(np.mean(np.abs(samples)))
        target = "angry" if energy > 0.15 else "neutral"
        probabilities = {label: 0.02 / (len(CANONICAL_LABELS) - 1) for label in CANONICAL_LABELS}
        probabilities[target] = 0.98
        return SimpleNamespace(
            emotion=target,
            confidence=0.98,
            probabilities=probabilities,
            chunks_analyzed=1,
            metadata=SimpleNamespace(model_id="fixture-deep"),
        )


def test_temporal_config_rejects_invalid_values():
    try:
        TemporalEmotionAnalyzer(TemporalConfig(hop_seconds=4.0, window_seconds=3.0))
    except TemporalAnalysisError:
        return
    raise AssertionError("Expected invalid temporal configuration to fail")


def test_temporal_windows_overlap_and_smooth_predictions():
    samples = make_segmented_signal()
    analysis = TemporalEmotionAnalyzer().analyze(
        samples,
        16_000,
        classical_model=FakeClassical(),
        deep_model=FakeDeep(),
        fusion_engine=make_engine(),
    )

    assert analysis.analyzed_windows == 7
    assert analysis.speech_windows < analysis.analyzed_windows
    assert analysis.dominant_emotion == "angry"
    assert len(analysis.segments) == 7
    assert len(analysis.trajectory) == 9
    assert abs(sum(point.duration_seconds for point in analysis.trajectory) - 9.0) < 1e-9
    assert all(point.start_seconds < point.end_seconds for point in analysis.trajectory)
    assert len(analysis.emotion_runs) >= 1
    assert any(segment.transition for segment in analysis.segments)
    assert all(abs(sum(segment.probabilities.values()) - 1.0) < 1e-9 for segment in analysis.segments if segment.is_speech)


def test_temporal_silent_windows_are_not_forced_into_an_emotion():
    samples = make_segmented_signal()
    analysis = TemporalEmotionAnalyzer().analyze(
        samples,
        16_000,
        classical_model=FakeClassical(),
        deep_model=FakeDeep(),
        fusion_engine=make_engine(),
    )
    silent = [segment for segment in analysis.segments if not segment.is_speech]
    assert silent
    assert all(segment.emotion is None for segment in silent)
    assert all(segment.probabilities == {} for segment in silent)


def test_temporal_endpoint_returns_timeline(monkeypatch):
    import app.routers.temporal as temporal_router

    monkeypatch.setattr(temporal_router, "get_classical_model", lambda: FakeClassical())
    monkeypatch.setattr(temporal_router, "get_deep_model", lambda: FakeDeep())
    monkeypatch.setattr(temporal_router, "get_engine", make_engine)

    response = client.post(
        "/api/v1/audio/emotion/temporal",
        files={"file": ("segmented.wav", make_wav_from_samples(make_segmented_signal()), "audio/wav")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["analyzed_windows"] == 7
    assert payload["speech_windows"] < payload["analyzed_windows"]
    assert payload["calibration_version"] == "temporal-fixture"
    assert len(payload["segments"]) == 7
    assert len(payload["trajectory"]) == 9
    assert abs(sum(point["duration_seconds"] for point in payload["trajectory"]) - 9.0) < 1e-9
    assert sum(1 for segment in payload["segments"] if segment["transition"]) >= 1
    assert all(
        abs(sum(item["probability"] for item in segment["probabilities"]) - 1.0) < 1e-9
        for segment in payload["segments"]
        if segment["is_speech"]
    )


def test_temporal_status_describes_window_contract():
    response = client.get("/api/v1/audio/emotion/temporal/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["window_seconds"] == 3.0
    assert payload["hop_seconds"] == 1.0
