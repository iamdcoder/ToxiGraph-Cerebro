from __future__ import annotations

import io
import wave

import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.personal_baseline.service import PersonalBaselineError, PersonalBaselineService


client = TestClient(app)


def make_wav(
    *,
    duration: float = 1.5,
    sample_rate: int = 16_000,
    frequency: float = 220.0,
    amplitude: float = 0.2,
) -> bytes:
    frame_count = int(duration * sample_rate)
    t = np.arange(frame_count, dtype=np.float32) / sample_rate
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    pcm = np.clip(signal * 32767.0, -32768, 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


def feature_payload(*, energy: float = 0.1, pitch: float = 220.0, rate: float = 3.5) -> dict:
    return {
        "duration_seconds": 1.5,
        "pitch_hz": {"mean": pitch, "std": 5.0, "min": pitch - 3, "max": pitch + 3},
        "energy": {"mean": energy, "std": 0.01, "min": energy * 0.6, "max": min(1.0, energy * 1.4)},
        "spectral_centroid_hz": {"mean": 1500.0, "std": 100.0, "min": 1300.0, "max": 1700.0},
        "spectral_bandwidth_hz": {"mean": 1200.0, "std": 80.0, "min": 1000.0, "max": 1400.0},
        "spectral_rolloff_hz": {"mean": 2800.0, "std": 120.0, "min": 2500.0, "max": 3100.0},
        "zero_crossing_rate": {"mean": 0.05, "std": 0.01, "min": 0.01, "max": 0.09},
        "estimated_syllable_rate_sps": rate,
        "estimated_pause_ratio": 0.08,
    }


def test_baseline_requires_three_samples(tmp_path):
    service = PersonalBaselineService(tmp_path)
    try:
        service.create("default", [feature_payload(), feature_payload()])
    except PersonalBaselineError as exc:
        assert "3" in str(exc)
    else:
        raise AssertionError("Expected minimum-sample validation")


def test_baseline_persists_and_loads(tmp_path):
    service = PersonalBaselineService(tmp_path)
    profile = service.create("default", [feature_payload(), feature_payload(energy=0.105), feature_payload(pitch=221)])
    loaded = service.load("default")

    assert profile.sample_count == 3
    assert loaded.profile_id == "default"
    assert loaded.version == "personal-baseline-v1"
    assert len(loaded.metrics) >= 10
    assert (tmp_path / "default.json").exists()


def test_baseline_comparison_marks_typical_voice(tmp_path):
    service = PersonalBaselineService(tmp_path)
    samples = [
        feature_payload(energy=0.10, pitch=220.0, rate=3.5),
        feature_payload(energy=0.105, pitch=221.0, rate=3.6),
        feature_payload(energy=0.095, pitch=219.0, rate=3.4),
    ]
    service.create("speaker_1", samples)
    result = service.compare("speaker_1", feature_payload(energy=0.102, pitch=220.5, rate=3.5))

    assert result.normality == "typical"
    assert result.overall_deviation < 0.20
    assert result.metrics


def test_baseline_comparison_detects_large_deviation(tmp_path):
    service = PersonalBaselineService(tmp_path)
    samples = [
        feature_payload(energy=0.10, pitch=220.0, rate=3.5),
        feature_payload(energy=0.105, pitch=221.0, rate=3.6),
        feature_payload(energy=0.095, pitch=219.0, rate=3.4),
    ]
    service.create("speaker_1", samples)
    result = service.compare("speaker_1", feature_payload(energy=0.40, pitch=280.0, rate=7.5))

    assert result.normality == "strongly_deviant"
    assert result.overall_deviation > 0.45
    assert any(metric.significance == "strong" for metric in result.metrics)


def test_baseline_rejects_unsafe_profile_id(tmp_path):
    service = PersonalBaselineService(tmp_path)
    try:
        service.create("../escape", [feature_payload()] * 3)
    except PersonalBaselineError as exc:
        assert "Profile ID" in str(exc)
    else:
        raise AssertionError("Expected unsafe profile ID rejection")


def test_baseline_service_endpoint_status(tmp_path, monkeypatch):
    import app.routers.personal_baseline as router

    monkeypatch.setattr(router.settings, "PERSONAL_BASELINE_DIR", str(tmp_path))
    response = client.get("/api/v1/audio/baseline/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["min_samples"] == 3


def test_baseline_create_and_compare_endpoints(tmp_path, monkeypatch):
    import app.routers.personal_baseline as router

    monkeypatch.setattr(router.settings, "PERSONAL_BASELINE_DIR", str(tmp_path))
    raw = make_wav()
    response = client.post(
        "/api/v1/audio/baseline/test_speaker",
        files=[
            ("files", ("one.wav", raw, "audio/wav")),
            ("files", ("two.wav", raw, "audio/wav")),
            ("files", ("three.wav", raw, "audio/wav")),
        ],
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["profile"]["sample_count"] == 3

    compare = client.post(
        "/api/v1/audio/baseline/test_speaker/compare",
        files={"file": ("current.wav", raw, "audio/wav")},
    )
    assert compare.status_code == 200
    compared = compare.json()
    assert compared["profile_id"] == "test_speaker"
    assert compared["metrics"]
    assert compared["normality"] == "typical"


def test_baseline_missing_profile_returns_422(tmp_path, monkeypatch):
    import app.routers.personal_baseline as router

    monkeypatch.setattr(router.settings, "PERSONAL_BASELINE_DIR", str(tmp_path))
    response = client.post(
        "/api/v1/audio/baseline/missing/compare",
        files={"file": ("current.wav", make_wav(), "audio/wav")},
    )
    assert response.status_code == 422
    assert "not found" in response.json()["detail"].lower()


def test_insights_can_attach_personal_baseline(tmp_path, monkeypatch):
    import app.routers.insights as insights_router
    from app.fusion.calibration import FusionCalibrationArtifact
    from app.fusion.constants import CANONICAL_LABELS
    from app.fusion.engine import FusionEngine
    from types import SimpleNamespace

    def probability(label: str, high: float = 0.8) -> dict[str, float]:
        rest = (1.0 - high) / (len(CANONICAL_LABELS) - 1)
        return {emotion: (high if emotion == label else rest) for emotion in CANONICAL_LABELS}

    class FakeClassical:
        def predict(self, vector):
            return {
                "emotion": "angry", "confidence": 0.72, "probabilities": probability("angry", 0.72),
                "model_name": "fixture-classical", "feature_version": "acoustic-v1", "training_metadata": {},
            }

    class FakeDeep:
        def predict(self, samples, sample_rate):
            return SimpleNamespace(
                emotion="angry", confidence=0.82, probabilities=probability("angry", 0.82),
                chunks_analyzed=1, metadata=SimpleNamespace(model_id="fixture-deep"),
            )

    artifact = FusionCalibrationArtifact(
        version="phase8-fixture", labels=tuple(CANONICAL_LABELS), classical_temperature=1,
        deep_temperature=1, classical_weight=0.5, deep_weight=0.5, validation_metrics={}, metadata={}
    )
    monkeypatch.setattr(insights_router.settings, "PERSONAL_BASELINE_DIR", str(tmp_path))
    service = PersonalBaselineService(tmp_path)
    raw = make_wav()
    from app.audio.processor import AudioProcessor
    from app.features.acoustic import extract_acoustic_features
    processed = AudioProcessor().process(raw)
    actual_features = extract_acoustic_features(processed.samples, processed.sample_rate).to_dict()
    service.create("default", [actual_features, actual_features, actual_features])
    monkeypatch.setattr(insights_router, "get_engine", lambda: FusionEngine(artifact))
    monkeypatch.setattr(insights_router, "get_classical_model", lambda: FakeClassical())
    monkeypatch.setattr(insights_router, "get_deep_model", lambda: FakeDeep())

    response = client.post(
        "/api/v1/audio/emotion/insights",
        data={"profile_id": "default"},
        files={"file": ("speech.wav", raw, "audio/wav")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["insights"]["personal_baseline"]["profile_id"] == "default"
    assert payload["insights"]["personal_baseline"]["normality"] == "typical"
