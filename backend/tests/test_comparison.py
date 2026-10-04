from __future__ import annotations

import io
import wave
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from app.comparison.engine import compare_insights
from app.fusion.constants import CANONICAL_LABELS
from app.fusion.engine import FusionEngine
from app.fusion.calibration import FusionCalibrationArtifact
from app.main import app
from app.models.audio import AudioMetadata
from app.models.features import AcousticFeatureSet, EmotionProbability, FusionEmotionResponse, FusionModelContribution
from app.models.insights import AudioEmotionInsightsResponse, EmotionInsights, InsightEvidence, UncertaintyAnalysis
from app.transcription.service import TranscriptionResult


client = TestClient(app)


def probability(label: str, high: float = 0.82) -> dict[str, float]:
    rest = (1.0 - high) / (len(CANONICAL_LABELS) - 1)
    return {emotion: (high if emotion == label else rest) for emotion in CANONICAL_LABELS}


def make_wav(*, duration: float = 1.6, frequency: float = 220.0, amplitude: float = 0.12) -> bytes:
    count = int(duration * 16_000)
    t = np.arange(count, dtype=np.float32) / 16_000
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    pcm = np.clip(signal * 32767.0, -32768, 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


def fake_insight(emotion: str, *, pitch: float, energy: float, confidence: float):
    features = SimpleNamespace(
        pitch_hz={"mean": pitch, "std": 8.0, "min": pitch - 5, "max": pitch + 5},
        energy={"mean": energy, "std": energy * 0.15, "min": energy * 0.7, "max": min(1.0, energy * 1.3)},
        spectral_centroid_hz={"mean": 1700.0, "std": 80.0},
        spectral_bandwidth_hz={"mean": 1200.0, "std": 50.0},
        spectral_rolloff_hz={"mean": 3000.0, "std": 120.0},
        zero_crossing_rate={"mean": 0.08, "std": 0.01},
        estimated_syllable_rate_sps=3.2,
        estimated_pause_ratio=0.08,
        silence_ratio=0.08,
        mfcc_mean=[float(index + 1) for index in range(13)],
    )
    probabilities = probability(emotion, confidence)
    contribution = FusionModelContribution(
        model_type="fixture",
        model_name="fixture",
        emotion=emotion,
        confidence=confidence,
        calibrated_confidence=confidence,
        weight=0.5,
        temperature=1.0,
        probabilities=[EmotionProbability(emotion=label, probability=value) for label, value in probabilities.items()],
    )
    fusion = FusionEmotionResponse(
        success=True,
        message="fixture",
        emotion=emotion,
        confidence=confidence,
        probabilities=[EmotionProbability(emotion=label, probability=value) for label, value in probabilities.items()],
        fusion_strategy="fixture",
        agreement="high",
        js_divergence=0.01,
        classical=contribution,
        deep=contribution,
        calibration_version="fixture",
        sample_rate=16000,
        chunks_analyzed=1,
    )
    acoustic = AcousticFeatureSet(
        version="acoustic-v1",
        sample_rate=16000,
        duration_seconds=1.6,
        frame_count=160,
        frame_length_seconds=0.025,
        hop_length_seconds=0.01,
        voiced_frame_count=147,
        silence_ratio=0.08,
        speech_duration_seconds=1.47,
        estimated_syllable_rate_sps=3.2,
        estimated_pause_count=2,
        estimated_pause_ratio=0.08,
        pitch_hz={"mean": pitch, "std": 8.0, "min": pitch - 5, "max": pitch + 5},
        energy={"mean": energy, "std": energy * 0.15, "min": energy * 0.7, "max": min(1.0, energy * 1.3)},
        spectral_centroid_hz={"mean": 1700.0, "std": 80.0, "min": 1600.0, "max": 1800.0},
        spectral_bandwidth_hz={"mean": 1200.0, "std": 50.0, "min": 1120.0, "max": 1280.0},
        spectral_rolloff_hz={"mean": 3000.0, "std": 120.0, "min": 2800.0, "max": 3200.0},
        spectral_flatness={"mean": 0.12, "std": 0.01, "min": 0.09, "max": 0.15},
        spectral_contrast_db=[1.0] * 6,
        zero_crossing_rate={"mean": 0.08, "std": 0.01, "min": 0.06, "max": 0.10},
        chroma=[1 / 12] * 12,
        mfcc_mean=[float(index + 1) for index in range(13)],
        mfcc_std=[0.2] * 13,
        model_vector_length=77,
    )
    response = AudioEmotionInsightsResponse(
        success=True,
        message="fixture",
        audio=AudioMetadata(
            format="wav", encoding="pcm", sample_rate=16000, channels=1,
            original_sample_rate=16000, original_channels=1, original_sample_width_bytes=2,
            duration_seconds=1.6, rms=energy, peak=min(1.0, energy * 1.8), is_silent=False,
        ),
        features=acoustic,
        fusion=fusion,
        insights=EmotionInsights(
            summary="fixture",
            evidence=[InsightEvidence(category="model", title="Fixture", detail="Fixture", strength="strong")],
            uncertainty=UncertaintyAnalysis(
                uncertainty_index=0.1, level="low", normalized_entropy=0.1, top_probability=confidence,
                top_margin=confidence - 0.02, model_disagreement=0.01, quality_score=0.95,
                factors=[{"name": "fixture", "value": 0.01, "interpretation": "Fixture"}], recommendation="Fixture.",
            ),
            caveat="Fixture response.",
        ),
    )
    return response


def test_comparison_engine_reports_feature_and_distribution_shift():
    a = fake_insight("neutral", pitch=180.0, energy=0.08, confidence=0.75)
    b = fake_insight("angry", pitch=300.0, energy=0.30, confidence=0.86)
    result = compare_insights(a, b)

    assert result.emotion_changed is True
    assert 0.0 <= result.acoustic_change_score <= 1.0
    assert 0.0 <= result.acoustic_similarity_score <= 1.0
    assert 0.0 <= result.emotion_distribution_shift <= 1.0
    assert result.acoustic_features
    assert result.emotion_probabilities[0].absolute_delta >= 0.0
    assert "changed" in result.interpretation.lower()


def test_comparison_engine_marks_stable_recordings():
    a = fake_insight("happy", pitch=220.0, energy=0.12, confidence=0.82)
    b = fake_insight("happy", pitch=221.0, energy=0.121, confidence=0.82)
    result = compare_insights(a, b)

    assert result.emotion_changed is False
    assert result.acoustic_change_score < 0.20
    assert result.emotion_distribution_shift < 0.03


def test_comparison_engine_normalizes_alias_labels():
    a = fake_insight("neutral", pitch=220.0, energy=0.11, confidence=0.82)
    b = fake_insight("surprised", pitch=250.0, energy=0.22, confidence=0.82)
    result = compare_insights(a, b)
    assert {item.emotion for item in result.emotion_probabilities} == set(CANONICAL_LABELS)


def test_comparison_endpoint_runs_pipeline_twice_and_returns_both_results(monkeypatch):
    import app.routers.comparison as comparison_router

    a_bytes = make_wav(frequency=220.0, amplitude=0.11)
    b_bytes = make_wav(frequency=300.0, amplitude=0.26)

    a = fake_insight("neutral", pitch=220.0, energy=0.07, confidence=0.80)
    b = fake_insight("angry", pitch=300.0, energy=0.18, confidence=0.88)
    calls = iter([a, b])
    monkeypatch.setattr(comparison_router, "_run_insights", lambda payload, profile_id=None: next(calls))

    response = client.post(
        "/api/v1/audio/emotion/compare",
        files={
            "recording_a": ("a.wav", a_bytes, "audio/wav"),
            "recording_b": ("b.wav", b_bytes, "audio/wav"),
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["recording_a"]["fusion"]["emotion"] == "neutral"
    assert payload["recording_b"]["fusion"]["emotion"] == "angry"
    assert payload["comparison"]["emotion_changed"] is True
    assert payload["acoustic_features"]
    assert payload["emotion_probabilities"]
    assert payload["content_verification"]["status"] == "unavailable"
    assert "normalized ASR word sequences" in payload["protocol_note"]


def test_comparison_endpoint_rejects_non_wav(monkeypatch):
    import app.routers.comparison as comparison_router

    monkeypatch.setattr(comparison_router, "_run_insights", lambda *args, **kwargs: None)
    raw = b"not wav"
    response = client.post(
        "/api/v1/audio/emotion/compare",
        files={
            "recording_a": ("a.txt", raw, "text/plain"),
            "recording_b": ("b.wav", make_wav(), "audio/wav"),
        },
    )
    assert response.status_code == 415



def test_comparison_endpoint_verifies_same_sentence_when_asr_is_available(monkeypatch):
    import app.routers.comparison as comparison_router

    a_bytes = make_wav(frequency=220.0, amplitude=0.11)
    b_bytes = make_wav(frequency=300.0, amplitude=0.26)
    a = fake_insight("neutral", pitch=220.0, energy=0.07, confidence=0.80)
    b = fake_insight("angry", pitch=300.0, energy=0.18, confidence=0.88)
    insight_calls = iter([a, b])
    monkeypatch.setattr(comparison_router, "_run_insights", lambda payload, profile_id=None: next(insight_calls))

    class FakeTranscriber:
        def transcribe(self, samples, sample_rate):
            text = "I am completely fine" if float(np.max(np.abs(samples))) < 0.20 else "I am completely fine"
            return TranscriptionResult(
                text=text,
                model_id="fixture-whisper",
                language="en",
                duration_seconds=float(len(samples) / sample_rate),
            )

    monkeypatch.setattr(comparison_router, "get_transcriber", lambda: FakeTranscriber())
    response = client.post(
        "/api/v1/audio/emotion/compare",
        files={
            "recording_a": ("a.wav", a_bytes, "audio/wav"),
            "recording_b": ("b.wav", b_bytes, "audio/wav"),
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["content_verification"]["status"] == "same"
    assert payload["content_verification"]["similarity"] == 1.0
    assert payload["content_verification"]["asr_available"] is True
    assert payload["content_verification"]["asr_model_id"] == "fixture-whisper"

def test_comparison_status():
    response = client.get("/api/v1/audio/emotion/compare/status")
    assert response.status_code == 200
    assert response.json()["service"] == "recording_comparison"
