from __future__ import annotations

import io
import wave

import numpy as np
from fastapi.testclient import TestClient

from app.features.acoustic import AcousticConfig, extract_acoustic_features
from app.main import app


def make_wav(samples: np.ndarray, sample_rate: int = 16_000) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


def test_silence_features_are_finite_and_quiet():
    samples = np.zeros(16_000, dtype=np.float32)
    features = extract_acoustic_features(samples, 16_000)

    payload = features.to_dict()
    assert payload["frame_count"] > 0
    assert payload["voiced_frame_count"] == 0
    assert payload["silence_ratio"] == 1.0
    assert payload["pitch_hz"]["mean"] == 0.0
    assert all(np.isfinite(value) for value in payload["mfcc_mean"])


def test_tone_has_measurable_pitch_and_energy():
    sample_rate = 16_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    samples = 0.35 * np.sin(2 * np.pi * 220.0 * time)

    features = extract_acoustic_features(samples, sample_rate)
    payload = features.to_dict()

    assert payload["energy"]["mean"] > 0.1
    assert payload["silence_ratio"] < 0.1
    assert payload["estimated_pause_count"] == 0
    assert payload["pitch_hz"]["mean"] > 0
    assert abs(payload["pitch_hz"]["mean"] - 220.0) < 20.0
    assert payload["spectral_centroid_hz"]["mean"] > 0
    assert len(payload["mfcc_mean"]) == 13
    assert len(payload["mfcc_std"]) == 13
    assert len(payload["chroma"]) == 12
    assert payload["model_vector_length"] == 77
    assert features.to_vector().shape == (77,)


def test_feature_extraction_is_deterministic():
    sample_rate = 16_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    samples = 0.2 * np.sin(2 * np.pi * 180.0 * time) + 0.05 * np.sin(2 * np.pi * 360.0 * time)

    first = extract_acoustic_features(samples, sample_rate).to_dict()
    second = extract_acoustic_features(samples, sample_rate).to_dict()

    assert first == second


def test_short_frames_are_padded_without_crashing():
    samples = np.array([0.1, -0.1, 0.05], dtype=np.float32)
    features = extract_acoustic_features(
        samples,
        16_000,
        AcousticConfig(min_voiced_rms=0.0),
    )
    assert features.frame_count == 1
    assert len(features.mfcc_mean) == 13


def test_api_extracts_features_from_phase1_wav():
    sample_rate = 16_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    samples = 0.25 * np.sin(2 * np.pi * 200.0 * time)
    audio = make_wav(samples, sample_rate)

    client = TestClient(app)
    response = client.post(
        "/api/v1/audio/features",
        files={"file": ("sample.wav", audio, "audio/wav")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["audio"]["sample_rate"] == 16_000
    assert payload["audio"]["channels"] == 1
    assert payload["features"]["version"] == "acoustic-v1"
    assert payload["features"]["frame_count"] > 0
    assert payload["features"]["model_vector_length"] == 77
    assert len(payload["features"]["spectral_contrast_db"]) == 6
    assert len(payload["features"]["spectral_contrast_db"]) == 6


def test_api_rejects_non_wav_content_type():
    client = TestClient(app)
    response = client.post(
        "/api/v1/audio/features",
        files={"file": ("sample.mp3", b"not audio", "audio/mpeg")},
    )
    assert response.status_code == 415


def test_temporal_voice_activity_detects_internal_pauses():
    sample_rate = 16_000
    time = np.arange(sample_rate // 2, dtype=np.float32) / sample_rate
    tone = 0.25 * np.sin(2 * np.pi * 220.0 * time)
    samples = np.concatenate([tone, np.zeros_like(tone), tone, np.zeros_like(tone)])

    features = extract_acoustic_features(samples, sample_rate)

    assert 0.35 < features.silence_ratio < 0.65
    assert features.estimated_pause_count == 2
    assert features.voiced_frame_count < features.frame_count
    assert features.pitch_hz["mean"] > 0


def test_non_finite_samples_are_safely_sanitized():
    samples = np.array([0.2, np.nan, np.inf, -np.inf, -0.2] * 4000, dtype=np.float32)
    features = extract_acoustic_features(samples, 16_000)

    vector = features.to_vector()
    assert np.all(np.isfinite(vector))
    assert np.all(np.isfinite(np.asarray(features.mfcc_mean)))
