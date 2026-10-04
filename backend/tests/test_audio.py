from __future__ import annotations

import io
import wave

import numpy as np
from fastapi.testclient import TestClient

from app.audio.processor import AudioProcessingError, AudioProcessor
from app.main import app


client = TestClient(app)


def make_wav(
    *,
    duration: float = 1.0,
    sample_rate: int = 16_000,
    channels: int = 1,
    frequency: float = 220.0,
    amplitude: float = 0.2,
) -> bytes:
    frame_count = int(duration * sample_rate)
    t = np.arange(frame_count, dtype=np.float32) / sample_rate
    mono = amplitude * np.sin(2 * np.pi * frequency * t)
    pcm = np.clip(mono * 32767.0, -32768, 32767).astype("<i2")

    if channels == 2:
        pcm = np.repeat(pcm[:, None], 2, axis=1).reshape(-1)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())

    return buffer.getvalue()


def test_processor_normalizes_to_mono_16khz() -> None:
    raw = make_wav(duration=1.25, sample_rate=48_000, channels=2)
    processed = AudioProcessor().process(raw)

    assert processed.sample_rate == 16_000
    assert processed.channels == 1
    assert processed.original_sample_rate == 48_000
    assert processed.original_channels == 2
    assert processed.samples.dtype == np.float32
    assert 1.2 < processed.duration_seconds < 1.3
    assert 0 < processed.rms < 1
    assert 0 < processed.peak <= 1
    assert processed.is_silent is False


def test_processor_rejects_empty_audio() -> None:
    try:
        AudioProcessor().process(b"")
    except AudioProcessingError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError("Expected empty-audio rejection")


def test_processor_rejects_short_audio() -> None:
    raw = make_wav(duration=0.1)
    try:
        AudioProcessor().process(raw)
    except AudioProcessingError as exc:
        assert "too short" in str(exc).lower()
    else:
        raise AssertionError("Expected short-audio rejection")


def test_processor_flags_silence_without_failing() -> None:
    raw = make_wav(amplitude=0.0)
    processed = AudioProcessor().process(raw)
    assert processed.is_silent is True
    assert processed.rms == 0.0


def test_validate_endpoint_accepts_pcm_wav() -> None:
    raw = make_wav(duration=1.0, sample_rate=48_000, channels=2)
    response = client.post(
        "/api/v1/audio/validate",
        files={
            "file": (
                "voice.wav",
                raw,
                "audio/wav",
            )
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["audio"]["sample_rate"] == 16_000
    assert payload["audio"]["channels"] == 1
    assert payload["audio"]["original_sample_rate"] == 48_000
    assert payload["audio"]["original_channels"] == 2


def test_validate_endpoint_rejects_unsupported_type() -> None:
    response = client.post(
        "/api/v1/audio/validate",
        files={
            "file": (
                "voice.mp3",
                b"not-an-mp3",
                "audio/mpeg",
            )
        },
    )

    assert response.status_code == 415


def test_validate_endpoint_rejects_invalid_wav() -> None:
    response = client.post(
        "/api/v1/audio/validate",
        files={
            "file": (
                "voice.wav",
                b"this-is-not-a-wav",
                "audio/wav",
            )
        },
    )

    assert response.status_code == 422
