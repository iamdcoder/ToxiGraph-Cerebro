import io
import wave

import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.quality.engine import analyze_audio_quality

client = TestClient(app)


def make_wav(signal: np.ndarray, sample_rate: int = 16_000) -> bytes:
    pcm = np.clip(signal * 32767.0, -32768, 32767).astype('<i2')
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


def tone(duration: float = 3.0, amplitude: float = 0.08, frequency: float = 220.0) -> np.ndarray:
    t = np.arange(int(duration * 16_000), dtype=np.float32) / 16_000
    return amplitude * np.sin(2 * np.pi * frequency * t)


def test_clean_audio_gets_usable_quality_score():
    result = analyze_audio_quality(tone(), 16_000)
    assert 0.0 <= result.quality_score <= 1.0
    assert result.verdict in {'good', 'acceptable', 'poor', 'unreliable'}
    assert result.speech_ratio > 0.5
    assert result.clipping_ratio == 0.0


def test_silence_is_flagged_as_poor_quality():
    result = analyze_audio_quality(np.zeros(48_000, dtype=np.float32), 16_000)
    assert result.verdict in {'poor', 'unreliable'}
    assert 'excessive_silence' in result.issues


def test_clipped_audio_is_flagged():
    signal = tone(amplitude=1.0)
    result = analyze_audio_quality(signal, 16_000)
    assert result.clipping_ratio > 0.01
    assert 'clipping' in result.issues


def test_quiet_audio_is_flagged():
    result = analyze_audio_quality(tone(amplitude=0.002), 16_000)
    assert 'low_recording_level' in result.issues


def test_noisy_audio_gets_a_noise_estimate():
    rng = np.random.default_rng(42)
    signal = tone(amplitude=0.10) + rng.normal(0.0, 0.006, 48_000).astype(np.float32)
    result = analyze_audio_quality(signal, 16_000)
    assert result.estimated_snr_db is not None
    assert result.noise_floor_rms is not None


def test_quality_endpoint_returns_structured_report():
    response = client.post(
        '/api/v1/audio/quality',
        files={'file': ('sample.wav', make_wav(tone()), 'audio/wav')},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload['success'] is True
    assert 'quality_score' in payload['quality']
    assert 'issues' in payload['quality']


def test_quality_endpoint_status():
    response = client.get('/api/v1/audio/quality/status')
    assert response.status_code == 200
    assert response.json()['service'] == 'audio_quality'
