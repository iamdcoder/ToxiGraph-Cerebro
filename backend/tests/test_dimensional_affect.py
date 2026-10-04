from types import SimpleNamespace

import numpy as np
import pytest

from app.dimensional_emotion.interpretation import interpret_affect
from app.dimensional_emotion.model import (
    DimensionalEmotionError,
    DimensionalSpeechEmotionModel,
    parse_dimensional_output,
)


def test_parse_named_dimensions_independent_of_model_order():
    values = np.array([0.2, 0.8, 0.6], dtype=np.float32)
    parsed = parse_dimensional_output(["valence", "dominance", "arousal"], values)
    assert parsed == pytest.approx({"valence": 0.2, "arousal": 0.6, "dominance": 0.8})


def test_parse_reference_arousal_dominance_valence_order():
    values = np.array([0.3, 0.7, 0.9], dtype=np.float32)
    parsed = parse_dimensional_output(["arousal", "dominance", "valence"], values)
    assert parsed == pytest.approx({"valence": 0.9, "arousal": 0.3, "dominance": 0.7})


def test_parse_rejects_wrong_output_length():
    with pytest.raises(DimensionalEmotionError):
        parse_dimensional_output(["valence", "arousal", "dominance"], np.array([0.2, 0.4]))


def test_parse_rejects_non_finite_values():
    with pytest.raises(DimensionalEmotionError):
        parse_dimensional_output(
            ["valence", "arousal", "dominance"],
            np.array([0.2, np.nan, 0.5], dtype=np.float32),
        )


def test_parse_rejects_out_of_range_values():
    with pytest.raises(DimensionalEmotionError):
        parse_dimensional_output(
            ["valence", "arousal", "dominance"],
            np.array([0.2, 1.4, 0.5], dtype=np.float32),
        )


def test_interpretation_positive_activated():
    result = interpret_affect(0.9, 0.85, 0.7)
    assert result.quadrant == "positive / activated"
    assert result.valence_label == "positive"
    assert result.arousal_label == "high"
    assert result.dominance_label == "high"


def test_interpretation_negative_calm():
    result = interpret_affect(0.15, 0.2, 0.3)
    assert result.quadrant == "negative / calm"
    assert result.summary.startswith("Speech affect is negative / calm")


def test_model_metadata_exposes_native_and_canonical_dimensions():
    fake_model = SimpleNamespace(
        config=SimpleNamespace(id2label={0: "arousal", 1: "dominance", 2: "valence"}, mean=0.0, std=1.0)
    )
    model = DimensionalSpeechEmotionModel(
        model_id="fake/dimensional",
        model=fake_model,
        processor=None,
        device="cpu",
    )
    metadata = model.metadata()
    assert metadata["dimensions"] == ["valence", "arousal", "dominance"]
    assert metadata["native_labels"] == ["arousal", "dominance", "valence"]
    assert metadata["output_range"] == "[0, 1]"


def test_dimensional_endpoint_uses_fake_model(monkeypatch):
    import io
    import wave
    from fastapi.testclient import TestClient
    from app.main import app
    from app.dimensional_emotion.model import DimensionalEmotionMetadata, DimensionalEmotionPrediction
    from app.routers import affect

    t = np.arange(48_000, dtype=np.float32) / 16_000.0
    signal = 0.08 * np.sin(2 * np.pi * 220.0 * t)
    pcm = np.clip(signal * 32767.0, -32768, 32767).astype('<i2')
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(pcm.tobytes())

    prediction = DimensionalEmotionPrediction(
        valence=0.82,
        arousal=0.74,
        dominance=0.61,
        metadata=DimensionalEmotionMetadata(
            model_id='fake/dimensional',
            dimensions=('valence', 'arousal', 'dominance'),
            sample_rate=16_000,
            device='cpu',
            output_range='[0, 1]',
        ),
    )

    class FakeModel:
        def predict(self, samples, sample_rate):
            return prediction

    monkeypatch.setattr(affect, 'get_model', lambda: FakeModel())
    monkeypatch.setattr(affect.settings, 'DIMENSIONAL_EMOTION_ENABLED', True)

    response = TestClient(app).post(
        '/api/v1/audio/emotion/affect',
        files={'file': ('sample.wav', buf.getvalue(), 'audio/wav')},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload['success'] is True
    assert payload['values']['valence'] == pytest.approx(0.82)
    assert payload['values']['arousal'] == pytest.approx(0.74)
    assert payload['interpretation']['quadrant'] == 'positive / activated'
