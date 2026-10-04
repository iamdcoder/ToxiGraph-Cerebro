from __future__ import annotations

import io
import sys
import types
import wave
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.deep_emotion.model import (
    DeepEmotionModelError,
    DeepSpeechEmotionModel,
    chunk_audio,
    normalize_label,
    normalize_probabilities,
)
from app.deep_emotion.service import clear_model_cache
from app.main import app


client = TestClient(app)


def make_wav(duration: float = 1.0, sample_rate: int = 16_000, frequency: float = 240.0) -> bytes:
    count = int(duration * sample_rate)
    t = np.arange(count, dtype=np.float32) / sample_rate
    pcm = (0.2 * np.sin(2 * np.pi * frequency * t) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


def test_label_normalization_handles_common_aliases():
    assert normalize_label("Anger") == "angry"
    assert normalize_label("Happiness") == "happy"
    assert normalize_label("Neutral") == "neutral"
    assert normalize_label("surprise") == "surprised"


def test_probability_normalization_merges_aliases():
    result = normalize_probabilities(
        ["Anger", "Angry", "Neutral"],
        np.array([0.2, 0.3, 0.5], dtype=np.float32),
    )
    assert set(result) == {"angry", "neutral"}
    assert abs(result["angry"] - 0.5) < 1e-6
    assert abs(sum(result.values()) - 1.0) < 1e-9


def test_probability_normalization_rejects_invalid_scores():
    with pytest.raises(DeepEmotionModelError):
        normalize_probabilities(["angry", "sad"], np.array([0.5, np.nan]))


def test_chunk_audio_keeps_short_input_whole():
    samples = np.zeros(3_200, dtype=np.float32)
    chunks = chunk_audio(samples, 16_000, max_seconds=10, chunk_seconds=8, overlap_seconds=1)
    assert len(chunks) == 1
    assert len(chunks[0]) == 3_200


def test_chunk_audio_splits_long_input_with_overlap():
    samples = np.zeros(20 * 16_000, dtype=np.float32)
    chunks = chunk_audio(samples, 16_000, max_seconds=10, chunk_seconds=8, overlap_seconds=1)
    assert len(chunks) == 3
    assert len(chunks[0]) == 8 * 16_000
    assert len(chunks[1]) == 8 * 16_000
    assert len(chunks[2]) == 6 * 16_000


def test_chunk_audio_rejects_invalid_configuration():
    with pytest.raises(DeepEmotionModelError):
        chunk_audio(np.zeros(10, dtype=np.float32), 16_000, chunk_seconds=10, overlap_seconds=10)


def _install_fake_transformers(monkeypatch):
    import torch

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_id):
            return cls()

        def __call__(self, audio, sampling_rate, return_tensors, padding):
            return {"input_values": torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)}

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(
                id2label={
                    0: "angry",
                    1: "neutral",
                    2: "sad",
                }
            )
            self._device = "cpu"

        @classmethod
        def from_pretrained(cls, model_id):
            return cls()

        def to(self, device):
            self._device = str(device)
            return self

        def eval(self):
            return self

        def __call__(self, **inputs):
            batch = inputs["input_values"]
            count = batch.shape[0]
            logits = torch.tensor([[2.0, 1.0, 0.0]] * count, dtype=torch.float32)
            return SimpleNamespace(logits=logits)

    fake = types.SimpleNamespace(
        AutoProcessor=FakeProcessor,
        AutoModelForAudioClassification=FakeModel,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake)
    return fake


def test_deep_model_load_and_predict_without_network(monkeypatch):
    _install_fake_transformers(monkeypatch)
    model = DeepSpeechEmotionModel.load(
        "fixture/deep-emotion",
        device="cpu",
        sample_rate=16_000,
        max_seconds=10,
        chunk_seconds=8,
        overlap_seconds=1,
    )
    prediction = model.predict(np.ones(16_000, dtype=np.float32) * 0.1, 16_000)
    assert prediction.emotion == "angry"
    assert prediction.chunks_analyzed == 1
    assert abs(sum(prediction.probabilities.values()) - 1.0) < 1e-9
    assert prediction.metadata.model_id == "fixture/deep-emotion"


def test_deep_model_aggregates_multiple_chunks(monkeypatch):
    _install_fake_transformers(monkeypatch)
    model = DeepSpeechEmotionModel.load(
        "fixture/deep-emotion",
        device="cpu",
        sample_rate=16_000,
        max_seconds=10,
        chunk_seconds=8,
        overlap_seconds=1,
    )
    prediction = model.predict(np.ones(20 * 16_000, dtype=np.float32) * 0.1, 16_000)
    assert prediction.chunks_analyzed == 3
    assert prediction.emotion == "angry"


def test_deep_emotion_status_reports_unavailable_without_transformers(monkeypatch):
    clear_model_cache()
    monkeypatch.setitem(sys.modules, "transformers", None)
    response = client.get("/api/v1/audio/emotion/deep/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is False
    clear_model_cache()


def test_deep_emotion_api_returns_503_when_model_is_unavailable(monkeypatch):
    clear_model_cache()
    monkeypatch.setitem(sys.modules, "transformers", None)
    response = client.post(
        "/api/v1/audio/emotion/deep",
        files={"file": ("speech.wav", make_wav(), "audio/wav")},
    )
    assert response.status_code == 503
    assert "PyTorch and Transformers" in response.json()["detail"]
    clear_model_cache()


def test_deep_emotion_api_with_fake_model(monkeypatch):
    _install_fake_transformers(monkeypatch)
    clear_model_cache()
    response = client.post(
        "/api/v1/audio/emotion/deep",
        files={"file": ("speech.wav", make_wav(), "audio/wav")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["emotion"] == "angry"
    assert payload["chunks_analyzed"] == 1
    assert payload["sample_rate"] == 16_000
    assert abs(sum(item["probability"] for item in payload["probabilities"]) - 1.0) < 1e-9
    clear_model_cache()
