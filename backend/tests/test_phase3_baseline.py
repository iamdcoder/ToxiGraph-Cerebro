from __future__ import annotations

import io
import json
import wave
from pathlib import Path

import joblib
import numpy as np
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from app.datasets.ravdess import RAVDESS_EMOTIONS, parse_ravdess_filename
from app.emotion_baseline.model import BaselineBundle, ClassicalEmotionModel
from app.emotion_baseline.training import speaker_independent_split, train_baseline
from app.features.acoustic import FEATURE_VECTOR_NAMES
from app.config import settings
from app.main import app


def sample_path(name: str) -> Path:
    return Path("data") / name


def _write_tone_wav(path: Path, frequency: float, amplitude: float = 0.2, sample_rate: int = 16_000) -> None:
    duration = 0.75
    time = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
    samples = amplitude * np.sin(2 * np.pi * frequency * time)
    pcm = np.clip(samples, -1.0, 1.0)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes((pcm * 32767).astype(np.int16).tobytes())
    path.write_bytes(buffer.getvalue())


def test_ravdess_parser_reads_speech_filename():
    sample = parse_ravdess_filename(sample_path("03-01-05-02-02-01-12.wav"))
    assert sample.speaker_id == "actor_12"
    assert sample.emotion == "angry"
    assert sample.emotion_code == 5
    assert sample.intensity_code == 2


def test_ravdess_parser_rejects_non_speech_modality():
    try:
        parse_ravdess_filename(sample_path("01-01-05-01-01-01-01.wav"))
    except ValueError as exc:
        assert "audio-only" in str(exc)
    else:
        raise AssertionError("Expected invalid modality to be rejected")


def test_ravdess_emotion_mapping_is_complete():
    assert len(RAVDESS_EMOTIONS) == 8
    assert RAVDESS_EMOTIONS[1] == "neutral"
    assert RAVDESS_EMOTIONS[8] == "surprised"


def test_speaker_split_has_no_leakage():
    samples = []
    for speaker_index in range(1, 13):
        speaker = f"actor_{speaker_index:02d}"
        for emotion in ("neutral", "happy", "sad", "angry"):
            from app.datasets.ravdess import RAVDESSSample
            samples.append(RAVDESSSample(
                path=sample_path(f"{speaker}-{emotion}.wav"),
                speaker_id=speaker,
                emotion=emotion,
                emotion_code=1,
                intensity_code=1,
                statement_code=1,
                repetition_code=1,
            ))

    split = speaker_independent_split(samples)
    train_speakers = {item.speaker_id for item in split.train}
    val_speakers = {item.speaker_id for item in split.validation}
    test_speakers = {item.speaker_id for item in split.test}

    assert train_speakers.isdisjoint(val_speakers)
    assert train_speakers.isdisjoint(test_speakers)
    assert val_speakers.isdisjoint(test_speakers)
    assert split.train and split.validation and split.test


def test_classical_model_emits_probability_distribution():
    rng = np.random.default_rng(42)
    X = np.vstack([
        rng.normal(loc=-2.0, scale=0.2, size=(12, len(FEATURE_VECTOR_NAMES))),
        rng.normal(loc=0.0, scale=0.2, size=(12, len(FEATURE_VECTOR_NAMES))),
        rng.normal(loc=2.0, scale=0.2, size=(12, len(FEATURE_VECTOR_NAMES))),
    ])
    y = np.repeat(np.arange(3), 12)
    model = LogisticRegression(max_iter=1000).fit(X, y)
    bundle = BaselineBundle(
        model_name="fixture",
        model=model,
        labels=("angry", "neutral", "happy"),
        feature_names=tuple(FEATURE_VECTOR_NAMES),
        feature_version="acoustic-v1",
        training_metadata={"fixture": True},
    )
    predictor = ClassicalEmotionModel(bundle)
    result = predictor.predict(X[0])

    assert result["emotion"] in bundle.labels
    assert 0 <= result["confidence"] <= 1
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-9
    assert result["feature_version"] == "acoustic-v1"


def test_model_rejects_wrong_feature_length():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(10, len(FEATURE_VECTOR_NAMES)))
    y = np.arange(10) % 2
    model = LogisticRegression(max_iter=1000).fit(X, y)
    bundle = BaselineBundle(
        model_name="fixture",
        model=model,
        labels=("a", "b"),
        feature_names=tuple(FEATURE_VECTOR_NAMES),
        feature_version="acoustic-v1",
        training_metadata={},
    )
    predictor = ClassicalEmotionModel(bundle)
    try:
        predictor.predict(np.zeros(10))
    except Exception as exc:
        assert "Expected 77" in str(exc)
    else:
        raise AssertionError("Wrong feature length should be rejected")


def test_training_smoke_builds_model_artifact(tmp_path):
    frequencies = {
        "neutral": 180.0,
        "calm": 220.0,
        "happy": 280.0,
        "sad": 320.0,
        "angry": 390.0,
        "fearful": 460.0,
        "disgust": 540.0,
        "surprised": 620.0,
    }
    data_dir = tmp_path / "ravdess"
    for actor in range(1, 7):
        actor_dir = data_dir / f"Actor_{actor:02d}"
        actor_dir.mkdir(parents=True)
        for code, (emotion, frequency) in enumerate(frequencies.items(), start=1):
            # RAVDESS names use numeric emotion codes, independent of this fixture's order.
            name = {
                "neutral": 1, "calm": 2, "happy": 3, "sad": 4,
                "angry": 5, "fearful": 6, "disgust": 7, "surprised": 8,
            }[emotion]
            wav_path = actor_dir / f"03-01-{name:02d}-01-01-01-{actor:02d}.wav"
            _write_tone_wav(wav_path, frequency + actor * 3.0, amplitude=0.16 + actor * 0.005)

    model_path = tmp_path / "models" / "baseline.joblib"
    artifact_dir = tmp_path / "artifacts"
    result = train_baseline(data_dir, model_path, artifact_dir, seed=42)

    assert result["success"] is True
    assert model_path.exists()
    assert result["feature_count"] == len(FEATURE_VECTOR_NAMES)
    assert set(result["candidate_validation"]) == {
        "logistic_regression", "linear_svm", "random_forest"
    }
    assert 0.0 <= result["test"]["macro_f1"] <= 1.0
    assert (artifact_dir / "evaluation.json").exists()
    assert (artifact_dir / "training_metadata.json").exists()
    assert (artifact_dir / "manifest.csv").exists()


def test_emotion_inference_api_uses_saved_model(tmp_path, monkeypatch):
    rng = np.random.default_rng(42)
    X = np.vstack([
        rng.normal(loc=-1.5, scale=0.15, size=(30, len(FEATURE_VECTOR_NAMES))),
        rng.normal(loc=1.5, scale=0.15, size=(30, len(FEATURE_VECTOR_NAMES))),
    ])
    y = np.repeat(np.arange(2), 30)
    classifier = LogisticRegression(max_iter=1000).fit(X, y)
    model_path = tmp_path / "baseline.joblib"
    joblib.dump({
        "model_name": "fixture",
        "model": classifier,
        "labels": ["angry", "neutral"],
        "feature_names": FEATURE_VECTOR_NAMES,
        "feature_version": "acoustic-v1",
        "training_metadata": {"fixture": True},
    }, model_path)

    wav_path = tmp_path / "speech.wav"
    _write_tone_wav(wav_path, 240.0)
    monkeypatch.setattr(settings, "CLASSICAL_MODEL_PATH", str(model_path))

    client = TestClient(app)
    status = client.get("/api/v1/audio/emotion/status")
    assert status.status_code == 200
    assert status.json()["available"] is True

    response = client.post(
        "/api/v1/audio/emotion",
        files={"file": ("speech.wav", wav_path.read_bytes(), "audio/wav")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["emotion"] in {"angry", "neutral"}
    assert 0.0 <= payload["confidence"] <= 1.0
    assert abs(sum(item["probability"] for item in payload["probabilities"]) - 1.0) < 1e-6
    assert payload["feature_count"] == 77
