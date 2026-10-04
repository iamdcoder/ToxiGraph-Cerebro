from __future__ import annotations

import io
import types
import wave
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from app.fusion.calibration import (
    apply_temperature,
    canonical_probability_vector,
    evaluation_metrics,
    fit_temperature,
)
from app.fusion.constants import CANONICAL_LABELS
from app.fusion.engine import FusionEngine
from app.fusion.training import (
    fit_fusion_calibrator,
    save_artifact,
    load_artifact,
    evaluate_fusion_records,
)
from app.fusion.calibration import FusionCalibrationArtifact
from app.deep_emotion.model import DeepEmotionMetadata, DeepEmotionPrediction
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


def probability(label: str, high: float = 0.85) -> dict[str, float]:
    rest = (1.0 - high) / (len(CANONICAL_LABELS) - 1)
    return {emotion: (high if emotion == label else rest) for emotion in CANONICAL_LABELS}


def make_calibration_records() -> list[dict]:
    records = []
    for index in range(42):
        true = CANONICAL_LABELS[index % len(CANONICAL_LABELS)]
        classical_label = true
        deep_label = true
        if index % 11 == 0:
            deep_label = CANONICAL_LABELS[(CANONICAL_LABELS.index(true) + 1) % len(CANONICAL_LABELS)]
        records.append(
            {
                "id": f"sample-{index}",
                "label": true,
                "classical_probabilities": probability(classical_label, 0.72 + (index % 4) * 0.03),
                "deep_probabilities": probability(deep_label, 0.78 + (index % 3) * 0.04),
            }
        )
    return records


def test_canonical_probability_merges_calm_into_neutral():
    values = canonical_probability_vector({"calm": 0.4, "neutral": 0.2, "angry": 0.4})
    assert abs(sum(values) - 1.0) < 1e-12
    assert values[CANONICAL_LABELS.index("neutral")] == 0.6000000000000001


def test_temperature_scaling_preserves_probability_distribution():
    raw = probability("angry", 0.8)
    calibrated = apply_temperature(raw, 2.0)
    assert set(calibrated) == set(CANONICAL_LABELS)
    assert abs(sum(calibrated.values()) - 1.0) < 1e-12
    assert calibrated["angry"] < raw["angry"]


def test_temperature_fit_returns_positive_value_and_metrics_are_finite():
    records = make_calibration_records()
    from app.fusion.training import _extract_arrays
    classical, _, targets = _extract_arrays(records)
    temperature = fit_temperature(classical, targets)
    assert temperature > 0
    metrics = evaluation_metrics(classical, targets)
    assert all(np.isfinite(value) for value in metrics.values())


def test_fusion_calibrator_learns_positive_weights_and_artifact_roundtrip(tmp_path):
    artifact = fit_fusion_calibrator(make_calibration_records())
    assert artifact.version == "fusion-v1"
    assert artifact.classical_weight > 0
    assert artifact.deep_weight > 0
    assert abs(artifact.classical_weight + artifact.deep_weight - 1.0) < 1e-9
    assert artifact.classical_temperature > 0
    assert artifact.deep_temperature > 0
    assert "fused" in artifact.validation_metrics

    path = tmp_path / "fusion.joblib"
    save_artifact(artifact, path)
    loaded = load_artifact(path)
    assert loaded == artifact


def test_held_out_evaluation_uses_frozen_artifact():
    records = make_calibration_records()
    artifact = fit_fusion_calibrator(records)
    metrics = evaluate_fusion_records(records, artifact)
    assert metrics["records"] == len(records)
    assert set(metrics) >= {"classical_raw", "deep_raw", "classical_calibrated", "deep_calibrated", "fused"}
    assert all(0.0 <= metrics["fused"][key] for key in ("nll", "brier", "ece"))
    assert 0.0 <= metrics["fused"]["accuracy"] <= 1.0


def test_fusion_engine_reports_agreement_and_valid_probabilities():
    artifact = FusionCalibrationArtifact(
        version="fixture",
        labels=tuple(CANONICAL_LABELS),
        classical_temperature=1.5,
        deep_temperature=1.2,
        classical_weight=0.4,
        deep_weight=0.6,
        validation_metrics={},
        metadata={},
    )
    prediction = FusionEngine(artifact).combine(
        probability("angry", 0.82),
        probability("angry", 0.82),
    )
    assert prediction.emotion == "angry"
    assert prediction.agreement == "high"
    assert 0 <= prediction.js_divergence < 0.06
    assert abs(sum(prediction.probabilities.values()) - 1.0) < 1e-12


def test_fusion_engine_detects_model_disagreement():
    artifact = FusionCalibrationArtifact(
        version="fixture",
        labels=tuple(CANONICAL_LABELS),
        classical_temperature=1.0,
        deep_temperature=1.0,
        classical_weight=0.5,
        deep_weight=0.5,
        validation_metrics={},
        metadata={},
    )
    prediction = FusionEngine(artifact).combine(
        probability("angry", 0.92),
        probability("sad", 0.92),
    )
    assert prediction.agreement == "low"
    assert prediction.classical_prediction == "angry"
    assert prediction.deep_prediction == "sad"


def test_fusion_api_joins_classical_and_deep_predictions(monkeypatch, tmp_path):
    import app.routers.fusion as fusion_router

    artifact = FusionCalibrationArtifact(
        version="fixture",
        labels=tuple(CANONICAL_LABELS),
        classical_temperature=1.0,
        deep_temperature=1.0,
        classical_weight=0.4,
        deep_weight=0.6,
        validation_metrics={"fused": {"nll": 0.2, "brier": 0.1, "ece": 0.05, "accuracy": 0.9}},
        metadata={"fixture": True},
    )
    engine = FusionEngine(artifact)

    class FakeClassical:
        def predict(self, vector):
            return {
                "emotion": "angry",
                "confidence": 0.74,
                "probabilities": probability("angry", 0.74),
                "model_name": "fixture_classical",
                "feature_version": "acoustic-v1",
                "training_metadata": {"fixture": True},
            }

    class FakeDeep:
        def predict(self, samples, sample_rate):
            return DeepEmotionPrediction(
                emotion="angry",
                confidence=0.86,
                probabilities=probability("angry", 0.86),
                chunks_analyzed=1,
                metadata=DeepEmotionMetadata(
                    model_id="fixture_deep",
                    labels=tuple(CANONICAL_LABELS),
                    sample_rate=sample_rate,
                    device="cpu",
                ),
            )

    monkeypatch.setattr(fusion_router, "get_engine", lambda: engine)
    monkeypatch.setattr(fusion_router, "get_classical_model", lambda: FakeClassical())
    monkeypatch.setattr(fusion_router, "get_deep_model", lambda: FakeDeep())

    response = client.post(
        "/api/v1/audio/emotion/fusion",
        files={"file": ("speech.wav", make_wav(), "audio/wav")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["emotion"] == "angry"
    assert payload["agreement"] == "high"
    assert payload["chunks_analyzed"] == 1
    assert abs(sum(item["probability"] for item in payload["probabilities"]) - 1.0) < 1e-9
    assert payload["classical"]["weight"] == 0.4
    assert payload["deep"]["weight"] == 0.6


def test_fusion_api_reports_503_without_calibration(monkeypatch):
    import app.routers.fusion as fusion_router
    from app.fusion.training import FusionTrainingError

    monkeypatch.setattr(
        fusion_router,
        "get_engine",
        lambda: (_ for _ in ()).throw(FusionTrainingError("Fusion calibration is not trained yet.")),
    )
    response = client.post(
        "/api/v1/audio/emotion/fusion",
        files={"file": ("speech.wav", make_wav(), "audio/wav")},
    )
    assert response.status_code == 503
    assert "not trained" in response.json()["detail"]
