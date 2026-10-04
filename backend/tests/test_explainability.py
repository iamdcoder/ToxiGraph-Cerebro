from __future__ import annotations

import io
import wave
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from app.explainability.engine import analyze_uncertainty, build_insights
from app.fusion.calibration import FusionCalibrationArtifact
from app.fusion.constants import CANONICAL_LABELS
from app.fusion.engine import FusionEngine
from app.main import app


client = TestClient(app)


def probability(label: str, high: float = 0.8) -> dict[str, float]:
    rest = (1.0 - high) / (len(CANONICAL_LABELS) - 1)
    return {emotion: (high if emotion == label else rest) for emotion in CANONICAL_LABELS}


def make_wav(duration: float = 2.0, sample_rate: int = 16_000) -> bytes:
    count = int(duration * sample_rate)
    t = np.arange(count, dtype=np.float32) / sample_rate
    signal = 0.2 * np.sin(2 * np.pi * 220 * t)
    pcm = (signal * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


class FakeClassical:
    def predict(self, vector):
        return {
            "emotion": "angry",
            "confidence": 0.72,
            "probabilities": probability("angry", 0.72),
            "model_name": "fixture-classical",
            "feature_version": "acoustic-v1",
            "training_metadata": {"fixture": True},
        }


class FakeDeep:
    def predict(self, samples, sample_rate):
        return SimpleNamespace(
            emotion="angry",
            confidence=0.82,
            probabilities=probability("angry", 0.82),
            chunks_analyzed=1,
            metadata=SimpleNamespace(model_id="fixture-deep"),
        )


def make_engine() -> FusionEngine:
    artifact = FusionCalibrationArtifact(
        version="phase7-fixture",
        labels=tuple(CANONICAL_LABELS),
        classical_temperature=1.0,
        deep_temperature=1.0,
        classical_weight=0.45,
        deep_weight=0.55,
        validation_metrics={},
        metadata={"fixture": True},
    )
    return FusionEngine(artifact)


def test_uncertainty_is_bounded_and_has_expected_low_risk_case():
    result = analyze_uncertainty(
        probability("angry", 0.96),
        js_divergence=0.01,
        duration_seconds=5.0,
        speech_coverage=0.9,
    )
    assert 0 <= result.uncertainty_index <= 1
    assert result.level == "low"
    assert result.top_margin > 0.8


def test_uncertainty_rises_with_ambiguous_distribution_and_disagreement():
    ambiguous = {label: 1.0 / len(CANONICAL_LABELS) for label in CANONICAL_LABELS}
    result = analyze_uncertainty(
        ambiguous,
        js_divergence=0.22,
        duration_seconds=1.0,
        speech_coverage=0.25,
    )
    assert result.uncertainty_index > 0.55
    assert result.level in {"high", "very_high"}


def test_build_insights_separates_evidence_from_caveat():
    features = SimpleNamespace(
        duration_seconds=4.0,
        silence_ratio=0.1,
        estimated_syllable_rate_sps=4.2,
        estimated_pause_ratio=0.08,
        pitch_hz={"mean": 220.0, "std": 28.0},
        energy={"mean": 0.11, "std": 0.02},
        to_vector=lambda: np.zeros(76, dtype=np.float64),
    )
    fused = make_engine().combine(probability("angry", 0.82), probability("angry", 0.86))
    insights = build_insights(
        fusion_prediction=fused,
        classical_model=FakeClassical(),
        acoustic_features=features,
        speech_coverage=0.9,
    )
    assert insights.summary
    assert insights.evidence
    assert insights.uncertainty.level in {"low", "moderate", "high", "very_high"}
    assert "internal" in insights.caveat.lower()


def test_insights_endpoint_combines_inference_features_and_explanations(monkeypatch):
    import app.routers.insights as router

    monkeypatch.setattr(router, "get_engine", make_engine)
    monkeypatch.setattr(router, "get_classical_model", lambda: FakeClassical())
    monkeypatch.setattr(router, "get_deep_model", lambda: FakeDeep())

    response = client.post(
        "/api/v1/audio/emotion/insights",
        files={"file": ("speech.wav", make_wav(), "audio/wav")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["fusion"]["emotion"] == "angry"
    assert payload["insights"]["summary"]
    assert payload["insights"]["evidence"]
    assert 0 <= payload["insights"]["uncertainty"]["uncertainty_index"] <= 1
    assert payload["features"]["model_vector_length"] == 77



def test_insights_endpoint_accepts_profile_without_personal_baseline(monkeypatch):
    import app.routers.insights as router

    monkeypatch.setattr(router, "get_engine", make_engine)
    monkeypatch.setattr(router, "get_classical_model", lambda: FakeClassical())
    monkeypatch.setattr(router, "get_deep_model", lambda: FakeDeep())

    response = client.post(
        "/api/v1/audio/emotion/insights",
        data={"profile_id": "speaker-without-baseline"},
        files={"file": ("speech.wav", make_wav(), "audio/wav")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["insights"]["personal_baseline"] is None
    assert payload["insights"]["personalization"]["model_ready"] is False


def test_insights_endpoint_returns_503_without_fusion(monkeypatch):
    import app.routers.insights as router
    from app.fusion.training import FusionTrainingError

    monkeypatch.setattr(
        router,
        "get_engine",
        lambda: (_ for _ in ()).throw(FusionTrainingError("Fusion calibration is not trained yet.")),
    )
    response = client.post(
        "/api/v1/audio/emotion/insights",
        files={"file": ("speech.wav", make_wav(), "audio/wav")},
    )
    assert response.status_code == 503
    assert "not trained" in response.json()["detail"]


def test_model_linked_evidence_uses_linear_classifier_contribution():
    from types import SimpleNamespace
    from sklearn.linear_model import LogisticRegression

    class FakeScaler:
        def transform(self, values):
            return values

    classifier = SimpleNamespace(
        coef_=np.concatenate([np.ones((1, 77), dtype=np.float64), -np.ones((1, 77), dtype=np.float64)], axis=0),
        classes_=np.asarray([0, 1]),
    )
    model = SimpleNamespace(named_steps={"scale": FakeScaler(), "classifier": classifier})
    bundle = SimpleNamespace(
        model=model,
        feature_names=tuple(["energy_mean"] + [f"f{i}" for i in range(76)]),
        labels=("angry", "sad"),
    )
    classical = SimpleNamespace(bundle=bundle)
    features = SimpleNamespace(
        duration_seconds=4.0,
        silence_ratio=0.1,
        estimated_syllable_rate_sps=4.0,
        estimated_pause_ratio=0.1,
        pitch_hz={"mean": 220.0, "std": 20.0},
        energy={"mean": 0.2, "std": 0.01},
        to_vector=lambda: np.ones(77, dtype=np.float64),
    )
    fused = SimpleNamespace(
        probabilities=probability("angry", 0.82),
        emotion="angry",
        js_divergence=0.01,
        classical_prediction="angry",
        deep_prediction="angry",
        confidence=0.82,
        agreement="high",
    )
    insights = build_insights(
        fusion_prediction=fused,
        classical_model=classical,
        acoustic_features=features,
        speech_coverage=0.9,
    )
    assert any(item.category == "model" for item in insights.evidence)
