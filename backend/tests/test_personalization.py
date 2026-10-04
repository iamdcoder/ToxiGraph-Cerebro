from __future__ import annotations

import math

import pytest

from app.fusion.constants import CANONICAL_LABELS
from app.personalization.service import (
    FEATURE_VERSION,
    MAX_PERSONAL_WEIGHT,
    MIN_FEEDBACK,
    MIN_UNIQUE_LABELS,
    PersonalizedCalibrationService,
    PersonalizationError,
)


def acoustic(seed: float = 0.0) -> dict[str, float]:
    return {
        "pitch_mean_hz": 180 + seed * 8,
        "pitch_std_hz": 30 + seed * 2,
        "energy_mean": 0.08 + seed * 0.01,
        "energy_std": 0.03 + seed * 0.004,
        "estimated_syllable_rate_sps": 3.4 + seed * 0.2,
        "estimated_pause_ratio": 0.22 + seed * 0.01,
    }


def probs(label: str, confidence: float = 0.72) -> dict[str, float]:
    residual = (1 - confidence) / (len(CANONICAL_LABELS) - 1)
    return {emotion: confidence if emotion == label else residual for emotion in CANONICAL_LABELS}


def test_status_starts_untrained(tmp_path):
    service = PersonalizedCalibrationService(tmp_path)
    status = service.status("speaker1")
    assert status["feature_version"] == FEATURE_VERSION
    assert status["feedback_count"] == 0
    assert status["model_ready"] is False


def test_feedback_requires_canonical_label_and_valid_audio_features(tmp_path):
    service = PersonalizedCalibrationService(tmp_path)
    with pytest.raises(PersonalizationError):
        service.record_feedback(
            "speaker1",
            session_id="s1",
            label="mystery",
            fused_probabilities=probs("neutral"),
            acoustic=acoustic(),
        )


def test_feedback_is_idempotent_by_session_id(tmp_path):
    service = PersonalizedCalibrationService(tmp_path)
    for index in range(2):
        service.record_feedback(
            "speaker1",
            session_id="same-session",
            label="neutral" if index == 0 else "happy",
            fused_probabilities=probs("neutral" if index == 0 else "happy"),
            acoustic=acoustic(index),
        )
    status = service.status("speaker1")
    assert status["feedback_count"] == 1
    assert status["labels"] == ["happy"]


def test_model_is_not_ready_until_feedback_gate(tmp_path):
    service = PersonalizedCalibrationService(tmp_path)
    for index in range(MIN_FEEDBACK - 1):
        label = CANONICAL_LABELS[index % MIN_UNIQUE_LABELS]
        service.record_feedback(
            "speaker1",
            session_id=f"s{index}",
            label=label,
            fused_probabilities=probs(label),
            acoustic=acoustic(index),
        )
    assert service.status("speaker1")["model_ready"] is False


def test_model_becomes_ready_with_diverse_high_quality_feedback(tmp_path):
    service = PersonalizedCalibrationService(tmp_path)
    labels = ["neutral", "happy", "angry"] * 4
    for index, label in enumerate(labels):
        service.record_feedback(
            "speaker1",
            session_id=f"s{index}",
            label=label,
            fused_probabilities=probs(label, 0.70 + (index % 3) * 0.05),
            acoustic=acoustic(index % 4),
            quality_score=0.9,
        )
    status = service.status("speaker1")
    assert status["model_ready"] is True
    assert status["unique_labels"] == 3
    assert status["model_sample_count"] == len(labels)
    assert 0 <= status["training_fit"] <= 1


def test_prediction_preserves_probability_normalization(tmp_path):
    service = PersonalizedCalibrationService(tmp_path)
    labels = ["neutral", "happy", "angry"] * 4
    for index, label in enumerate(labels):
        service.record_feedback(
            "speaker1",
            session_id=f"s{index}",
            label=label,
            fused_probabilities=probs(label),
            acoustic=acoustic(index % 3),
            quality_score=0.95,
        )
    result = service.predict("speaker1", probs("happy"), acoustic(2))
    assert result["model_ready"] is True
    assert result["global_emotion"] == "happy"
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)
    assert 0 < result["adjustment_weight"] <= MAX_PERSONAL_WEIGHT
    assert all(math.isfinite(value) for value in result["probabilities"].values())


def test_low_quality_feedback_counts_but_does_not_train(tmp_path):
    service = PersonalizedCalibrationService(tmp_path)
    labels = ["neutral", "happy", "angry"] * 4
    for index, label in enumerate(labels):
        service.record_feedback(
            "speaker1",
            session_id=f"s{index}",
            label=label,
            fused_probabilities=probs(label),
            acoustic=acoustic(index % 2),
            quality_score=0.2,
        )
    status = service.status("speaker1")
    assert status["feedback_count"] == 12
    assert status["eligible_feedback_count"] == 0
    assert status["model_ready"] is False


def test_personalized_model_persistence_survives_service_restart(tmp_path):
    service = PersonalizedCalibrationService(tmp_path)
    labels = ["neutral", "happy", "angry"] * 4
    for index, label in enumerate(labels):
        service.record_feedback(
            "speaker1",
            session_id=f"s{index}",
            label=label,
            fused_probabilities=probs(label),
            acoustic=acoustic(index % 3),
            quality_score=0.95,
        )
    reloaded = PersonalizedCalibrationService(tmp_path)
    result = reloaded.predict("speaker1", probs("happy"), acoustic())
    assert result["model_ready"] is True
    assert result["sample_count"] == 12


def test_invalid_probability_mapping_is_rejected(tmp_path):
    service = PersonalizedCalibrationService(tmp_path)
    invalid = probs("neutral")
    invalid["happy"] = float("nan")
    with pytest.raises(PersonalizationError):
        service.predict("speaker1", invalid, acoustic())


def test_predict_without_profile_keeps_global_prediction(tmp_path):
    service = PersonalizedCalibrationService(tmp_path)
    result = service.predict("speaker1", probs("sad"), acoustic())
    assert result["model_ready"] is False
    assert result["personalized_emotion"] == result["global_emotion"] == "sad"
    assert result["adjustment_weight"] == 0
