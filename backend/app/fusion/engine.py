from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from app.fusion.calibration import FusionCalibrationArtifact, apply_temperature, canonical_probability_vector
from app.fusion.constants import CANONICAL_LABELS, normalize_emotion_label


class FusionError(ValueError):
    """Raised when model outputs cannot be fused safely."""


def _js_divergence(first: np.ndarray, second: np.ndarray) -> float:
    midpoint = 0.5 * (first + second)
    first_safe = np.clip(first, 1e-12, 1.0)
    second_safe = np.clip(second, 1e-12, 1.0)
    midpoint_safe = np.clip(midpoint, 1e-12, 1.0)
    kl_first = np.sum(first_safe * np.log(first_safe / midpoint_safe))
    kl_second = np.sum(second_safe * np.log(second_safe / midpoint_safe))
    return float(0.5 * (kl_first + kl_second))


@dataclass(frozen=True)
class FusionPrediction:
    emotion: str
    confidence: float
    probabilities: dict[str, float]
    agreement: str
    js_divergence: float
    classical_prediction: str
    deep_prediction: str
    classical_confidence: float
    deep_confidence: float
    classical_weight: float
    deep_weight: float
    classical_temperature: float
    deep_temperature: float
    calibration_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "emotion": self.emotion,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "agreement": self.agreement,
            "js_divergence": self.js_divergence,
            "classical_prediction": self.classical_prediction,
            "deep_prediction": self.deep_prediction,
            "classical_confidence": self.classical_confidence,
            "deep_confidence": self.deep_confidence,
            "classical_weight": self.classical_weight,
            "deep_weight": self.deep_weight,
            "classical_temperature": self.classical_temperature,
            "deep_temperature": self.deep_temperature,
            "calibration_version": self.calibration_version,
        }


def _agreement_level(
    classical_emotion: str,
    deep_emotion: str,
    js_divergence: float,
) -> str:
    if classical_emotion == deep_emotion and js_divergence < 0.06:
        return "high"
    if classical_emotion == deep_emotion or js_divergence < 0.14:
        return "moderate"
    return "low"


class FusionEngine:
    def __init__(self, artifact: FusionCalibrationArtifact) -> None:
        if not math.isclose(artifact.classical_weight + artifact.deep_weight, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise FusionError("Fusion weights must sum to 1.")
        if artifact.classical_weight <= 0 or artifact.deep_weight <= 0:
            raise FusionError("Fusion weights must both be positive.")
        self.artifact = artifact
        self.labels = tuple(artifact.labels) or CANONICAL_LABELS

    def combine(
        self,
        classical_probabilities: Mapping[str, float],
        deep_probabilities: Mapping[str, float],
    ) -> FusionPrediction:
        classical = apply_temperature(
            classical_probabilities,
            self.artifact.classical_temperature,
            self.labels,
        )
        deep = apply_temperature(
            deep_probabilities,
            self.artifact.deep_temperature,
            self.labels,
        )

        classical_vector = canonical_probability_vector(classical, self.labels)
        deep_vector = canonical_probability_vector(deep, self.labels)
        fused = (
            self.artifact.classical_weight * classical_vector
            + self.artifact.deep_weight * deep_vector
        )
        fused /= fused.sum()

        classical_emotion = max(classical, key=classical.get)
        deep_emotion = max(deep, key=deep.get)
        fused_index = int(np.argmax(fused))
        emotion = self.labels[fused_index]
        confidence = float(fused[fused_index])
        js_divergence = _js_divergence(classical_vector, deep_vector)

        return FusionPrediction(
            emotion=emotion,
            confidence=confidence,
            probabilities={
                label: float(value)
                for label, value in zip(self.labels, fused, strict=True)
            },
            agreement=_agreement_level(classical_emotion, deep_emotion, js_divergence),
            js_divergence=js_divergence,
            classical_prediction=normalize_emotion_label(classical_emotion),
            deep_prediction=normalize_emotion_label(deep_emotion),
            classical_confidence=float(classical[classical_emotion]),
            deep_confidence=float(deep[deep_emotion]),
            classical_weight=self.artifact.classical_weight,
            deep_weight=self.artifact.deep_weight,
            classical_temperature=self.artifact.classical_temperature,
            deep_temperature=self.artifact.deep_temperature,
            calibration_version=self.artifact.version,
        )
