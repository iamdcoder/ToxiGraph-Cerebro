from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.fusion.constants import CANONICAL_LABELS


@dataclass(frozen=True)
class FeatureDelta:
    name: str
    a_value: float
    b_value: float
    absolute_delta: float
    relative_delta: float
    direction: str
    unit: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "a_value": self.a_value,
            "b_value": self.b_value,
            "absolute_delta": self.absolute_delta,
            "relative_delta": self.relative_delta,
            "direction": self.direction,
            "unit": self.unit,
        }


@dataclass(frozen=True)
class EmotionProbabilityDelta:
    emotion: str
    a_probability: float
    b_probability: float
    delta: float
    absolute_delta: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "emotion": self.emotion,
            "a_probability": self.a_probability,
            "b_probability": self.b_probability,
            "delta": self.delta,
            "absolute_delta": self.absolute_delta,
        }


@dataclass(frozen=True)
class ComparisonResult:
    acoustic_change_score: float
    acoustic_similarity_score: float
    emotion_distribution_shift: float
    emotion_changed: bool
    top_emotion_shift: str
    largest_acoustic_change: str
    acoustic_features: list[FeatureDelta]
    emotion_probabilities: list[EmotionProbabilityDelta]
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "acoustic_change_score": self.acoustic_change_score,
            "acoustic_similarity_score": self.acoustic_similarity_score,
            "emotion_distribution_shift": self.emotion_distribution_shift,
            "emotion_changed": self.emotion_changed,
            "top_emotion_shift": self.top_emotion_shift,
            "largest_acoustic_change": self.largest_acoustic_change,
            "acoustic_features": [item.to_dict() for item in self.acoustic_features],
            "emotion_probabilities": [item.to_dict() for item in self.emotion_probabilities],
            "interpretation": self.interpretation,
        }


FEATURE_SPECS = (
    ("pitch_mean_hz", "pitch_hz.mean", "Hz"),
    ("pitch_std_hz", "pitch_hz.std", "Hz"),
    ("energy_mean", "energy.mean", "normalized"),
    ("energy_std", "energy.std", "normalized"),
    ("spectral_centroid_hz", "spectral_centroid_hz.mean", "Hz"),
    ("spectral_bandwidth_hz", "spectral_bandwidth_hz.mean", "Hz"),
    ("spectral_rolloff_hz", "spectral_rolloff_hz.mean", "Hz"),
    ("zero_crossing_rate", "zero_crossing_rate.mean", "ratio"),
    ("estimated_syllable_rate_sps", "estimated_syllable_rate_sps", "per_second"),
    ("estimated_pause_ratio", "estimated_pause_ratio", "ratio"),
    ("speech_coverage", "speech_coverage", "ratio"),
)

WEIGHTED_FEATURES = {
    "pitch_mean_hz": 1.3,
    "pitch_std_hz": 1.0,
    "energy_mean": 1.4,
    "energy_std": 0.8,
    "spectral_centroid_hz": 0.9,
    "spectral_bandwidth_hz": 0.7,
    "spectral_rolloff_hz": 0.6,
    "zero_crossing_rate": 0.4,
    "estimated_syllable_rate_sps": 1.0,
    "estimated_pause_ratio": 0.8,
    "speech_coverage": 0.7,
}


def compare_insights(analysis_a: Any, analysis_b: Any) -> ComparisonResult:
    feature_a = _as_dict(analysis_a.features)
    feature_b = _as_dict(analysis_b.features)

    acoustic_deltas: list[FeatureDelta] = []
    weighted_scores: list[tuple[str, float, float]] = []
    for name, path, unit in FEATURE_SPECS:
        a_value = _read(path, feature_a)
        b_value = _read(path, feature_b)
        relative = _relative_delta(a_value, b_value)
        magnitude = min(1.0, math.tanh(abs(relative)))
        weight = WEIGHTED_FEATURES.get(name, 1.0)
        weighted_scores.append((name, magnitude, weight))
        acoustic_deltas.append(
            FeatureDelta(
                name=name,
                a_value=a_value,
                b_value=b_value,
                absolute_delta=b_value - a_value,
                relative_delta=relative,
                direction=_direction(a_value, b_value),
                unit=unit,
            )
        )

    total_weight = sum(item[2] for item in weighted_scores)
    change_score = sum(score * weight for _, score, weight in weighted_scores) / max(total_weight, 1e-9)
    change_score = float(np.clip(change_score, 0.0, 1.0))
    similarity = float(np.clip(1.0 - change_score, 0.0, 1.0))

    mfcc_a = np.asarray(feature_a.get("mfcc_mean", []), dtype=np.float64)
    mfcc_b = np.asarray(feature_b.get("mfcc_mean", []), dtype=np.float64)
    if mfcc_a.size and mfcc_a.size == mfcc_b.size:
        cosine = float(np.dot(mfcc_a, mfcc_b) / max(np.linalg.norm(mfcc_a) * np.linalg.norm(mfcc_b), 1e-12))
        similarity = float(np.clip(0.5 * similarity + 0.5 * ((cosine + 1.0) / 2.0), 0.0, 1.0))

    probabilities_a = {item.emotion: float(item.probability) for item in analysis_a.fusion.probabilities}
    probabilities_b = {item.emotion: float(item.probability) for item in analysis_b.fusion.probabilities}
    probabilities_a = _canonical_distribution(probabilities_a)
    probabilities_b = _canonical_distribution(probabilities_b)

    deltas = [
        EmotionProbabilityDelta(
            emotion=emotion,
            a_probability=probabilities_a[emotion],
            b_probability=probabilities_b[emotion],
            delta=probabilities_b[emotion] - probabilities_a[emotion],
            absolute_delta=abs(probabilities_b[emotion] - probabilities_a[emotion]),
        )
        for emotion in CANONICAL_LABELS
    ]
    deltas.sort(key=lambda item: item.absolute_delta, reverse=True)

    js_shift = _js_divergence(probabilities_a, probabilities_b)
    emotion_changed = analysis_a.fusion.emotion != analysis_b.fusion.emotion
    top_shift = deltas[0].emotion if deltas else "unknown"
    strongest_acoustic = max(acoustic_deltas, key=lambda item: abs(item.relative_delta)).name if acoustic_deltas else "unknown"

    interpretation = _build_interpretation(
        analysis_a.fusion.emotion,
        analysis_b.fusion.emotion,
        change_score,
        js_shift,
        top_shift,
        strongest_acoustic,
    )

    return ComparisonResult(
        acoustic_change_score=change_score,
        acoustic_similarity_score=similarity,
        emotion_distribution_shift=js_shift,
        emotion_changed=emotion_changed,
        top_emotion_shift=top_shift,
        largest_acoustic_change=strongest_acoustic,
        acoustic_features=acoustic_deltas,
        emotion_probabilities=deltas,
        interpretation=interpretation,
    )



def _as_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    if isinstance(value, dict):
        return value
    raise TypeError("Comparison features must be a mapping or model-like object.")

def _read(path: str, payload: dict[str, Any]) -> float:
    current: Any = payload
    for part in path.split("."):
        current = current.get(part, 0.0) if isinstance(current, dict) else 0.0
    return float(current or 0.0)


def _relative_delta(a: float, b: float) -> float:
    scale = max(abs(a), abs(b), 1e-6)
    return (b - a) / scale


def _direction(a: float, b: float) -> str:
    delta = b - a
    scale = max(abs(a), abs(b), 1e-6)
    if abs(delta) <= scale * 0.05:
        return "stable"
    return "increased" if delta > 0 else "decreased"


def _canonical_distribution(values: dict[str, float]) -> dict[str, float]:
    result = {emotion: 0.0 for emotion in CANONICAL_LABELS}
    aliases = {
        "surprise": "surprised",
        "calm": "neutral",
        "happiness": "happy",
        "fearful": "fear",
    }
    for label, probability in values.items():
        canonical = aliases.get(label.lower(), label.lower())
        if canonical in result:
            result[canonical] += max(0.0, float(probability))
    total = sum(result.values())
    if total <= 0:
        uniform = 1.0 / len(CANONICAL_LABELS)
        return {emotion: uniform for emotion in CANONICAL_LABELS}
    return {emotion: result[emotion] / total for emotion in CANONICAL_LABELS}


def _js_divergence(a: dict[str, float], b: dict[str, float]) -> float:
    p = np.asarray([a[label] for label in CANONICAL_LABELS], dtype=np.float64)
    q = np.asarray([b[label] for label in CANONICAL_LABELS], dtype=np.float64)
    m = 0.5 * (p + q)
    p_safe = np.maximum(p, 1e-12)
    q_safe = np.maximum(q, 1e-12)
    m_safe = np.maximum(m, 1e-12)
    divergence = 0.5 * np.sum(p_safe * np.log2(p_safe / m_safe)) + 0.5 * np.sum(q_safe * np.log2(q_safe / m_safe))
    return float(np.clip(divergence, 0.0, 1.0))


def _build_interpretation(
    emotion_a: str,
    emotion_b: str,
    acoustic_change: float,
    distribution_shift: float,
    top_shift: str,
    strongest_feature: str,
) -> str:
    emotion_phrase = (
        f"The final conveyed-emotion estimate changed from {emotion_a} to {emotion_b}."
        if emotion_a != emotion_b
        else f"Both recordings received the same final conveyed-emotion estimate: {emotion_a}."
    )
    change_phrase = (
        "The acoustic profile changed substantially between the recordings."
        if acoustic_change >= 0.55
        else "The acoustic profile changed moderately between the recordings."
        if acoustic_change >= 0.30
        else "The acoustic profile stayed relatively similar between the recordings."
    )
    return (
        f"{emotion_phrase} {change_phrase} The largest acoustic relative change was in {strongest_feature.replace('_', ' ')}, "
        f"while the largest fused probability movement was in {top_shift}. "
        f"Emotion-distribution shift (Jensen–Shannon divergence) was {distribution_shift:.3f}."
    )
