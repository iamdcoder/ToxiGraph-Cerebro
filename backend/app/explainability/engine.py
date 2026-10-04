from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from app.fusion.constants import CANONICAL_LABELS, normalize_emotion_label


class ExplainabilityError(ValueError):
    """Raised when explanation or uncertainty analysis cannot be produced."""


@dataclass(frozen=True)
class EvidenceItem:
    category: str
    title: str
    detail: str
    strength: str

    def to_dict(self) -> dict[str, str]:
        return {
            "category": self.category,
            "title": self.title,
            "detail": self.detail,
            "strength": self.strength,
        }


@dataclass(frozen=True)
class UncertaintyFactor:
    name: str
    value: float
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": round(float(self.value), 6),
            "interpretation": self.interpretation,
        }


@dataclass(frozen=True)
class UncertaintyAnalysis:
    uncertainty_index: float
    level: str
    normalized_entropy: float
    top_probability: float
    top_margin: float
    model_disagreement: float
    quality_score: float
    factors: tuple[UncertaintyFactor, ...]
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "uncertainty_index": round(self.uncertainty_index, 6),
            "level": self.level,
            "normalized_entropy": round(self.normalized_entropy, 6),
            "top_probability": round(self.top_probability, 6),
            "top_margin": round(self.top_margin, 6),
            "model_disagreement": round(self.model_disagreement, 6),
            "quality_score": round(self.quality_score, 6),
            "factors": [factor.to_dict() for factor in self.factors],
            "recommendation": self.recommendation,
        }


@dataclass(frozen=True)
class EmotionInsights:
    summary: str
    evidence: tuple[EvidenceItem, ...]
    uncertainty: UncertaintyAnalysis
    alternatives: tuple[tuple[str, float], ...]
    caveat: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "evidence": [item.to_dict() for item in self.evidence],
            "uncertainty": self.uncertainty.to_dict(),
            "alternatives": [
                {"emotion": emotion, "probability": round(probability, 6)}
                for emotion, probability in self.alternatives
            ],
            "caveat": self.caveat,
        }


_FEATURE_LABELS = {
    "pitch_hz_mean": ("Mean pitch", "Hz", 1),
    "pitch_hz_std": ("Pitch variability", "Hz", 1),
    "pitch_hz_min": ("Minimum pitch", "Hz", 1),
    "pitch_hz_max": ("Maximum pitch", "Hz", 1),
    "energy_mean": ("Mean vocal energy", "RMS", 4),
    "energy_std": ("Energy variability", "RMS", 4),
    "spectral_centroid_hz_mean": ("Spectral centroid", "Hz", 0),
    "spectral_bandwidth_hz_mean": ("Spectral bandwidth", "Hz", 0),
    "spectral_rolloff_hz_mean": ("Spectral rolloff", "Hz", 0),
    "spectral_flatness_mean": ("Spectral flatness", "ratio", 4),
    "zero_crossing_rate_mean": ("Zero-crossing rate", "ratio", 4),
    "estimated_syllable_rate_sps": ("Estimated syllable rate", "syllables/s", 1),
    "estimated_pause_count": ("Estimated pause count", "pauses", 0),
    "duration_seconds": ("Utterance duration", "s", 1),
    "silence_ratio": ("Silence ratio", "ratio", 4),
    "speech_duration_seconds": ("Active speech duration", "s", 1),
}


def _clean_probabilities(probabilities: Mapping[str, float]) -> dict[str, float]:
    cleaned = {label: 0.0 for label in CANONICAL_LABELS}
    for raw_label, raw_value in probabilities.items():
        label = normalize_emotion_label(raw_label)
        if label not in cleaned:
            continue
        value = float(raw_value)
        if not math.isfinite(value) or value < 0:
            raise ExplainabilityError("Emotion probabilities must be finite and non-negative.")
        cleaned[label] += value
    total = sum(cleaned.values())
    if total <= 0:
        raise ExplainabilityError("Emotion probabilities contain no valid mass.")
    return {label: value / total for label, value in cleaned.items()}


def _normalized_entropy(probabilities: Mapping[str, float]) -> float:
    values = np.asarray(list(probabilities.values()), dtype=np.float64)
    entropy = -float(np.sum(values * np.log(np.clip(values, 1e-12, 1.0))))
    maximum = math.log(max(len(values), 2))
    return float(entropy / maximum)


def _quality_score(duration_seconds: float, speech_coverage: float) -> float:
    if duration_seconds <= 0:
        return 0.0
    duration_score = min(1.0, float(duration_seconds) / 3.0)
    speech_score = float(np.clip(speech_coverage, 0.0, 1.0))
    return float(0.6 * duration_score + 0.4 * speech_score)


def _uncertainty_level(index: float) -> str:
    if index < 0.25:
        return "low"
    if index < 0.50:
        return "moderate"
    if index < 0.75:
        return "high"
    return "very_high"


def analyze_uncertainty(
    probabilities: Mapping[str, float],
    *,
    js_divergence: float,
    duration_seconds: float,
    speech_coverage: float,
    audio_quality_score: float | None = None,
) -> UncertaintyAnalysis:
    canonical = _clean_probabilities(probabilities)
    ordered = sorted(canonical.values(), reverse=True)
    top_probability = float(ordered[0])
    second_probability = float(ordered[1] if len(ordered) > 1 else 0.0)
    margin = top_probability - second_probability
    entropy = _normalized_entropy(canonical)
    disagreement = float(np.clip(js_divergence / 0.25, 0.0, 1.0))
    quality = float(np.clip(audio_quality_score, 0.0, 1.0)) if audio_quality_score is not None else _quality_score(duration_seconds, speech_coverage)
    quality_penalty = 1.0 - quality

    uncertainty_index = float(
        0.45 * entropy
        + 0.25 * disagreement
        + 0.20 * (1.0 - margin)
        + 0.10 * quality_penalty
    )
    uncertainty_index = float(np.clip(uncertainty_index, 0.0, 1.0))
    level = _uncertainty_level(uncertainty_index)

    factors = (
        UncertaintyFactor(
            name="distribution_entropy",
            value=entropy,
            interpretation=(
                "Higher means probability mass is spread across more emotions."
            ),
        ),
        UncertaintyFactor(
            name="top_class_margin",
            value=margin,
            interpretation=(
                "Higher means the top prediction is more clearly separated from the runner-up."
            ),
        ),
        UncertaintyFactor(
            name="model_disagreement",
            value=js_divergence,
            interpretation=(
                "Higher Jensen–Shannon divergence means the classical and deep models disagree more."
            ),
        ),
        UncertaintyFactor(
            name="audio_quality_score",
            value=quality,
            interpretation=(
                "Higher means the recording passed more of CEREBRO's measured audio-quality checks."
            ),
        ),
    )

    if level in {"high", "very_high"}:
        recommendation = "Use this prediction cautiously; a clearer or longer recording is recommended."
    elif level == "moderate":
        recommendation = "Prediction is usable, but model ambiguity is still present."
    else:
        recommendation = "Prediction is comparatively stable under the current model signals."

    return UncertaintyAnalysis(
        uncertainty_index=uncertainty_index,
        level=level,
        normalized_entropy=entropy,
        top_probability=top_probability,
        top_margin=margin,
        model_disagreement=js_divergence,
        quality_score=quality,
        factors=factors,
        recommendation=recommendation,
    )


def _acoustic_measurements(features: Any) -> list[EvidenceItem]:
    measurements: list[EvidenceItem] = []
    pitch = getattr(features, "pitch_hz", {})
    energy = getattr(features, "energy", {})
    if pitch:
        measurements.append(
            EvidenceItem(
                category="acoustic",
                title="Pitch profile",
                detail=(
                    f"Mean {float(pitch['mean']):.1f} Hz with {float(pitch['std']):.1f} Hz spread."
                ),
                strength="observed",
            )
        )
    if energy:
        measurements.append(
            EvidenceItem(
                category="acoustic",
                title="Vocal energy profile",
                detail=(
                    f"Mean RMS {float(energy['mean']):.4f} with {float(energy['std']):.4f} variability."
                ),
                strength="observed",
            )
        )
    measurements.append(
        EvidenceItem(
            category="acoustic",
            title="Speech-rate estimate",
            detail=(
                f"Approximately {float(getattr(features, 'estimated_syllable_rate_sps', 0.0)):.2f} syllables/s."
            ),
            strength="observed",
        )
    )
    measurements.append(
        EvidenceItem(
            category="acoustic",
            title="Pause / silence profile",
            detail=(
                f"Estimated pause ratio {float(getattr(features, 'estimated_pause_ratio', 0.0)):.3f}; "
                f"silence ratio {float(getattr(features, 'silence_ratio', 0.0)):.3f}."
            ),
            strength="observed",
        )
    )
    return measurements


def _model_relevance_evidence(
    classical_model: Any,
    vector: np.ndarray,
    predicted_emotion: str,
) -> list[EvidenceItem]:
    bundle = getattr(classical_model, "bundle", None)
    model = getattr(bundle, "model", None)
    feature_names = tuple(getattr(bundle, "feature_names", ()))
    if model is None or not feature_names or len(feature_names) != len(vector):
        return []

    classifier = model
    scaler = None
    if hasattr(model, "named_steps"):
        scaler = model.named_steps.get("scale")
        classifier = model.named_steps.get("classifier", model)

    try:
        if scaler is not None:
            transformed = np.asarray(scaler.transform(vector.reshape(1, -1))[0], dtype=np.float64)
        else:
            transformed = np.asarray(vector, dtype=np.float64)
    except Exception:
        transformed = np.asarray(vector, dtype=np.float64)

    contributions: list[tuple[float, str, str]] = []
    if hasattr(classifier, "coef_"):
        coefficients = np.asarray(classifier.coef_, dtype=np.float64)
        classes = list(getattr(classifier, "classes_", range(coefficients.shape[0])))
        try:
            class_index = classes.index(getattr(bundle, "labels", tuple(CANONICAL_LABELS)).index(predicted_emotion))
        except (ValueError, AttributeError):
            class_index = int(np.argmax(np.asarray(getattr(classifier, "predict_proba")(vector.reshape(1, -1))[0])))
        row = coefficients[class_index if coefficients.shape[0] > 1 else 0]
        scores = transformed * row
        for index in np.argsort(scores)[::-1]:
            if scores[index] <= 0:
                break
            contributions.append((float(scores[index]), feature_names[index], "positive model contribution"))
            if len(contributions) == 3:
                break
    elif hasattr(classifier, "feature_importances_"):
        importances = np.asarray(classifier.feature_importances_, dtype=np.float64)
        for index in np.argsort(importances)[::-1][:3]:
            contributions.append(
                (
                    float(importances[index]),
                    feature_names[index],
                    "high global model importance",
                )
            )

    items: list[EvidenceItem] = []
    for score, name, meaning in contributions:
        label, unit, precision = _FEATURE_LABELS.get(name, (name.replace("_", " ").title(), "", 2))
        value = float(vector[feature_names.index(name)])
        if precision == 0:
            formatted_value = f"{value:.0f}"
        elif precision == 1:
            formatted_value = f"{value:.1f}"
        else:
            formatted_value = f"{value:.{precision}f}"
        suffix = f" {unit}" if unit else ""
        items.append(
            EvidenceItem(
                category="model",
                title=label,
                detail=f"Measured value {formatted_value}{suffix}; {meaning} for the classical model.",
                strength="model_linked",
            )
        )
    return items


def build_insights(
    *,
    fusion_prediction: Any,
    classical_model: Any,
    acoustic_features: Any,
    speech_coverage: float,
    audio_quality_score: float | None = None,
    audio_quality_verdict: str | None = None,
) -> EmotionInsights:
    probabilities = _clean_probabilities(fusion_prediction.probabilities)
    ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    emotion = normalize_emotion_label(fusion_prediction.emotion)
    uncertainty = analyze_uncertainty(
        probabilities,
        js_divergence=float(fusion_prediction.js_divergence),
        duration_seconds=float(acoustic_features.duration_seconds),
        speech_coverage=float(speech_coverage),
        audio_quality_score=audio_quality_score,
    )

    classical_emotion = normalize_emotion_label(fusion_prediction.classical_prediction)
    deep_emotion = normalize_emotion_label(fusion_prediction.deep_prediction)
    if classical_emotion == deep_emotion == emotion:
        consensus = EvidenceItem(
            category="fusion",
            title="Cross-model consensus",
            detail=(
                f"Both the classical acoustic model and the deep speech model independently selected {emotion.upper()}."
            ),
            strength="strong",
        )
    else:
        consensus = EvidenceItem(
            category="fusion",
            title="Model disagreement",
            detail=(
                f"Classical model selected {classical_emotion.upper()}, while the deep model selected {deep_emotion.upper()}. "
                f"The fused result is {emotion.upper()}."
            ),
            strength="caution",
        )

    evidence = [
        consensus,
        EvidenceItem(
            category="fusion",
            title="Calibrated probability",
            detail=(
                f"The fused distribution assigns {float(fusion_prediction.confidence) * 100:.1f}% to {emotion.upper()}, "
                f"with a top-two margin of {uncertainty.top_margin * 100:.1f} percentage points."
            ),
            strength="strong" if uncertainty.top_margin >= 0.25 else "moderate",
        ),
    ]
    evidence.extend(_model_relevance_evidence(
        classical_model,
        acoustic_features.to_vector(),
        emotion,
    ))
    evidence.extend(_acoustic_measurements(acoustic_features))
    quality_detail = (
        f"Duration {float(acoustic_features.duration_seconds):.2f} s with "
        f"{float(speech_coverage) * 100:.1f}% active speech coverage."
    )
    if audio_quality_score is not None:
        quality_detail += f" Overall recording-quality score {float(audio_quality_score) * 100:.1f}%"
        if audio_quality_verdict:
            quality_detail += f" ({audio_quality_verdict})."
        else:
            quality_detail += "."
    evidence.append(
        EvidenceItem(
            category="quality",
            title="Recording quality context",
            detail=quality_detail,
            strength="context",
        )
    )
    if uncertainty.level in {"high", "very_high"}:
        evidence.append(
            EvidenceItem(
                category="uncertainty",
                title="Prediction needs caution",
                detail=uncertainty.recommendation,
                strength="caution",
            )
        )

    alternatives = tuple(ordered[1:4])
    summary = (
        f"CEREBRO estimates {emotion.upper()} with {float(fusion_prediction.confidence) * 100:.1f}% calibrated probability. "
        f"Classical/deep agreement is {str(fusion_prediction.agreement).upper()}, and the operational uncertainty level is "
        f"{uncertainty.level.replace('_', ' ').upper()}."
    )

    return EmotionInsights(
        summary=summary,
        evidence=tuple(evidence),
        uncertainty=uncertainty,
        alternatives=alternatives,
        caveat=(
            "These explanations describe measurable acoustic/model signals used by CEREBRO. "
            "They do not establish the speaker's private or internal emotional state."
        ),
    )
