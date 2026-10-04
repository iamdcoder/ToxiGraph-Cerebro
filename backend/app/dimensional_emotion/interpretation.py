from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AffectInterpretation:
    valence_label: str
    arousal_label: str
    dominance_label: str
    quadrant: str
    summary: str


def _polarity(value: float, low: float = 0.4, high: float = 0.6) -> str:
    if value < low:
        return "negative"
    if value > high:
        return "positive"
    return "neutral"


def _activation(value: float, low: float = 0.4, high: float = 0.6) -> str:
    if value < low:
        return "low"
    if value > high:
        return "high"
    return "moderate"


def _dominance(value: float, low: float = 0.4, high: float = 0.6) -> str:
    if value < low:
        return "low"
    if value > high:
        return "high"
    return "moderate"


def interpret_affect(valence: float, arousal: float, dominance: float) -> AffectInterpretation:
    v = _polarity(valence)
    a = _activation(arousal)
    d = _dominance(dominance)

    if v == "positive" and a == "high":
        quadrant = "positive / activated"
    elif v == "positive" and a == "low":
        quadrant = "positive / calm"
    elif v == "negative" and a == "high":
        quadrant = "negative / activated"
    elif v == "negative" and a == "low":
        quadrant = "negative / calm"
    else:
        quadrant = "mixed / transitional"

    summary = (
        f"Speech affect is {quadrant}; valence is {v}, arousal is {a}, "
        f"and dominance is {d}."
    )
    return AffectInterpretation(
        valence_label=v,
        arousal_label=a,
        dominance_label=d,
        quadrant=quadrant,
        summary=summary,
    )
