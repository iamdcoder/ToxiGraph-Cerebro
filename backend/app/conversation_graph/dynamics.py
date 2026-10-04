from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from app.speaker_diarization.engine import SpeakerAwareAnalysis, SpeakerTurnAnalysis


class ConversationDynamicsError(ValueError):
    """Raised when conversation affect dynamics cannot be computed."""


@dataclass(frozen=True)
class ConversationDynamicsConfig:
    escalation_arousal_delta: float = 0.08
    escalation_valence_delta: float = -0.08
    deescalation_arousal_delta: float = -0.08
    deescalation_valence_delta: float = 0.08
    strong_shift: float = 0.12
    immediate_response_seconds: float = 0.50
    quick_response_seconds: float = 1.50


@dataclass(frozen=True)
class ConversationDynamicsEvent:
    index: int
    at_seconds: float
    from_speaker: str
    to_speaker: str
    gap_seconds: float
    previous_emotion: str
    next_emotion: str
    valence_change: float | None
    arousal_change: float | None
    dominance_change: float | None
    escalation_score: float
    deescalation_score: float
    affect_magnitude: float
    affect_alignment: float | None
    event_type: str
    response_speed: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "at_seconds": round(self.at_seconds, 4),
            "from_speaker": self.from_speaker,
            "to_speaker": self.to_speaker,
            "gap_seconds": round(self.gap_seconds, 4),
            "previous_emotion": self.previous_emotion,
            "next_emotion": self.next_emotion,
            "valence_change": _round_optional(self.valence_change),
            "arousal_change": _round_optional(self.arousal_change),
            "dominance_change": _round_optional(self.dominance_change),
            "escalation_score": round(self.escalation_score, 6),
            "deescalation_score": round(self.deescalation_score, 6),
            "affect_magnitude": round(self.affect_magnitude, 6),
            "affect_alignment": _round_optional(self.affect_alignment),
            "event_type": self.event_type,
            "response_speed": self.response_speed,
        }


@dataclass(frozen=True)
class PairDynamics:
    from_speaker: str
    to_speaker: str
    response_count: int
    escalation_count: int
    deescalation_count: int
    mean_gap_seconds: float
    median_gap_seconds: float
    mean_escalation_score: float
    mean_deescalation_score: float
    affect_synchrony: float | None
    mean_valence_change: float | None
    mean_arousal_change: float | None
    mean_affect_magnitude: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_speaker": self.from_speaker,
            "to_speaker": self.to_speaker,
            "response_count": self.response_count,
            "escalation_count": self.escalation_count,
            "deescalation_count": self.deescalation_count,
            "mean_gap_seconds": round(self.mean_gap_seconds, 4),
            "median_gap_seconds": round(self.median_gap_seconds, 4),
            "mean_escalation_score": round(self.mean_escalation_score, 6),
            "mean_deescalation_score": round(self.mean_deescalation_score, 6),
            "affect_synchrony": _round_optional(self.affect_synchrony),
            "mean_valence_change": _round_optional(self.mean_valence_change),
            "mean_arousal_change": _round_optional(self.mean_arousal_change),
            "mean_affect_magnitude": round(self.mean_affect_magnitude, 6),
        }


@dataclass(frozen=True)
class ConversationDynamics:
    event_count: int
    escalation_score: float
    deescalation_score: float
    net_tension: float
    tension_trend: float
    volatility: float
    stability_score: float
    affect_synchrony: float | None
    mean_response_gap_seconds: float | None
    median_response_gap_seconds: float | None
    p90_response_gap_seconds: float | None
    fast_response_share: float
    response_latency_shift_seconds: float | None
    arousal_rise_events: int
    valence_drop_events: int
    escalation_events: int
    deescalation_events: int
    stable_events: int
    dominant_pattern: str
    events: tuple[ConversationDynamicsEvent, ...]
    pair_metrics: tuple[PairDynamics, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_count": self.event_count,
            "escalation_score": round(self.escalation_score, 6),
            "deescalation_score": round(self.deescalation_score, 6),
            "net_tension": round(self.net_tension, 6),
            "tension_trend": round(self.tension_trend, 6),
            "volatility": round(self.volatility, 6),
            "stability_score": round(self.stability_score, 6),
            "affect_synchrony": _round_optional(self.affect_synchrony),
            "mean_response_gap_seconds": _round_optional(self.mean_response_gap_seconds),
            "median_response_gap_seconds": _round_optional(self.median_response_gap_seconds),
            "p90_response_gap_seconds": _round_optional(self.p90_response_gap_seconds),
            "fast_response_share": round(self.fast_response_share, 6),
            "response_latency_shift_seconds": _round_optional(self.response_latency_shift_seconds),
            "arousal_rise_events": self.arousal_rise_events,
            "valence_drop_events": self.valence_drop_events,
            "escalation_events": self.escalation_events,
            "deescalation_events": self.deescalation_events,
            "stable_events": self.stable_events,
            "dominant_pattern": self.dominant_pattern,
            "events": [item.to_dict() for item in self.events],
            "pair_metrics": [item.to_dict() for item in self.pair_metrics],
            "semantics": {
                "escalation": "Observed increases in activation and/or decreases in valence across adjacent speaker turns; not a causal claim.",
                "deescalation": "Observed decreases in activation and/or increases in valence across adjacent speaker turns; not a causal claim.",
                "affect_synchrony": "Similarity of expressed V/A/D state at speaker response boundaries; not evidence of shared internal emotion.",
                "volatility": "Mean normalized V/A/D movement between analyzed turns.",
                "latency_shift": "Change in response-gap timing between the early and late halves of the conversation.",
            },
        }


def _round_optional(value: float | None) -> float | None:
    return round(float(value), 6) if value is not None else None


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2.0)


def _p90(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.90
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _state(turn: SpeakerTurnAnalysis) -> dict[str, float] | None:
    values = {
        "valence": _finite(turn.valence),
        "arousal": _finite(turn.arousal),
        "dominance": _finite(turn.dominance),
    }
    if any(value is None for value in values.values()):
        return None
    return {key: float(value) for key, value in values.items() if value is not None}


def _distance(first: dict[str, float] | None, second: dict[str, float] | None) -> float | None:
    if first is None or second is None:
        return None
    keys = [key for key in first if key in second]
    if not keys:
        return None
    denominator = math.sqrt(len(keys))
    return float(math.sqrt(sum((second[key] - first[key]) ** 2 for key in keys)) / denominator)


def _directional_scores(
    valence_change: float | None,
    arousal_change: float | None,
    config: ConversationDynamicsConfig,
) -> tuple[float, float]:
    if config.escalation_arousal_delta <= 0 or config.deescalation_valence_delta <= 0:
        raise ConversationDynamicsError("Invalid conversation-dynamics thresholds.")
    weighted_escalation: list[tuple[float, float]] = []
    weighted_deescalation: list[tuple[float, float]] = []
    if arousal_change is not None:
        weighted_escalation.append((max(0.0, arousal_change) / 0.25, 0.6))
        weighted_deescalation.append((max(0.0, -arousal_change) / 0.25, 0.6))
    if valence_change is not None:
        weighted_escalation.append((max(0.0, -valence_change) / 0.25, 0.4))
        weighted_deescalation.append((max(0.0, valence_change) / 0.25, 0.4))
    def weighted_mean(values: list[tuple[float, float]]) -> float:
        if not values:
            return 0.0
        weight_sum = sum(weight for _, weight in values)
        return min(1.0, sum(value * weight for value, weight in values) / weight_sum)
    return weighted_mean(weighted_escalation), weighted_mean(weighted_deescalation)


def _event_type(
    escalation: float,
    deescalation: float,
    valence_change: float | None,
    arousal_change: float | None,
    config: ConversationDynamicsConfig,
) -> str:
    escalation_triggered = (
        (arousal_change is not None and arousal_change >= config.escalation_arousal_delta)
        or (valence_change is not None and valence_change <= config.escalation_valence_delta)
    )
    deescalation_triggered = (
        (arousal_change is not None and arousal_change <= config.deescalation_arousal_delta)
        or (valence_change is not None and valence_change >= config.deescalation_valence_delta)
    )
    if escalation_triggered and escalation >= deescalation + 0.04:
        return "escalation"
    if deescalation_triggered and deescalation >= escalation + 0.04:
        return "deescalation"
    if max(escalation, deescalation) < 0.10 and not escalation_triggered and not deescalation_triggered:
        return "stable"
    return "mixed_shift"


def _response_speed(gap_seconds: float, config: ConversationDynamicsConfig) -> str:
    if gap_seconds <= config.immediate_response_seconds:
        return "immediate"
    if gap_seconds <= config.quick_response_seconds:
        return "quick"
    return "delayed"


def _tension_score(events: list[ConversationDynamicsEvent]) -> float:
    if not events:
        return 0.0
    return float(sum(event.escalation_score - event.deescalation_score for event in events) / len(events))


def _dominant_pattern(events: list[ConversationDynamicsEvent], net_tension: float) -> str:
    if not events:
        return "insufficient_data"
    if net_tension >= 0.15:
        return "escalating"
    if net_tension <= -0.15:
        return "deescalating"
    types = {event.event_type for event in events}
    if types == {"stable"}:
        return "stable"
    if "escalation" in types and "deescalation" in types:
        return "mixed"
    return "stable"


def _pair_metrics(events: list[ConversationDynamicsEvent]) -> tuple[PairDynamics, ...]:
    grouped: dict[tuple[str, str], list[ConversationDynamicsEvent]] = {}
    for event in events:
        grouped.setdefault((event.from_speaker, event.to_speaker), []).append(event)
    result: list[PairDynamics] = []
    for pair in sorted(grouped):
        items = grouped[pair]
        alignments = [item.affect_alignment for item in items if item.affect_alignment is not None]
        valence = [item.valence_change for item in items if item.valence_change is not None]
        arousal = [item.arousal_change for item in items if item.arousal_change is not None]
        result.append(
            PairDynamics(
                from_speaker=pair[0],
                to_speaker=pair[1],
                response_count=len(items),
                escalation_count=sum(item.event_type == "escalation" for item in items),
                deescalation_count=sum(item.event_type == "deescalation" for item in items),
                mean_gap_seconds=float(_mean([item.gap_seconds for item in items]) or 0.0),
                median_gap_seconds=float(_median([item.gap_seconds for item in items]) or 0.0),
                mean_escalation_score=float(_mean([item.escalation_score for item in items]) or 0.0),
                mean_deescalation_score=float(_mean([item.deescalation_score for item in items]) or 0.0),
                affect_synchrony=_mean(alignments),
                mean_valence_change=_mean(valence),
                mean_arousal_change=_mean(arousal),
                mean_affect_magnitude=float(_mean([item.affect_magnitude for item in items]) or 0.0),
            )
        )
    return tuple(result)


class ConversationDynamicsAnalyzer:
    """Compute non-causal conversation-level affect dynamics from analyzed speaker turns."""

    def __init__(self, config: ConversationDynamicsConfig | None = None) -> None:
        self.config = config or ConversationDynamicsConfig()
        if self.config.escalation_arousal_delta <= 0 or self.config.deescalation_valence_delta <= 0:
            raise ConversationDynamicsError("Conversation-dynamics thresholds must be positive in magnitude.")

    def analyze(self, analysis: SpeakerAwareAnalysis) -> ConversationDynamics:
        turns = sorted(analysis.turns, key=lambda item: (item.start_seconds, item.end_seconds, item.turn_index))
        if not turns:
            raise ConversationDynamicsError("Cannot analyze conversation dynamics without speaker turns.")

        events: list[ConversationDynamicsEvent] = []
        all_state_distances: list[float] = []
        for previous, current in zip(turns, turns[1:], strict=False):
            previous_state = _state(previous)
            current_state = _state(current)
            distance = _distance(previous_state, current_state)
            if distance is not None:
                all_state_distances.append(distance)
            if previous.speaker == current.speaker:
                continue
            gap = max(0.0, current.start_seconds - previous.end_seconds)
            valence_change = None if previous.valence is None or current.valence is None else current.valence - previous.valence
            arousal_change = None if previous.arousal is None or current.arousal is None else current.arousal - previous.arousal
            dominance_change = None if previous.dominance is None or current.dominance is None else current.dominance - previous.dominance
            escalation, deescalation = _directional_scores(valence_change, arousal_change, self.config)
            event_type = _event_type(escalation, deescalation, valence_change, arousal_change, self.config)
            event = ConversationDynamicsEvent(
                index=len(events),
                at_seconds=current.start_seconds,
                from_speaker=previous.speaker,
                to_speaker=current.speaker,
                gap_seconds=gap,
                previous_emotion=previous.emotion,
                next_emotion=current.emotion,
                valence_change=valence_change,
                arousal_change=arousal_change,
                dominance_change=dominance_change,
                escalation_score=escalation,
                deescalation_score=deescalation,
                affect_magnitude=distance or 0.0,
                affect_alignment=(1.0 - distance) if distance is not None else None,
                event_type=event_type,
                response_speed=_response_speed(gap, self.config),
            )
            events.append(event)

        if not events:
            return ConversationDynamics(
                event_count=0,
                escalation_score=0.0,
                deescalation_score=0.0,
                net_tension=0.0,
                tension_trend=0.0,
                volatility=float(_mean(all_state_distances) or 0.0),
                stability_score=max(0.0, 1.0 - float(_mean(all_state_distances) or 0.0)),
                affect_synchrony=None,
                mean_response_gap_seconds=None,
                median_response_gap_seconds=None,
                p90_response_gap_seconds=None,
                fast_response_share=0.0,
                response_latency_shift_seconds=None,
                arousal_rise_events=0,
                valence_drop_events=0,
                escalation_events=0,
                deescalation_events=0,
                stable_events=0,
                dominant_pattern="insufficient_data",
                events=tuple(),
                pair_metrics=tuple(),
            )

        escalation_score = float(_mean([event.escalation_score for event in events]) or 0.0)
        deescalation_score = float(_mean([event.deescalation_score for event in events]) or 0.0)
        net_tension = max(-1.0, min(1.0, escalation_score - deescalation_score))
        midpoint = max(1, len(events) // 2)
        early_score = _tension_score(events[:midpoint])
        late_score = _tension_score(events[midpoint:]) if events[midpoint:] else early_score
        tension_trend = max(-1.0, min(1.0, late_score - early_score))

        gaps = [event.gap_seconds for event in events]
        fast_share = sum(event.response_speed in {"immediate", "quick"} for event in events) / len(events)
        early_gaps = gaps[:midpoint]
        late_gaps = gaps[midpoint:] if gaps[midpoint:] else early_gaps
        latency_shift = None
        early_median = _median(early_gaps)
        late_median = _median(late_gaps)
        if early_median is not None and late_median is not None:
            latency_shift = late_median - early_median

        alignments = [event.affect_alignment for event in events if event.affect_alignment is not None]
        volatility = float(_mean(all_state_distances) or 0.0)
        stability = max(0.0, min(1.0, 1.0 - volatility))

        return ConversationDynamics(
            event_count=len(events),
            escalation_score=escalation_score,
            deescalation_score=deescalation_score,
            net_tension=net_tension,
            tension_trend=tension_trend,
            volatility=volatility,
            stability_score=stability,
            affect_synchrony=_mean([float(item) for item in alignments]),
            mean_response_gap_seconds=float(_mean(gaps)) if gaps else None,
            median_response_gap_seconds=_median(gaps),
            p90_response_gap_seconds=_p90(gaps),
            fast_response_share=fast_share,
            response_latency_shift_seconds=latency_shift,
            arousal_rise_events=sum((event.arousal_change or 0.0) >= self.config.escalation_arousal_delta for event in events),
            valence_drop_events=sum((event.valence_change or 0.0) <= self.config.escalation_valence_delta for event in events),
            escalation_events=sum(event.event_type == "escalation" for event in events),
            deescalation_events=sum(event.event_type == "deescalation" for event in events),
            stable_events=sum(event.event_type == "stable" for event in events),
            dominant_pattern=_dominant_pattern(events, net_tension),
            events=tuple(events),
            pair_metrics=_pair_metrics(events),
        )
