from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from app.conversation_graph.dynamics import ConversationDynamics


class ConversationStateError(ValueError):
    """Raised when the conversation state machine cannot be evaluated."""


STATE_INSUFFICIENT = "insufficient_data"
STATE_CALM = "calm"
STATE_ENGAGED = "engaged"
STATE_STABLE = "stable"
STATE_TENSE = "tense"
STATE_ESCALATING = "escalating"
STATE_PEAK_TENSION = "peak_tension"
STATE_COOLING = "cooling"
STATE_DEESCALATING = "deescalating"


@dataclass(frozen=True)
class ConversationStateConfig:
    """Deterministic thresholds for the higher-level conversation state machine."""

    smoothing: float = 0.35
    calm_upper: float = 0.14
    engaged_upper: float = 0.28
    stable_upper: float = 0.42
    tense_upper: float = 0.56
    escalating_upper: float = 0.76
    peak_threshold: float = 0.76
    state_change_delta: float = 0.06
    cooling_score: float = 0.07
    strong_affect: float = 0.14
    strong_signal: float = 0.15


@dataclass(frozen=True)
class ConversationStatePoint:
    index: int
    at_seconds: float
    from_speaker: str
    to_speaker: str
    previous_state: str
    state: str
    tension_score: float
    tension_change: float
    directional_signal: float
    affect_magnitude: float
    response_speed: str
    event_type: str
    emotion_transition: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "at_seconds": round(self.at_seconds, 4),
            "from_speaker": self.from_speaker,
            "to_speaker": self.to_speaker,
            "previous_state": self.previous_state,
            "state": self.state,
            "tension_score": round(self.tension_score, 6),
            "tension_change": round(self.tension_change, 6),
            "directional_signal": round(self.directional_signal, 6),
            "affect_magnitude": round(self.affect_magnitude, 6),
            "response_speed": self.response_speed,
            "event_type": self.event_type,
            "emotion_transition": self.emotion_transition,
            "confidence": round(self.confidence, 6),
        }


@dataclass(frozen=True)
class ConversationStateTransition:
    index: int
    at_seconds: float
    from_state: str
    to_state: str
    trigger: str
    tension_score: float
    tension_change: float
    speaker: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "at_seconds": round(self.at_seconds, 4),
            "from_state": self.from_state,
            "to_state": self.to_state,
            "trigger": self.trigger,
            "tension_score": round(self.tension_score, 6),
            "tension_change": round(self.tension_change, 6),
            "speaker": self.speaker,
        }


@dataclass(frozen=True)
class ConversationStateAnalysis:
    state_count: int
    transition_count: int
    start_state: str
    end_state: str
    dominant_state: str
    peak_state: str
    peak_tension: float
    peak_tension_at_seconds: float | None
    duration_by_state: dict[str, float]
    state_distribution: dict[str, float]
    points: tuple[ConversationStatePoint, ...]
    transitions: tuple[ConversationStateTransition, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_count": self.state_count,
            "transition_count": self.transition_count,
            "start_state": self.start_state,
            "end_state": self.end_state,
            "dominant_state": self.dominant_state,
            "peak_state": self.peak_state,
            "peak_tension": round(self.peak_tension, 6),
            "peak_tension_at_seconds": (
                round(self.peak_tension_at_seconds, 4)
                if self.peak_tension_at_seconds is not None
                else None
            ),
            "duration_by_state": {
                key: round(value, 4)
                for key, value in self.duration_by_state.items()
            },
            "state_distribution": {
                key: round(value, 6)
                for key, value in self.state_distribution.items()
            },
            "points": [item.to_dict() for item in self.points],
            "transitions": [item.to_dict() for item in self.transitions],
            "semantics": {
                "state": "An interpretable state-machine label derived from observed cross-speaker affect direction and interaction behavior.",
                "tension_score": "A bounded latent index summarizing cumulative observed escalation pressure; it is not a psychological measurement.",
                "transition": "A state-machine change triggered by observed turn-level evidence with hysteresis to reduce one-event flicker.",
                "duration": "Estimated occupancy between observed cross-speaker event timestamps, not a claim about continuous internal state.",
            },
        }


def _clip(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, float(value)))


def _finite(value: float | None) -> float:
    if value is None:
        return 0.0
    number = float(value)
    return number if math.isfinite(number) else 0.0


def _directional_signal(event: Any) -> float:
    return _clip(float(event.escalation_score) - float(event.deescalation_score), -1.0, 1.0)


def _confidence(
    tension_score: float,
    directional_signal: float,
    state: str,
    config: ConversationStateConfig,
) -> float:
    boundary_distance = min(
        abs(tension_score - config.calm_upper),
        abs(tension_score - config.engaged_upper),
        abs(tension_score - config.stable_upper),
        abs(tension_score - config.tense_upper),
        abs(tension_score - config.peak_threshold),
    )
    boundary_confidence = _clip(boundary_distance / 0.18)
    signal_confidence = _clip(abs(directional_signal) / 0.35)
    stability_bonus = 0.10 if state in {STATE_CALM, STATE_STABLE, STATE_ENGAGED} and abs(directional_signal) < 0.06 else 0.0
    return _clip(0.50 * boundary_confidence + 0.40 * signal_confidence + stability_bonus)


def _base_state(tension_score: float, event: Any, config: ConversationStateConfig) -> str:
    if tension_score < config.calm_upper:
        return STATE_CALM
    if tension_score < config.engaged_upper:
        if event.response_speed in {"immediate", "quick"} or event.affect_magnitude >= config.strong_affect:
            return STATE_ENGAGED
        return STATE_STABLE
    if tension_score < config.stable_upper:
        return STATE_STABLE
    if tension_score < config.tense_upper:
        return STATE_TENSE
    if tension_score < config.escalating_upper:
        return STATE_ESCALATING
    return STATE_PEAK_TENSION


def _state_for_event(
    tension_score: float,
    previous_tension: float,
    previous_state: str,
    event: Any,
    directional_signal: float,
    config: ConversationStateConfig,
) -> str:
    change = tension_score - previous_tension

    if tension_score >= config.peak_threshold:
        if directional_signal <= -config.cooling_score and change < 0:
            return STATE_COOLING
        return STATE_PEAK_TENSION

    if previous_state == STATE_PEAK_TENSION:
        if directional_signal <= -config.cooling_score and change <= -config.state_change_delta / 2:
            return STATE_COOLING
        return STATE_PEAK_TENSION if tension_score >= config.tense_upper else STATE_ESCALATING

    if previous_state == STATE_ESCALATING and directional_signal <= -config.cooling_score and change <= 0:
        return STATE_COOLING

    if previous_state == STATE_COOLING:
        if directional_signal >= config.strong_signal and change >= config.state_change_delta / 2:
            return STATE_ESCALATING if tension_score >= config.tense_upper else STATE_TENSE
        if tension_score <= config.engaged_upper:
            return STATE_DEESCALATING
        return STATE_COOLING

    if previous_state == STATE_DEESCALATING:
        if directional_signal >= config.strong_signal and change >= config.state_change_delta:
            return STATE_ESCALATING if tension_score >= config.tense_upper else STATE_TENSE
        if tension_score <= config.calm_upper:
            return STATE_CALM
        return STATE_DEESCALATING

    if directional_signal >= config.strong_signal and change >= config.state_change_delta:
        if tension_score >= config.peak_threshold:
            return STATE_PEAK_TENSION
        if tension_score >= config.tense_upper:
            return STATE_ESCALATING

    if directional_signal <= -config.strong_signal and change <= -config.state_change_delta:
        if tension_score <= config.engaged_upper:
            return STATE_DEESCALATING
        return STATE_COOLING

    return _base_state(tension_score, event, config)


def _transition_trigger(previous: str, current: str, event: Any, directional_signal: float) -> str:
    if current == STATE_COOLING:
        return "negative_directional_shift_after_elevated_tension"
    if current == STATE_DEESCALATING:
        return "sustained_deescalation"
    if current == STATE_ESCALATING:
        return "sustained_escalation"
    if current == STATE_PEAK_TENSION:
        return "high_cumulative_tension"
    if previous == STATE_CALM and current == STATE_ENGAGED:
        return "increased_interaction_activation"
    if event.response_speed in {"immediate", "quick"} and abs(directional_signal) < 0.06:
        return "rapid_stable_exchange"
    return "threshold_crossing"


def _dominant_state(duration_by_state: dict[str, float]) -> str:
    eligible = {key: value for key, value in duration_by_state.items() if value > 0}
    if not eligible:
        return STATE_INSUFFICIENT
    return max(eligible.items(), key=lambda item: (item[1], item[0]))[0]


class ConversationStateAnalyzer:
    """Convert turn-level conversation dynamics into a hysteresis-aware state sequence."""

    def __init__(self, config: ConversationStateConfig | None = None) -> None:
        self.config = config or ConversationStateConfig()
        if not 0.0 < self.config.smoothing <= 1.0:
            raise ConversationStateError("State smoothing must be in (0, 1].")
        thresholds = [
            self.config.calm_upper,
            self.config.engaged_upper,
            self.config.stable_upper,
            self.config.tense_upper,
            self.config.escalating_upper,
            self.config.peak_threshold,
        ]
        if any(value < 0.0 or value > 1.0 for value in thresholds):
            raise ConversationStateError("Conversation-state thresholds must be within [0, 1].")
        if thresholds != sorted(thresholds):
            raise ConversationStateError("Conversation-state thresholds must be non-decreasing.")
        if self.config.state_change_delta <= 0 or self.config.cooling_score <= 0 or self.config.strong_signal <= 0:
            raise ConversationStateError("Conversation-state transition thresholds must be positive.")

    def analyze(self, dynamics: ConversationDynamics, duration_seconds: float | None = None) -> ConversationStateAnalysis:
        events = list(dynamics.events)
        if not events:
            return ConversationStateAnalysis(
                state_count=0,
                transition_count=0,
                start_state=STATE_INSUFFICIENT,
                end_state=STATE_INSUFFICIENT,
                dominant_state=STATE_INSUFFICIENT,
                peak_state=STATE_INSUFFICIENT,
                peak_tension=0.0,
                peak_tension_at_seconds=None,
                duration_by_state={},
                state_distribution={},
                points=tuple(),
                transitions=tuple(),
            )

        tension_score = 0.0
        previous_tension = 0.0
        previous_state = STATE_CALM
        points: list[ConversationStatePoint] = []
        transitions: list[ConversationStateTransition] = []

        for index, event in enumerate(events):
            signal = _directional_signal(event)
            target = _clip(tension_score + signal)
            tension_score = _clip(
                (1.0 - self.config.smoothing) * tension_score
                + self.config.smoothing * target
            )
            state = _state_for_event(
                tension_score,
                previous_tension,
                previous_state,
                event,
                signal,
                self.config,
            )
            confidence = _confidence(tension_score, signal, state, self.config)
            emotion_transition = f"{event.previous_emotion} → {event.next_emotion}"
            point = ConversationStatePoint(
                index=index,
                at_seconds=float(event.at_seconds),
                from_speaker=event.from_speaker,
                to_speaker=event.to_speaker,
                previous_state=previous_state,
                state=state,
                tension_score=tension_score,
                tension_change=tension_score - previous_tension,
                directional_signal=signal,
                affect_magnitude=float(event.affect_magnitude),
                response_speed=event.response_speed,
                event_type=event.event_type,
                emotion_transition=emotion_transition,
                confidence=confidence,
            )
            points.append(point)

            if state != previous_state:
                transitions.append(
                    ConversationStateTransition(
                        index=index,
                        at_seconds=float(event.at_seconds),
                        from_state=previous_state,
                        to_state=state,
                        trigger=_transition_trigger(previous_state, state, event, signal),
                        tension_score=tension_score,
                        tension_change=tension_score - previous_tension,
                        speaker=event.to_speaker,
                    )
                )

            previous_tension = tension_score
            previous_state = state

        duration_by_state: dict[str, float] = {}
        resolved_duration = float(duration_seconds) if duration_seconds is not None else None
        for index, point in enumerate(points):
            if index + 1 < len(points):
                next_time = points[index + 1].at_seconds
            elif resolved_duration is not None:
                next_time = max(point.at_seconds, resolved_duration)
            else:
                next_time = max(
                    float(dynamics.events[-1].at_seconds),
                    float(dynamics.events[-1].at_seconds + dynamics.events[-1].gap_seconds),
                )
            duration = max(0.0, next_time - point.at_seconds)
            duration_by_state[point.state] = duration_by_state.get(point.state, 0.0) + duration

        if not any(duration_by_state.values()):
            # A single response boundary has no interval after it; give the observed state
            # an occupancy of one unit so summaries remain useful without implying real seconds.
            duration_by_state[points[-1].state] = 1.0

        duration_total = sum(duration_by_state.values())
        state_distribution = {
            key: value / duration_total
            for key, value in duration_by_state.items()
            if duration_total > 0
        }
        peak_point = max(points, key=lambda item: (item.tension_score, item.at_seconds))

        return ConversationStateAnalysis(
            state_count=len(points),
            transition_count=len(transitions),
            start_state=points[0].state,
            end_state=points[-1].state,
            dominant_state=_dominant_state(duration_by_state),
            peak_state=peak_point.state,
            peak_tension=peak_point.tension_score,
            peak_tension_at_seconds=peak_point.at_seconds,
            duration_by_state=duration_by_state,
            state_distribution=state_distribution,
            points=tuple(points),
            transitions=tuple(transitions),
        )
