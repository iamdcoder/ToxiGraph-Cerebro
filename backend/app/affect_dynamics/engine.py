from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.dimensional_emotion.interpretation import interpret_affect
from app.temporal.engine import TemporalConfig, _speech_mask, _window_bounds, _window_speech_ratio


class AffectDynamicsError(ValueError):
    """Raised when continuous affect dynamics cannot be computed."""


@dataclass(frozen=True)
class AffectDynamicsConfig:
    window_seconds: float = 3.0
    hop_seconds: float = 1.0
    min_window_seconds: float = 0.8
    min_speech_ratio: float = 0.35
    frame_length_ms: float = 25.0
    hop_length_ms: float = 10.0
    smoothing_alpha: float = 0.35
    event_delta_threshold: float = 0.08
    shift_distance_threshold: float = 0.12


@dataclass(frozen=True)
class AffectPoint:
    index: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    speech_ratio: float
    is_speech: bool
    raw: dict[str, float]
    smoothed: dict[str, float]
    deltas: dict[str, float]
    affect_shift: float
    velocity: float
    quadrant: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start_seconds": round(self.start_seconds, 4),
            "end_seconds": round(self.end_seconds, 4),
            "duration_seconds": round(self.duration_seconds, 4),
            "speech_ratio": round(self.speech_ratio, 4),
            "is_speech": self.is_speech,
            "raw": {key: round(value, 6) for key, value in self.raw.items()},
            "smoothed": {key: round(value, 6) for key, value in self.smoothed.items()},
            "deltas": {key: round(value, 6) for key, value in self.deltas.items()},
            "affect_shift": round(self.affect_shift, 6),
            "velocity": round(self.velocity, 6),
            "quadrant": self.quadrant,
        }


@dataclass(frozen=True)
class AffectEvent:
    event_type: str
    at_seconds: float
    score: float
    dimension: str | None
    from_value: float | None
    to_value: float | None
    source_point_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "at_seconds": round(self.at_seconds, 4),
            "score": round(self.score, 6),
            "dimension": self.dimension,
            "from_value": round(self.from_value, 6) if self.from_value is not None else None,
            "to_value": round(self.to_value, 6) if self.to_value is not None else None,
            "source_point_index": self.source_point_index,
        }


@dataclass(frozen=True)
class AffectDynamics:
    duration_seconds: float
    sample_rate: int
    window_seconds: float
    hop_seconds: float
    smoothing_alpha: float
    points: tuple[AffectPoint, ...]
    events: tuple[AffectEvent, ...]
    start_state: dict[str, float] | None
    end_state: dict[str, float] | None
    net_change: dict[str, float]
    active_speech_seconds: float
    speech_coverage: float
    mean_velocity: float
    peak_velocity: float
    total_path_length: float
    volatility: float
    stability_score: float
    largest_shift: float
    peak_arousal: float | None
    peak_arousal_at_seconds: float | None
    lowest_valence: float | None
    lowest_valence_at_seconds: float | None
    dominant_quadrant: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_seconds": round(self.duration_seconds, 4),
            "sample_rate": self.sample_rate,
            "window_seconds": self.window_seconds,
            "hop_seconds": self.hop_seconds,
            "smoothing_alpha": self.smoothing_alpha,
            "points": [point.to_dict() for point in self.points],
            "events": [event.to_dict() for event in self.events],
            "start_state": _round_state(self.start_state),
            "end_state": _round_state(self.end_state),
            "net_change": {key: round(value, 6) for key, value in self.net_change.items()},
            "active_speech_seconds": round(self.active_speech_seconds, 4),
            "speech_coverage": round(self.speech_coverage, 6),
            "mean_velocity": round(self.mean_velocity, 6),
            "peak_velocity": round(self.peak_velocity, 6),
            "total_path_length": round(self.total_path_length, 6),
            "volatility": round(self.volatility, 6),
            "stability_score": round(self.stability_score, 6),
            "largest_shift": round(self.largest_shift, 6),
            "peak_arousal": round(self.peak_arousal, 6) if self.peak_arousal is not None else None,
            "peak_arousal_at_seconds": round(self.peak_arousal_at_seconds, 4)
            if self.peak_arousal_at_seconds is not None
            else None,
            "lowest_valence": round(self.lowest_valence, 6) if self.lowest_valence is not None else None,
            "lowest_valence_at_seconds": round(self.lowest_valence_at_seconds, 4)
            if self.lowest_valence_at_seconds is not None
            else None,
            "dominant_quadrant": self.dominant_quadrant,
        }


def _round_state(state: dict[str, float] | None) -> dict[str, float] | None:
    if state is None:
        return None
    return {key: round(value, 6) for key, value in state.items()}


def _validate_config(config: AffectDynamicsConfig) -> None:
    if config.window_seconds <= 0:
        raise AffectDynamicsError("Affect-dynamics window duration must be positive.")
    if config.hop_seconds <= 0 or config.hop_seconds > config.window_seconds:
        raise AffectDynamicsError("Affect-dynamics hop duration must be positive and cannot exceed the window.")
    if config.min_window_seconds <= 0 or config.min_window_seconds > config.window_seconds:
        raise AffectDynamicsError("Affect-dynamics minimum window duration is invalid.")
    if not 0 < config.min_speech_ratio <= 1:
        raise AffectDynamicsError("Affect-dynamics speech ratio must be between 0 and 1.")
    if not 0 < config.smoothing_alpha <= 1:
        raise AffectDynamicsError("Affect-dynamics smoothing alpha must be between 0 and 1.")
    if config.event_delta_threshold <= 0 or config.shift_distance_threshold <= 0:
        raise AffectDynamicsError("Affect-dynamics event thresholds must be positive.")


def _as_unit_state(values: Any) -> dict[str, float]:
    try:
        state = {
            "valence": float(values["valence"]),
            "arousal": float(values["arousal"]),
            "dominance": float(values["dominance"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise AffectDynamicsError("Dimensional model output is missing V/A/D values.") from exc
    for key, value in state.items():
        if not math.isfinite(value) or value < 0 or value > 1:
            raise AffectDynamicsError(f"Dimensional model returned invalid {key} value.")
    return state


def _normalized_distance(first: dict[str, float], second: dict[str, float]) -> float:
    distance = math.sqrt(sum((second[key] - first[key]) ** 2 for key in first))
    return float(distance / math.sqrt(3.0))


def _quadrant(state: dict[str, float]) -> str:
    return interpret_affect(state["valence"], state["arousal"], state["dominance"]).quadrant


def _dominant_quadrant(points: list[AffectPoint]) -> str | None:
    weights: dict[str, float] = {}
    for point in points:
        if not point.is_speech or point.quadrant is None:
            continue
        weights[point.quadrant] = weights.get(point.quadrant, 0.0) + point.duration_seconds
    if not weights:
        return None
    return max(weights, key=weights.get)


class AffectDynamicsAnalyzer:
    """Temporal valence/arousal/dominance analysis over overlapping speech windows."""

    def __init__(self, config: AffectDynamicsConfig | None = None) -> None:
        self.config = config or AffectDynamicsConfig()
        _validate_config(self.config)

    def analyze(self, samples: np.ndarray, sample_rate: int, *, dimensional_model) -> AffectDynamics:
        array = np.asarray(samples, dtype=np.float32).reshape(-1)
        if array.size == 0:
            raise AffectDynamicsError("Cannot analyze an empty audio signal.")
        if sample_rate <= 0:
            raise AffectDynamicsError("Sample rate must be positive.")

        duration = len(array) / float(sample_rate)
        temporal_config = TemporalConfig(
            window_seconds=self.config.window_seconds,
            hop_seconds=self.config.hop_seconds,
            min_window_seconds=self.config.min_window_seconds,
            min_speech_ratio=self.config.min_speech_ratio,
            frame_length_ms=self.config.frame_length_ms,
            hop_length_ms=self.config.hop_length_ms,
            smoothing_alpha=self.config.smoothing_alpha,
        )
        speech_mask, _ = _speech_mask(array, sample_rate, temporal_config)
        bounds = _window_bounds(len(array), sample_rate, temporal_config)

        points: list[AffectPoint] = []
        previous_smoothed: dict[str, float] | None = None
        previous_speech_start: float | None = None
        events: list[AffectEvent] = []

        for index, (start, end) in enumerate(bounds):
            ratio = _window_speech_ratio(
                speech_mask,
                start_sample=start,
                end_sample=end,
                sample_rate=sample_rate,
                frame_hop_ms=self.config.hop_length_ms,
            )
            start_seconds = start / sample_rate
            end_seconds = end / sample_rate
            duration_seconds = end_seconds - start_seconds
            if ratio < self.config.min_speech_ratio:
                points.append(
                    AffectPoint(
                        index=index,
                        start_seconds=start_seconds,
                        end_seconds=end_seconds,
                        duration_seconds=duration_seconds,
                        speech_ratio=ratio,
                        is_speech=False,
                        raw={},
                        smoothed={},
                        deltas={},
                        affect_shift=0.0,
                        velocity=0.0,
                        quadrant=None,
                    )
                )
                continue

            try:
                prediction = dimensional_model.predict(array[start:end], sample_rate)
                raw_state = _as_unit_state(prediction.values)
            except Exception as exc:
                raise AffectDynamicsError(
                    f"Dimensional affect dynamics inference failed for window {index}: {exc}"
                ) from exc

            if previous_smoothed is None:
                smoothed = dict(raw_state)
                deltas = {key: 0.0 for key in raw_state}
                shift = 0.0
                velocity = 0.0
            else:
                alpha = self.config.smoothing_alpha
                smoothed = {
                    key: alpha * raw_state[key] + (1.0 - alpha) * previous_smoothed[key]
                    for key in raw_state
                }
                deltas = {key: smoothed[key] - previous_smoothed[key] for key in smoothed}
                shift = _normalized_distance(previous_smoothed, smoothed)
                step_seconds = max(start_seconds - (previous_speech_start or start_seconds), 1e-6)
                velocity = shift / step_seconds

                for dimension in ("valence", "arousal", "dominance"):
                    delta = deltas[dimension]
                    if delta >= self.config.event_delta_threshold:
                        events.append(
                            AffectEvent(
                                event_type="dimension_rise",
                                at_seconds=start_seconds,
                                score=min(1.0, delta / max(self.config.event_delta_threshold * 2.0, 1e-6)),
                                dimension=dimension,
                                from_value=previous_smoothed[dimension],
                                to_value=smoothed[dimension],
                                source_point_index=index,
                            )
                        )
                    elif delta <= -self.config.event_delta_threshold:
                        events.append(
                            AffectEvent(
                                event_type="dimension_fall",
                                at_seconds=start_seconds,
                                score=min(1.0, abs(delta) / max(self.config.event_delta_threshold * 2.0, 1e-6)),
                                dimension=dimension,
                                from_value=previous_smoothed[dimension],
                                to_value=smoothed[dimension],
                                source_point_index=index,
                            )
                        )

                if shift >= self.config.shift_distance_threshold:
                    events.append(
                        AffectEvent(
                            event_type="affect_shift",
                            at_seconds=start_seconds,
                            score=min(1.0, shift / max(self.config.shift_distance_threshold * 2.0, 1e-6)),
                            dimension=None,
                            from_value=None,
                            to_value=None,
                            source_point_index=index,
                        )
                    )

                previous_quadrant = _quadrant(previous_smoothed)
                current_quadrant = _quadrant(smoothed)
                if current_quadrant != previous_quadrant:
                    events.append(
                        AffectEvent(
                            event_type="quadrant_transition",
                            at_seconds=start_seconds,
                            score=min(1.0, max(shift, self.config.shift_distance_threshold) / max(self.config.shift_distance_threshold * 2.0, 1e-6)),
                            dimension=None,
                            from_value=None,
                            to_value=None,
                            source_point_index=index,
                        )
                    )

            point = AffectPoint(
                index=index,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
                duration_seconds=duration_seconds,
                speech_ratio=ratio,
                is_speech=True,
                raw=raw_state,
                smoothed=smoothed,
                deltas=deltas,
                affect_shift=shift,
                velocity=velocity,
                quadrant=_quadrant(smoothed),
            )
            points.append(point)
            previous_smoothed = smoothed
            previous_speech_start = start_seconds

        speech_points = [point for point in points if point.is_speech]
        frame_hop_seconds = self.config.hop_length_ms / 1000.0
        active_speech_seconds = min(duration, float(np.sum(speech_mask)) * frame_hop_seconds)
        speech_coverage = min(1.0, active_speech_seconds / max(duration, 1e-12))

        if speech_points:
            start_state = dict(speech_points[0].smoothed)
            end_state = dict(speech_points[-1].smoothed)
            net_change = {key: end_state[key] - start_state[key] for key in start_state}
            velocities = [point.velocity for point in speech_points if point.velocity > 0]
            mean_velocity = float(np.mean(velocities)) if velocities else 0.0
            peak_velocity = float(max(velocities, default=0.0))
            shifts = [point.affect_shift for point in speech_points if point.affect_shift > 0]
            total_path_length = float(sum(shifts))
            volatility = float(np.mean(shifts)) if shifts else 0.0
            stability_score = float(max(0.0, min(1.0, 1.0 - volatility / 0.20)))
            largest_shift = float(max(shifts, default=0.0))
            peak_point = max(speech_points, key=lambda point: point.smoothed["arousal"])
            low_valence_point = min(speech_points, key=lambda point: point.smoothed["valence"])
            peak_arousal = float(peak_point.smoothed["arousal"])
            peak_arousal_at_seconds = float(peak_point.start_seconds)
            lowest_valence = float(low_valence_point.smoothed["valence"])
            lowest_valence_at_seconds = float(low_valence_point.start_seconds)
        else:
            start_state = None
            end_state = None
            net_change = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
            mean_velocity = 0.0
            peak_velocity = 0.0
            total_path_length = 0.0
            volatility = 0.0
            stability_score = 0.0
            largest_shift = 0.0
            peak_arousal = None
            peak_arousal_at_seconds = None
            lowest_valence = None
            lowest_valence_at_seconds = None

        return AffectDynamics(
            duration_seconds=duration,
            sample_rate=int(sample_rate),
            window_seconds=self.config.window_seconds,
            hop_seconds=self.config.hop_seconds,
            smoothing_alpha=self.config.smoothing_alpha,
            points=tuple(points),
            events=tuple(events),
            start_state=start_state,
            end_state=end_state,
            net_change=net_change,
            active_speech_seconds=active_speech_seconds,
            speech_coverage=speech_coverage,
            mean_velocity=mean_velocity,
            peak_velocity=peak_velocity,
            total_path_length=total_path_length,
            volatility=volatility,
            stability_score=stability_score,
            largest_shift=largest_shift,
            peak_arousal=peak_arousal,
            peak_arousal_at_seconds=peak_arousal_at_seconds,
            lowest_valence=lowest_valence,
            lowest_valence_at_seconds=lowest_valence_at_seconds,
            dominant_quadrant=_dominant_quadrant(speech_points),
        )
