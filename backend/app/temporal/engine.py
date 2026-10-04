from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from app.features.acoustic import extract_acoustic_features
from app.fusion.engine import FusionEngine


class TemporalAnalysisError(ValueError):
    """Raised when temporal speech-emotion analysis cannot be performed."""


@dataclass(frozen=True)
class TemporalConfig:
    window_seconds: float = 3.0
    hop_seconds: float = 1.0
    min_window_seconds: float = 0.8
    min_speech_ratio: float = 0.35
    frame_length_ms: float = 25.0
    hop_length_ms: float = 10.0
    min_voiced_rms: float = 0.005
    smoothing_alpha: float = 0.35
    transition_js_threshold: float = 0.08


@dataclass(frozen=True)
class TemporalSegment:
    index: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    speech_ratio: float
    is_speech: bool
    raw_emotion: str | None
    raw_confidence: float
    emotion: str | None
    confidence: float
    probabilities: dict[str, float]
    agreement: str | None
    js_divergence: float | None
    transition: bool
    transition_score: float

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "start_seconds": round(self.start_seconds, 4),
            "end_seconds": round(self.end_seconds, 4),
            "duration_seconds": round(self.duration_seconds, 4),
            "speech_ratio": round(self.speech_ratio, 4),
            "is_speech": self.is_speech,
            "raw_emotion": self.raw_emotion,
            "raw_confidence": round(self.raw_confidence, 6),
            "emotion": self.emotion,
            "confidence": round(self.confidence, 6),
            "probabilities": {
                label: round(value, 6) for label, value in self.probabilities.items()
            },
            "agreement": self.agreement,
            "js_divergence": (
                round(self.js_divergence, 6)
                if self.js_divergence is not None
                else None
            ),
            "transition": self.transition,
            "transition_score": round(self.transition_score, 6),
        }


@dataclass(frozen=True)
class EmotionRun:
    emotion: str
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    segment_count: int
    mean_confidence: float

    def to_dict(self) -> dict:
        return {
            "emotion": self.emotion,
            "start_seconds": round(self.start_seconds, 4),
            "end_seconds": round(self.end_seconds, 4),
            "duration_seconds": round(self.duration_seconds, 4),
            "segment_count": self.segment_count,
            "mean_confidence": round(self.mean_confidence, 6),
        }


@dataclass(frozen=True)
class TrajectoryPoint:
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    is_speech: bool
    emotion: str | None
    confidence: float
    speech_ratio: float
    probabilities: dict[str, float]
    covered_windows: int

    def to_dict(self) -> dict:
        return {
            "start_seconds": round(self.start_seconds, 4),
            "end_seconds": round(self.end_seconds, 4),
            "duration_seconds": round(self.duration_seconds, 4),
            "is_speech": self.is_speech,
            "emotion": self.emotion,
            "confidence": round(self.confidence, 6),
            "speech_ratio": round(self.speech_ratio, 6),
            "probabilities": {label: round(value, 6) for label, value in self.probabilities.items()},
            "covered_windows": self.covered_windows,
        }


@dataclass(frozen=True)
class EmotionTransition:
    from_emotion: str
    to_emotion: str
    at_seconds: float
    score: float
    source_segment_index: int

    def to_dict(self) -> dict:
        return {
            "from_emotion": self.from_emotion,
            "to_emotion": self.to_emotion,
            "at_seconds": round(self.at_seconds, 4),
            "score": round(self.score, 6),
            "source_segment_index": self.source_segment_index,
        }


@dataclass(frozen=True)
class TemporalAnalysis:
    duration_seconds: float
    sample_rate: int
    window_seconds: float
    hop_seconds: float
    smoothing_alpha: float
    segments: tuple[TemporalSegment, ...]
    trajectory: tuple[TrajectoryPoint, ...]
    emotion_runs: tuple[EmotionRun, ...]
    transitions: tuple[EmotionTransition, ...]
    dominant_emotion: str | None
    dominant_confidence: float
    active_speech_seconds: float
    speech_coverage: float
    analyzed_windows: int
    speech_windows: int

    def to_dict(self) -> dict:
        return {
            "duration_seconds": round(self.duration_seconds, 4),
            "sample_rate": self.sample_rate,
            "window_seconds": self.window_seconds,
            "hop_seconds": self.hop_seconds,
            "smoothing_alpha": self.smoothing_alpha,
            "segments": [segment.to_dict() for segment in self.segments],
            "trajectory": [point.to_dict() for point in self.trajectory],
            "emotion_runs": [run.to_dict() for run in self.emotion_runs],
            "transitions": [transition.to_dict() for transition in self.transitions],
            "dominant_emotion": self.dominant_emotion,
            "dominant_confidence": round(self.dominant_confidence, 6),
            "active_speech_seconds": round(self.active_speech_seconds, 4),
            "speech_coverage": round(self.speech_coverage, 6),
            "analyzed_windows": self.analyzed_windows,
            "speech_windows": self.speech_windows,
        }


def _validate_config(config: TemporalConfig) -> None:
    if config.window_seconds <= 0:
        raise TemporalAnalysisError("Temporal window duration must be positive.")
    if config.hop_seconds <= 0:
        raise TemporalAnalysisError("Temporal hop duration must be positive.")
    if config.hop_seconds > config.window_seconds:
        raise TemporalAnalysisError("Temporal hop cannot exceed the window duration.")
    if config.min_window_seconds <= 0 or config.min_window_seconds > config.window_seconds:
        raise TemporalAnalysisError("Temporal minimum window duration is invalid.")
    if not 0 < config.min_speech_ratio <= 1:
        raise TemporalAnalysisError("Minimum speech ratio must be between 0 and 1.")
    if not 0 < config.smoothing_alpha <= 1:
        raise TemporalAnalysisError("Temporal smoothing alpha must be between 0 and 1.")
    if config.transition_js_threshold < 0:
        raise TemporalAnalysisError("Temporal transition threshold cannot be negative.")


def _frame_rms(samples: np.ndarray, sample_rate: int, frame_length_ms: float, hop_length_ms: float) -> np.ndarray:
    frame_length = max(1, int(round(sample_rate * frame_length_ms / 1000.0)))
    hop_length = max(1, int(round(sample_rate * hop_length_ms / 1000.0)))
    array = np.asarray(samples, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return np.zeros(0, dtype=np.float64)
    frame_count = max(1, 1 + int(math.ceil(max(0, len(array) - frame_length) / hop_length)))
    padded_length = (frame_count - 1) * hop_length + frame_length
    padded = np.pad(array.astype(np.float64), (0, max(0, padded_length - len(array))))
    rms = []
    for start in range(0, padded_length - frame_length + 1, hop_length):
        frame = padded[start : start + frame_length]
        rms.append(float(np.sqrt(np.mean(frame * frame))))
    return np.asarray(rms, dtype=np.float64)


def _speech_mask(samples: np.ndarray, sample_rate: int, config: TemporalConfig) -> tuple[np.ndarray, float]:
    rms = _frame_rms(samples, sample_rate, config.frame_length_ms, config.hop_length_ms)
    if rms.size == 0:
        return rms.astype(bool), config.min_voiced_rms

    noise_floor = float(np.percentile(rms, 20))
    signal_peak = float(np.percentile(rms, 95))
    if signal_peak <= config.min_voiced_rms:
        return np.zeros_like(rms, dtype=bool), config.min_voiced_rms

    dynamic_range = max(0.0, signal_peak - noise_floor)
    threshold = max(
        config.min_voiced_rms,
        noise_floor + dynamic_range * 0.15 if dynamic_range else signal_peak * 0.25,
    )
    return rms >= threshold, threshold


def _window_bounds(length: int, sample_rate: int, config: TemporalConfig) -> list[tuple[int, int]]:
    window_samples = max(1, int(round(config.window_seconds * sample_rate)))
    hop_samples = max(1, int(round(config.hop_seconds * sample_rate)))
    min_samples = max(1, int(round(config.min_window_seconds * sample_rate)))

    if length <= window_samples:
        return [(0, length)]

    bounds: list[tuple[int, int]] = []
    start = 0
    while start < length:
        end = min(start + window_samples, length)
        if end - start >= min_samples:
            bounds.append((start, end))
        if end >= length:
            break
        start += hop_samples
    return bounds


def _window_speech_ratio(
    speech_mask: np.ndarray,
    *,
    start_sample: int,
    end_sample: int,
    sample_rate: int,
    frame_hop_ms: float,
) -> float:
    if speech_mask.size == 0:
        return 0.0
    frame_hop = max(1, int(round(sample_rate * frame_hop_ms / 1000.0)))
    start_frame = max(0, int(start_sample // frame_hop))
    end_frame = min(len(speech_mask), int(math.ceil(end_sample / frame_hop)))
    if end_frame <= start_frame:
        return 0.0
    return float(np.mean(speech_mask[start_frame:end_frame]))


def _js_divergence(first: Mapping[str, float], second: Mapping[str, float], labels: tuple[str, ...]) -> float:
    first_vector = np.asarray([max(0.0, float(first.get(label, 0.0))) for label in labels], dtype=np.float64)
    second_vector = np.asarray([max(0.0, float(second.get(label, 0.0))) for label in labels], dtype=np.float64)
    first_vector /= max(float(first_vector.sum()), 1e-12)
    second_vector /= max(float(second_vector.sum()), 1e-12)
    midpoint = 0.5 * (first_vector + second_vector)
    safe_first = np.clip(first_vector, 1e-12, 1.0)
    safe_second = np.clip(second_vector, 1e-12, 1.0)
    safe_midpoint = np.clip(midpoint, 1e-12, 1.0)
    return float(
        0.5 * np.sum(safe_first * np.log(safe_first / safe_midpoint))
        + 0.5 * np.sum(safe_second * np.log(safe_second / safe_midpoint))
    )


class TemporalEmotionAnalyzer:
    """Offline temporal inference over overlapping speech windows."""

    def __init__(self, config: TemporalConfig | None = None) -> None:
        self.config = config or TemporalConfig()
        _validate_config(self.config)

    def analyze(
        self,
        samples: np.ndarray,
        sample_rate: int,
        *,
        classical_model,
        deep_model,
        fusion_engine: FusionEngine,
    ) -> TemporalAnalysis:
        array = np.asarray(samples, dtype=np.float32).reshape(-1)
        if array.size == 0:
            raise TemporalAnalysisError("Cannot analyze an empty audio signal.")
        if sample_rate <= 0:
            raise TemporalAnalysisError("Sample rate must be positive.")

        duration = len(array) / float(sample_rate)
        speech_mask, _ = _speech_mask(array, sample_rate, self.config)
        bounds = _window_bounds(len(array), sample_rate, self.config)

        raw_segments: list[dict] = []
        for index, (start, end) in enumerate(bounds):
            segment = array[start:end]
            start_seconds = start / sample_rate
            end_seconds = end / sample_rate
            ratio = _window_speech_ratio(
                speech_mask,
                start_sample=start,
                end_sample=end,
                sample_rate=sample_rate,
                frame_hop_ms=self.config.hop_length_ms,
            )
            is_speech = ratio >= self.config.min_speech_ratio
            if not is_speech:
                raw_segments.append(
                    {
                        "index": index,
                        "start_seconds": start_seconds,
                        "end_seconds": end_seconds,
                        "duration_seconds": end_seconds - start_seconds,
                        "speech_ratio": ratio,
                        "is_speech": False,
                        "raw_emotion": None,
                        "raw_confidence": 0.0,
                        "probabilities": {},
                        "agreement": None,
                        "js_divergence": None,
                    }
                )
                continue

            try:
                features = extract_acoustic_features(segment, sample_rate)
                classical_prediction = classical_model.predict(features.to_vector())
                deep_prediction = deep_model.predict(segment, sample_rate)
                fused = fusion_engine.combine(
                    classical_prediction["probabilities"],
                    deep_prediction.probabilities,
                )
            except Exception as exc:
                raise TemporalAnalysisError(
                    f"Temporal inference failed for window {index}: {exc}"
                ) from exc

            raw_segments.append(
                {
                    "index": index,
                    "start_seconds": start_seconds,
                    "end_seconds": end_seconds,
                    "duration_seconds": end_seconds - start_seconds,
                    "speech_ratio": ratio,
                    "is_speech": True,
                    "raw_emotion": fused.emotion,
                    "raw_confidence": fused.confidence,
                    "probabilities": dict(fused.probabilities),
                    "agreement": fused.agreement,
                    "js_divergence": fused.js_divergence,
                }
            )

        labels = tuple(fusion_engine.labels)
        previous_smoothed: dict[str, float] | None = None
        previous_raw: dict[str, float] | None = None
        previous_emotion: str | None = None
        segments: list[TemporalSegment] = []
        transitions: list[EmotionTransition] = []

        for raw in raw_segments:
            if not raw["is_speech"]:
                segments.append(
                    TemporalSegment(
                        index=raw["index"],
                        start_seconds=raw["start_seconds"],
                        end_seconds=raw["end_seconds"],
                        duration_seconds=raw["duration_seconds"],
                        speech_ratio=raw["speech_ratio"],
                        is_speech=False,
                        raw_emotion=None,
                        raw_confidence=0.0,
                        emotion=None,
                        confidence=0.0,
                        probabilities={},
                        agreement=None,
                        js_divergence=None,
                        transition=False,
                        transition_score=0.0,
                    )
                )
                continue

            raw_probabilities = {label: float(raw["probabilities"].get(label, 0.0)) for label in labels}
            if previous_smoothed is None:
                smoothed = raw_probabilities
            else:
                alpha = self.config.smoothing_alpha
                smoothed = {
                    label: alpha * raw_probabilities[label]
                    + (1.0 - alpha) * previous_smoothed.get(label, 0.0)
                    for label in labels
                }
                total = sum(smoothed.values())
                smoothed = {label: value / max(total, 1e-12) for label, value in smoothed.items()}

            emotion = max(smoothed, key=smoothed.get)
            confidence = float(smoothed[emotion])
            smooth_change = (
                _js_divergence(previous_smoothed, smoothed, labels)
                if previous_smoothed is not None
                else 0.0
            )
            raw_change = (
                _js_divergence(previous_raw, raw_probabilities, labels)
                if previous_raw is not None
                else 0.0
            )
            transition_score = max(smooth_change, raw_change)
            transition = (
                previous_emotion is not None
                and emotion != previous_emotion
                and transition_score >= self.config.transition_js_threshold
            )
            if transition:
                transitions.append(
                    EmotionTransition(
                        from_emotion=previous_emotion,
                        to_emotion=emotion,
                        at_seconds=float(raw["start_seconds"]),
                        score=transition_score,
                        source_segment_index=raw["index"],
                    )
                )

            segments.append(
                TemporalSegment(
                    index=raw["index"],
                    start_seconds=raw["start_seconds"],
                    end_seconds=raw["end_seconds"],
                    duration_seconds=raw["duration_seconds"],
                    speech_ratio=raw["speech_ratio"],
                    is_speech=True,
                    raw_emotion=raw["raw_emotion"],
                    raw_confidence=raw["raw_confidence"],
                    emotion=emotion,
                    confidence=confidence,
                    probabilities=smoothed,
                    agreement=raw["agreement"],
                    js_divergence=raw["js_divergence"],
                    transition=transition,
                    transition_score=transition_score,
                )
            )
            previous_smoothed = smoothed
            previous_raw = raw_probabilities
            previous_emotion = emotion

        trajectory = _build_trajectory(
            segments,
            labels=labels,
            duration=duration,
            hop_seconds=self.config.hop_seconds,
        )
        emotion_runs = _build_runs(segments)
        frame_hop_seconds = self.config.hop_length_ms / 1000.0
        active_speech_seconds = min(
            duration,
            float(np.sum(speech_mask)) * frame_hop_seconds,
        )
        speech_coverage = min(1.0, active_speech_seconds / max(duration, 1e-12))
        dominant_emotion, dominant_confidence = _dominant_over_time(trajectory, labels)

        return TemporalAnalysis(
            duration_seconds=duration,
            sample_rate=int(sample_rate),
            window_seconds=self.config.window_seconds,
            hop_seconds=self.config.hop_seconds,
            smoothing_alpha=self.config.smoothing_alpha,
            segments=tuple(segments),
            trajectory=tuple(trajectory),
            emotion_runs=tuple(emotion_runs),
            transitions=tuple(transitions),
            dominant_emotion=dominant_emotion,
            dominant_confidence=dominant_confidence,
            active_speech_seconds=active_speech_seconds,
            speech_coverage=speech_coverage,
            analyzed_windows=len(bounds),
            speech_windows=sum(1 for segment in segments if segment.is_speech),
        )


def _build_runs(segments: list[TemporalSegment]) -> list[EmotionRun]:
    runs: list[EmotionRun] = []
    current_emotion: str | None = None
    current: list[TemporalSegment] = []

    def flush() -> None:
        nonlocal current, current_emotion
        if not current or current_emotion is None:
            current = []
            current_emotion = None
            return
        start = current[0].start_seconds
        end = current[-1].end_seconds
        runs.append(
            EmotionRun(
                emotion=current_emotion,
                start_seconds=start,
                end_seconds=end,
                duration_seconds=end - start,
                segment_count=len(current),
                mean_confidence=float(np.mean([item.confidence for item in current])),
            )
        )
        current = []
        current_emotion = None

    for segment in segments:
        if not segment.is_speech or segment.emotion is None:
            flush()
            continue
        if current_emotion is None:
            current_emotion = segment.emotion
            current = [segment]
        elif segment.emotion == current_emotion:
            current.append(segment)
        else:
            flush()
            current_emotion = segment.emotion
            current = [segment]
    flush()
    return runs


def _build_trajectory(
    segments: list[TemporalSegment],
    *,
    labels: tuple[str, ...],
    duration: float,
    hop_seconds: float,
) -> list[TrajectoryPoint]:
    points: list[TrajectoryPoint] = []
    start = 0.0
    epsilon = 1e-9
    while start < duration - epsilon:
        end = min(duration, start + hop_seconds)
        overlapping = [
            segment
            for segment in segments
            if segment.is_speech
            and segment.end_seconds > start + epsilon
            and segment.start_seconds < end - epsilon
        ]
        overlap_weights: list[tuple[TemporalSegment, float]] = []
        for segment in overlapping:
            weight = max(0.0, min(end, segment.end_seconds) - max(start, segment.start_seconds))
            if weight > 0:
                overlap_weights.append((segment, weight))

        if not overlap_weights:
            points.append(
                TrajectoryPoint(
                    start_seconds=start,
                    end_seconds=end,
                    duration_seconds=end - start,
                    is_speech=False,
                    emotion=None,
                    confidence=0.0,
                    speech_ratio=0.0,
                    probabilities={},
                    covered_windows=0,
                )
            )
        else:
            total_weight = sum(weight for _, weight in overlap_weights)
            probabilities = {
                label: sum(segment.probabilities.get(label, 0.0) * weight for segment, weight in overlap_weights) / max(total_weight, 1e-12)
                for label in labels
            }
            emotion = max(probabilities, key=probabilities.get)
            confidence = float(probabilities[emotion])
            speech_ratio = float(
                sum(segment.speech_ratio * weight for segment, weight in overlap_weights)
                / max(total_weight, 1e-12)
            )
            points.append(
                TrajectoryPoint(
                    start_seconds=start,
                    end_seconds=end,
                    duration_seconds=end - start,
                    is_speech=True,
                    emotion=emotion,
                    confidence=confidence,
                    speech_ratio=speech_ratio,
                    probabilities=probabilities,
                    covered_windows=len(overlap_weights),
                )
            )
        start = end
    return points


def _dominant_over_time(trajectory: list[TrajectoryPoint], labels: tuple[str, ...]) -> tuple[str | None, float]:
    speech_points = [point for point in trajectory if point.is_speech and point.emotion is not None]
    if not speech_points:
        return None, 0.0
    total_duration = sum(point.duration_seconds for point in speech_points)
    if total_duration <= 0:
        return None, 0.0

    weighted = {label: 0.0 for label in labels}
    for point in speech_points:
        for label in labels:
            weighted[label] += point.probabilities.get(label, 0.0) * point.duration_seconds
    dominant = max(weighted, key=weighted.get)
    confidence = weighted[dominant] / total_duration
    return dominant, float(confidence)
