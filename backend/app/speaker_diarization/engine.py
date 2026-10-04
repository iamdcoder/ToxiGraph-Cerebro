from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from app.speaker_diarization.model import SpeakerTurn


class SpeakerAnalysisError(ValueError):
    """Raised when speaker-aware emotion analysis cannot be completed."""


@dataclass(frozen=True)
class SpeakerTurnAnalysis:
    turn_index: int
    speaker: str
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    emotion: str
    confidence: float
    probabilities: dict[str, float]
    valence: float | None
    arousal: float | None
    dominance: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "speaker": self.speaker,
            "start_seconds": round(self.start_seconds, 4),
            "end_seconds": round(self.end_seconds, 4),
            "duration_seconds": round(self.duration_seconds, 4),
            "emotion": self.emotion,
            "confidence": round(self.confidence, 6),
            "probabilities": {key: round(value, 6) for key, value in self.probabilities.items()},
            "valence": round(self.valence, 6) if self.valence is not None else None,
            "arousal": round(self.arousal, 6) if self.arousal is not None else None,
            "dominance": round(self.dominance, 6) if self.dominance is not None else None,
        }


@dataclass(frozen=True)
class SpeakerSummary:
    speaker: str
    turn_count: int
    speaking_seconds: float
    speaking_share: float
    dominant_emotion: str
    dominant_confidence: float
    mean_confidence: float
    first_emotion: str
    last_emotion: str
    emotion_changed: bool
    mean_valence: float | None
    mean_arousal: float | None
    mean_dominance: float | None
    first_valence: float | None
    last_valence: float | None
    first_arousal: float | None
    last_arousal: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker": self.speaker,
            "turn_count": self.turn_count,
            "speaking_seconds": round(self.speaking_seconds, 4),
            "speaking_share": round(self.speaking_share, 6),
            "dominant_emotion": self.dominant_emotion,
            "dominant_confidence": round(self.dominant_confidence, 6),
            "mean_confidence": round(self.mean_confidence, 6),
            "first_emotion": self.first_emotion,
            "last_emotion": self.last_emotion,
            "emotion_changed": self.emotion_changed,
            "mean_valence": round(self.mean_valence, 6) if self.mean_valence is not None else None,
            "mean_arousal": round(self.mean_arousal, 6) if self.mean_arousal is not None else None,
            "mean_dominance": round(self.mean_dominance, 6) if self.mean_dominance is not None else None,
            "first_valence": round(self.first_valence, 6) if self.first_valence is not None else None,
            "last_valence": round(self.last_valence, 6) if self.last_valence is not None else None,
            "first_arousal": round(self.first_arousal, 6) if self.first_arousal is not None else None,
            "last_arousal": round(self.last_arousal, 6) if self.last_arousal is not None else None,
        }


@dataclass(frozen=True)
class SpeakerInteraction:
    from_speaker: str
    to_speaker: str
    at_seconds: float
    gap_seconds: float
    previous_emotion: str
    next_emotion: str
    arousal_change: float | None
    valence_change: float | None
    type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_speaker": self.from_speaker,
            "to_speaker": self.to_speaker,
            "at_seconds": round(self.at_seconds, 4),
            "gap_seconds": round(self.gap_seconds, 4),
            "previous_emotion": self.previous_emotion,
            "next_emotion": self.next_emotion,
            "arousal_change": round(self.arousal_change, 6) if self.arousal_change is not None else None,
            "valence_change": round(self.valence_change, 6) if self.valence_change is not None else None,
            "type": self.type,
        }


@dataclass(frozen=True)
class SpeakerAwareAnalysis:
    duration_seconds: float
    sample_rate: int
    speaker_count: int
    speakers: tuple[SpeakerSummary, ...]
    turns: tuple[SpeakerTurnAnalysis, ...]
    interactions: tuple[SpeakerInteraction, ...]
    total_speaking_seconds: float
    speech_coverage: float
    overlap_seconds: float
    speaker_switches: int
    dominant_speaker: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_seconds": round(self.duration_seconds, 4),
            "sample_rate": self.sample_rate,
            "speaker_count": self.speaker_count,
            "speakers": [speaker.to_dict() for speaker in self.speakers],
            "turns": [turn.to_dict() for turn in self.turns],
            "interactions": [item.to_dict() for item in self.interactions],
            "total_speaking_seconds": round(self.total_speaking_seconds, 4),
            "speech_coverage": round(self.speech_coverage, 6),
            "overlap_seconds": round(self.overlap_seconds, 4),
            "speaker_switches": self.speaker_switches,
            "dominant_speaker": self.dominant_speaker,
        }


Predictor = Callable[[np.ndarray, int], dict[str, Any]]


def _normalize_probs(values: Any) -> dict[str, float]:
    if not isinstance(values, dict) or not values:
        raise SpeakerAnalysisError("Speaker emotion predictor returned no probabilities.")
    result: dict[str, float] = {}
    for key, value in values.items():
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise SpeakerAnalysisError("Speaker emotion predictor returned invalid probabilities.")
        result[str(key)] = number
    total = sum(result.values())
    if total <= 0:
        raise SpeakerAnalysisError("Speaker emotion predictor returned zero total probability.")
    return {key: value / total for key, value in result.items()}


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    if number < 0 or number > 1:
        return None
    return number


def _merge_turns(turns: list[SpeakerTurn], merge_gap: float = 0.20) -> list[SpeakerTurn]:
    if not turns:
        return []
    merged: list[SpeakerTurn] = []
    for turn in turns:
        if (
            merged
            and merged[-1].speaker == turn.speaker
            and turn.start_seconds - merged[-1].end_seconds <= merge_gap
        ):
            previous = merged.pop()
            merged.append(
                SpeakerTurn(
                    speaker=turn.speaker,
                    start_seconds=previous.start_seconds,
                    end_seconds=max(previous.end_seconds, turn.end_seconds),
                    duration_seconds=max(previous.end_seconds, turn.end_seconds) - previous.start_seconds,
                )
            )
        else:
            merged.append(turn)
    return merged


def _overlap_seconds(turns: list[SpeakerTurn]) -> float:
    total = 0.0
    for index, first in enumerate(turns):
        for second in turns[index + 1 :]:
            if second.start_seconds >= first.end_seconds:
                break
            total += max(0.0, min(first.end_seconds, second.end_seconds) - max(first.start_seconds, second.start_seconds))
    return float(total)


def _speaker_summary(speaker: str, items: list[SpeakerTurnAnalysis], total_union_seconds: float) -> SpeakerSummary:
    speaking_seconds = sum(item.duration_seconds for item in items)
    prob_keys = sorted({key for item in items for key in item.probabilities})
    aggregate = {key: 0.0 for key in prob_keys}
    for item in items:
        for key in prob_keys:
            aggregate[key] += item.probabilities.get(key, 0.0) * item.duration_seconds
    if aggregate:
        dominant_emotion = max(aggregate, key=aggregate.get)
        dominant_mass = aggregate[dominant_emotion] / max(speaking_seconds, 1e-9)
    else:
        dominant_emotion = items[0].emotion
        dominant_mass = items[0].confidence
    confidence = [item.confidence for item in items]
    affect_values = {
        "valence": [item.valence for item in items if item.valence is not None],
        "arousal": [item.arousal for item in items if item.arousal is not None],
        "dominance": [item.dominance for item in items if item.dominance is not None],
    }
    return SpeakerSummary(
        speaker=speaker,
        turn_count=len(items),
        speaking_seconds=speaking_seconds,
        speaking_share=speaking_seconds / max(total_union_seconds, 1e-9),
        dominant_emotion=dominant_emotion,
        dominant_confidence=float(dominant_mass),
        mean_confidence=float(sum(confidence) / len(confidence)),
        first_emotion=items[0].emotion,
        last_emotion=items[-1].emotion,
        emotion_changed=items[0].emotion != items[-1].emotion,
        mean_valence=_mean_optional(affect_values["valence"]),
        mean_arousal=_mean_optional(affect_values["arousal"]),
        mean_dominance=_mean_optional(affect_values["dominance"]),
        first_valence=items[0].valence,
        last_valence=items[-1].valence,
        first_arousal=items[0].arousal,
        last_arousal=items[-1].arousal,
    )


def _mean_optional(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return float(sum(clean) / len(clean)) if clean else None


def _interaction_type(previous: SpeakerTurnAnalysis, current: SpeakerTurnAnalysis) -> str:
    if previous.emotion == current.emotion:
        return "continuity"
    arousal_change = None
    valence_change = None
    if previous.arousal is not None and current.arousal is not None:
        arousal_change = current.arousal - previous.arousal
    if previous.valence is not None and current.valence is not None:
        valence_change = current.valence - previous.valence
    if (arousal_change is not None and arousal_change >= 0.15) or (
        valence_change is not None and valence_change <= -0.15
    ):
        return "activated_shift"
    return "emotion_transition"


class SpeakerAwareAnalyzer:
    """Attach emotion/affect analysis to diarized speaker turns."""

    def analyze(
        self,
        samples: np.ndarray,
        sample_rate: int,
        turns: list[SpeakerTurn],
        *,
        predictor: Predictor,
        min_turn_seconds: float = 0.8,
    ) -> SpeakerAwareAnalysis:
        array = np.asarray(samples, dtype=np.float32).reshape(-1)
        if array.size == 0:
            raise SpeakerAnalysisError("Cannot analyze an empty audio signal.")
        if sample_rate <= 0:
            raise SpeakerAnalysisError("Sample rate must be positive.")
        if min_turn_seconds <= 0:
            raise SpeakerAnalysisError("Minimum speaker-turn duration must be positive.")

        clean_turns = _merge_turns(sorted(turns, key=lambda turn: (turn.start_seconds, turn.end_seconds)))
        bounded: list[SpeakerTurn] = []
        duration = len(array) / sample_rate
        for turn in clean_turns:
            start = max(0.0, min(duration, turn.start_seconds))
            end = max(0.0, min(duration, turn.end_seconds))
            if end <= start or end - start < min_turn_seconds:
                continue
            bounded.append(
                SpeakerTurn(
                    speaker=turn.speaker,
                    start_seconds=start,
                    end_seconds=end,
                    duration_seconds=end - start,
                )
            )
        if not bounded:
            raise SpeakerAnalysisError("No usable speaker turns remained after minimum-duration filtering.")

        analyses: list[SpeakerTurnAnalysis] = []
        for index, turn in enumerate(bounded):
            start_sample = max(0, int(round(turn.start_seconds * sample_rate)))
            end_sample = min(len(array), int(round(turn.end_seconds * sample_rate)))
            segment = array[start_sample:end_sample]
            if segment.size == 0:
                continue
            prediction = predictor(segment, sample_rate)
            probabilities = _normalize_probs(prediction.get("probabilities"))
            emotion = str(prediction.get("emotion") or max(probabilities, key=probabilities.get))
            confidence = float(prediction.get("confidence", probabilities.get(emotion, 0.0)))
            if not math.isfinite(confidence) or confidence < 0 or confidence > 1:
                raise SpeakerAnalysisError("Speaker emotion predictor returned invalid confidence.")
            analyses.append(
                SpeakerTurnAnalysis(
                    turn_index=index,
                    speaker=turn.speaker,
                    start_seconds=turn.start_seconds,
                    end_seconds=turn.end_seconds,
                    duration_seconds=turn.duration_seconds,
                    emotion=emotion,
                    confidence=confidence,
                    probabilities=probabilities,
                    valence=_float_or_none(prediction.get("valence")),
                    arousal=_float_or_none(prediction.get("arousal")),
                    dominance=_float_or_none(prediction.get("dominance")),
                )
            )

        if not analyses:
            raise SpeakerAnalysisError("No speaker turn produced a usable emotion prediction.")

        speakers = sorted({item.speaker for item in analyses})
        union_intervals = _union_duration([(item.start_seconds, item.end_seconds) for item in analyses])
        summary = [
            _speaker_summary(
                speaker,
                [item for item in analyses if item.speaker == speaker],
                union_intervals,
            )
            for speaker in speakers
        ]
        summary.sort(key=lambda item: item.speaking_seconds, reverse=True)

        interactions: list[SpeakerInteraction] = []
        switches = 0
        for previous, current in zip(analyses, analyses[1:], strict=False):
            if previous.speaker == current.speaker:
                continue
            switches += 1
            gap = max(0.0, current.start_seconds - previous.end_seconds)
            arousal_change = None
            valence_change = None
            if previous.arousal is not None and current.arousal is not None:
                arousal_change = current.arousal - previous.arousal
            if previous.valence is not None and current.valence is not None:
                valence_change = current.valence - previous.valence
            interactions.append(
                SpeakerInteraction(
                    from_speaker=previous.speaker,
                    to_speaker=current.speaker,
                    at_seconds=current.start_seconds,
                    gap_seconds=gap,
                    previous_emotion=previous.emotion,
                    next_emotion=current.emotion,
                    arousal_change=arousal_change,
                    valence_change=valence_change,
                    type=_interaction_type(previous, current),
                )
            )

        return SpeakerAwareAnalysis(
            duration_seconds=duration,
            sample_rate=sample_rate,
            speaker_count=len(speakers),
            speakers=tuple(summary),
            turns=tuple(analyses),
            interactions=tuple(interactions),
            total_speaking_seconds=union_intervals,
            speech_coverage=min(1.0, union_intervals / max(duration, 1e-9)),
            overlap_seconds=_overlap_seconds(bounded),
            speaker_switches=switches,
            dominant_speaker=summary[0].speaker if summary else None,
        )


def _union_duration(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0.0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + (end - start)
