from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from app.audio.processor import AudioProcessingError
from app.conversation_graph.engine import build_conversation_graph
from app.conversation_graph.dynamics import ConversationDynamicsAnalyzer
from app.conversation_state.engine import ConversationStateAnalyzer
from app.quality.engine import AudioQualityAnalyzer
from app.speaker_diarization.model import SpeakerTurn
from app.speaker_diarization.engine import SpeakerTurnAnalysis, _interaction_type


class LiveMultiSpeakerError(ValueError):
    """Raised when multi-speaker live analysis cannot continue safely."""


@dataclass(frozen=True)
class LiveMultiSpeakerConfig:
    target_sample_rate: int = 16_000
    context_seconds: float = 8.0
    hop_seconds: float = 1.0
    min_context_seconds: float = 3.0
    min_turn_seconds: float = 0.7
    stabilization_seconds: float = 0.8
    max_duration_seconds: float = 180.0
    max_chunk_bytes: int = 256 * 1024
    max_speakers: int = 6
    speaker_match_threshold: float = 0.18


@dataclass(frozen=True)
class LiveMultiSpeakerSnapshot:
    sequence: int
    window_start_seconds: float
    window_end_seconds: float
    duration_seconds: float
    speaker_count: int
    active_speakers: list[str]
    latest_speaker: str | None
    latest_emotion: str | None
    latest_confidence: float | None
    final_state: str
    state_transition: str | None
    speech_coverage: float
    quality_score: float
    quality_verdict: str
    latency_ms: float
    speakers: list[dict[str, Any]]
    turns: list[dict[str, Any]]
    interaction_graph: dict[str, Any]
    conversation_dynamics: dict[str, Any]
    conversation_state: dict[str, Any]
    speaker_id_stability: float
    provisional: bool
    model_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "analysis",
            "sequence": self.sequence,
            "window_start_seconds": round(self.window_start_seconds, 4),
            "window_end_seconds": round(self.window_end_seconds, 4),
            "duration_seconds": round(self.duration_seconds, 4),
            "speaker_count": self.speaker_count,
            "active_speakers": list(self.active_speakers),
            "latest_speaker": self.latest_speaker,
            "latest_emotion": self.latest_emotion,
            "latest_confidence": None if self.latest_confidence is None else round(self.latest_confidence, 6),
            "final_state": self.final_state,
            "state_transition": self.state_transition,
            "speech_coverage": round(self.speech_coverage, 6),
            "quality_score": round(self.quality_score, 6),
            "quality_verdict": self.quality_verdict,
            "latency_ms": round(self.latency_ms, 2),
            "speakers": self.speakers,
            "turns": self.turns,
            "interaction_graph": self.interaction_graph,
            "conversation_dynamics": self.conversation_dynamics,
            "conversation_state": self.conversation_state,
            "speaker_id_stability": round(self.speaker_id_stability, 6),
            "provisional": self.provisional,
            "model_metadata": self.model_metadata,
        }


Predictor = Callable[[np.ndarray, int], dict[str, Any]]
Diarizer = Callable[[np.ndarray, int], list[SpeakerTurn]]


def _normalize_probabilities(values: Any) -> dict[str, float]:
    if not isinstance(values, dict) or not values:
        raise LiveMultiSpeakerError("Live speaker predictor returned no probabilities.")
    normalized: dict[str, float] = {}
    for key, value in values.items():
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise LiveMultiSpeakerError("Live speaker predictor returned invalid probabilities.")
        normalized[str(key)] = number
    total = sum(normalized.values())
    if total <= 0:
        raise LiveMultiSpeakerError("Live speaker predictor returned zero total probability.")
    return {key: value / total for key, value in normalized.items()}


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0 or number > 1:
        return None
    return number


def _overlap(first: SpeakerTurn, second: SpeakerTurn) -> float:
    return max(0.0, min(first.end_seconds, second.end_seconds) - max(first.start_seconds, second.start_seconds))


def _iou(first: SpeakerTurn, second: SpeakerTurn) -> float:
    intersection = _overlap(first, second)
    union = max(first.end_seconds, second.end_seconds) - min(first.start_seconds, second.start_seconds)
    return intersection / max(union, 1e-9)


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
    return total + end - start


def _overlap_seconds(turns: list[SpeakerTurn]) -> float:
    total = 0.0
    ordered = sorted(turns, key=lambda item: (item.start_seconds, item.end_seconds))
    for index, first in enumerate(ordered):
        for second in ordered[index + 1 :]:
            if second.start_seconds >= first.end_seconds:
                break
            total += _overlap(first, second)
    return total


class LiveMultiSpeakerSession:
    """Maintain a bounded audio buffer and stabilize speaker identities across rolling diarization."""

    def __init__(self, source_sample_rate: int, config: LiveMultiSpeakerConfig | None = None) -> None:
        if not 8_000 <= source_sample_rate <= 96_000:
            raise LiveMultiSpeakerError("Source sample rate must be between 8 kHz and 96 kHz.")
        self.config = config or LiveMultiSpeakerConfig()
        self._validate_config(self.config)
        self.source_sample_rate = int(source_sample_rate)
        self.samples = np.zeros(0, dtype=np.float32)
        self.sequence = 0
        self.last_scheduled_end = 0
        self.next_speaker_number = 1
        self.global_turns: list[SpeakerTurn] = []
        self.analyses: list[SpeakerTurnAnalysis] = []
        self.speaker_match_hits = 0
        self.speaker_match_decisions = 0

    @staticmethod
    def _validate_config(config: LiveMultiSpeakerConfig) -> None:
        if config.target_sample_rate <= 0:
            raise LiveMultiSpeakerError("Target sample rate must be positive.")
        if config.context_seconds < 2.0:
            raise LiveMultiSpeakerError("Live multi-speaker context must be at least 2 seconds.")
        if config.hop_seconds <= 0 or config.hop_seconds > config.context_seconds:
            raise LiveMultiSpeakerError("Live multi-speaker hop is invalid.")
        if config.min_context_seconds <= 0 or config.min_context_seconds > config.context_seconds:
            raise LiveMultiSpeakerError("Minimum multi-speaker context is invalid.")
        if config.min_turn_seconds <= 0:
            raise LiveMultiSpeakerError("Minimum multi-speaker turn duration must be positive.")
        if config.stabilization_seconds < 0 or config.stabilization_seconds >= config.context_seconds:
            raise LiveMultiSpeakerError("Speaker stabilization horizon is invalid.")
        if config.max_duration_seconds <= config.context_seconds:
            raise LiveMultiSpeakerError("Maximum duration must exceed the rolling context.")
        if config.max_chunk_bytes <= 0:
            raise LiveMultiSpeakerError("Live multi-speaker chunk size must be positive.")
        if config.max_speakers <= 0:
            raise LiveMultiSpeakerError("Maximum speaker count must be positive.")
        if not 0 < config.speaker_match_threshold <= 1:
            raise LiveMultiSpeakerError("Speaker matching threshold must be between 0 and 1.")

    @property
    def duration_seconds(self) -> float:
        return float(len(self.samples) / self.config.target_sample_rate)

    def append_pcm16(self, chunk: bytes) -> float:
        if not chunk:
            raise LiveMultiSpeakerError("Live multi-speaker audio chunk is empty.")
        if len(chunk) > self.config.max_chunk_bytes:
            raise LiveMultiSpeakerError("Live multi-speaker audio chunk is too large.")
        if len(chunk) % 2:
            raise LiveMultiSpeakerError("Live multi-speaker PCM16 chunk has an incomplete sample.")
        pcm = np.frombuffer(chunk, dtype="<i2").astype(np.float32) / 32768.0
        normalized = self._resample(pcm, self.source_sample_rate, self.config.target_sample_rate)
        self.samples = np.concatenate((self.samples, normalized.astype(np.float32, copy=False)))
        if self.duration_seconds > self.config.max_duration_seconds + 1e-9:
            raise LiveMultiSpeakerError(
                f"Live multi-speaker recording exceeded the {self.config.max_duration_seconds:.0f}-second limit."
            )
        return self.duration_seconds

    def pending_window(self, *, final: bool = False) -> tuple[np.ndarray, float, float] | None:
        available = self.duration_seconds
        context = self.config.context_seconds
        if available >= context - 1e-6:
            elapsed_since_schedule = available - self.last_scheduled_end / self.config.target_sample_rate
            if elapsed_since_schedule < self.config.hop_seconds - 1e-6:
                return None
            start = max(0.0, available - context)
        elif final and available >= self.config.min_context_seconds:
            if available - self.last_scheduled_end / self.config.target_sample_rate < self.config.min_context_seconds - 1e-6:
                return None
            start = 0.0
        else:
            return None
        start_index = int(round(start * self.config.target_sample_rate))
        snapshot = self.samples[start_index:].copy()
        self.last_scheduled_end = len(self.samples)
        return snapshot, start, available

    def next_sequence(self) -> int:
        self.sequence += 1
        return self.sequence

    def analyze_window(
        self,
        snapshot: np.ndarray,
        window_start_seconds: float,
        window_end_seconds: float,
        *,
        diarizer: Diarizer,
        predictor: Predictor,
    ) -> LiveMultiSpeakerSnapshot:
        if snapshot.size == 0:
            raise LiveMultiSpeakerError("Cannot analyze an empty live multi-speaker window.")
        local_turns = diarizer(snapshot, self.config.target_sample_rate)
        bounded_local: list[SpeakerTurn] = []
        for turn in local_turns:
            duration = turn.end_seconds - turn.start_seconds
            if not math.isfinite(duration) or duration < self.config.min_turn_seconds:
                continue
            start = max(0.0, turn.start_seconds) + window_start_seconds
            end = min(window_end_seconds - window_start_seconds, turn.end_seconds) + window_start_seconds
            if end <= start:
                continue
            bounded_local.append(
                SpeakerTurn(
                    speaker=turn.speaker,
                    start_seconds=start,
                    end_seconds=end,
                    duration_seconds=end - start,
                )
            )
        bounded_local.sort(key=lambda item: (item.start_seconds, item.end_seconds, item.speaker))
        if len({item.speaker for item in bounded_local}) > self.config.max_speakers:
            raise LiveMultiSpeakerError("The live diarizer detected more speakers than the configured limit.")

        resolved = self._stabilize_speaker_ids(bounded_local)
        self._merge_into_global_turns(resolved)
        self._analyze_stable_turns(predictor, current_duration_seconds=window_end_seconds)
        active_speakers = sorted({turn.speaker for turn in resolved})

        quality = AudioQualityAnalyzer().analyze(snapshot, self.config.target_sample_rate)
        if self.analyses:
            analysis = self._build_analysis()
            graph = build_conversation_graph(analysis)
            dynamics = ConversationDynamicsAnalyzer().analyze(analysis)
            state = ConversationStateAnalyzer().analyze(dynamics, duration_seconds=window_end_seconds)
        else:
            analysis = None
            graph = self._provisional_graph(active_speakers)
            dynamics = self._provisional_dynamics()
            state = self._provisional_state(window_end_seconds)

        latest = max(self.analyses, key=lambda item: (item.end_seconds, item.turn_index), default=None)
        state_payload = state if isinstance(state, dict) else state.to_dict()
        snapshot_turns = [item.to_dict() | {"status": "analyzed"} for item in self.analyses if item.start_seconds >= max(0.0, window_end_seconds - self.config.context_seconds)]
        for turn in resolved:
            if not any(_iou(turn, analyzed_turn) > 0.55 and turn.speaker == analyzed_turn.speaker for analyzed_turn in self.global_turns if analyzed_turn is not turn):
                if turn.end_seconds > window_end_seconds - self.config.stabilization_seconds:
                    snapshot_turns.append(
                        {
                            "turn_id": self._turn_id(turn),
                            "speaker": turn.speaker,
                            "start_seconds": round(turn.start_seconds, 4),
                            "end_seconds": round(turn.end_seconds, 4),
                            "duration_seconds": round(turn.duration_seconds, 4),
                            "emotion": None,
                            "confidence": None,
                            "valence": None,
                            "arousal": None,
                            "dominance": None,
                            "status": "provisional",
                        }
                    )
        snapshot_turns.sort(key=lambda item: (item["start_seconds"], item["end_seconds"], item["speaker"]))
        provisional = any(item["status"] == "provisional" for item in snapshot_turns)

        speaker_id_stability = self.speaker_match_hits / max(self.speaker_match_decisions, 1)
        return LiveMultiSpeakerSnapshot(
            sequence=self.next_sequence(),
            window_start_seconds=window_start_seconds,
            window_end_seconds=window_end_seconds,
            duration_seconds=window_end_seconds,
            speaker_count=analysis.speaker_count if analysis is not None else len(active_speakers),
            active_speakers=active_speakers,
            latest_speaker=latest.speaker if latest else (resolved[-1].speaker if resolved else None),
            latest_emotion=latest.emotion if latest else None,
            latest_confidence=latest.confidence if latest else None,
            final_state=state_payload.get("end_state", "insufficient_data"),
            state_transition=(state.transitions[-1].transition if state.transitions else None) if not isinstance(state, dict) else ((state.get("transitions") or [{}])[-1].get("to_state") if state.get("transitions") else None),
            speech_coverage=analysis.speech_coverage if analysis is not None else 0.0,
            quality_score=quality.quality_score,
            quality_verdict=quality.verdict,
            latency_ms=0.0,
            speakers=[summary.to_dict() for summary in analysis.speakers] if analysis is not None else [],
            turns=snapshot_turns,
            interaction_graph=graph if isinstance(graph, dict) else graph.to_dict(),
            conversation_dynamics=dynamics if isinstance(dynamics, dict) else dynamics.to_dict(),
            conversation_state=state if isinstance(state, dict) else state.to_dict(),
            speaker_id_stability=float(np.clip(speaker_id_stability, 0.0, 1.0)),
            provisional=provisional,
            model_metadata={
                "mode": "rolling_multi_speaker",
                "context_seconds": self.config.context_seconds,
                "hop_seconds": self.config.hop_seconds,
                "speaker_stabilization_seconds": self.config.stabilization_seconds,
            },
        )

    @staticmethod
    def _provisional_graph(active_speakers: list[str]) -> dict[str, Any]:
        return {
            "graph_type": "directed_conversation_interaction",
            "directed": True,
            "nodes": [
                {
                    "speaker": speaker,
                    "turn_count": 0,
                    "speaking_seconds": 0.0,
                    "speaking_share": 0.0,
                    "dominant_emotion": "unknown",
                    "mean_confidence": 0.0,
                    "mean_valence": None,
                    "mean_arousal": None,
                    "mean_dominance": None,
                    "in_interactions": 0,
                    "out_interactions": 0,
                    "unique_contacts": 0,
                    "first_turn_seconds": 0.0,
                    "last_turn_seconds": 0.0,
                }
                for speaker in active_speakers
            ],
            "edges": [],
            "metrics": {
                "node_count": len(active_speakers),
                "edge_count": 0,
                "total_interactions": 0,
                "density": 0.0,
                "reciprocity": 0.0,
                "active_pairs": 0,
                "connected_speakers": 0,
            },
            "semantics": {
                "status": "provisional_diarization_only",
                "edge_direction": "observed turn-to-turn response direction",
            },
        }

    @staticmethod
    def _provisional_dynamics() -> dict[str, Any]:
        return {
            "event_count": 0,
            "escalation_score": 0.0,
            "deescalation_score": 0.0,
            "net_tension": 0.0,
            "tension_trend": 0.0,
            "volatility": 0.0,
            "stability_score": 0.0,
            "affect_synchrony": None,
            "mean_response_gap_seconds": None,
            "median_response_gap_seconds": None,
            "p90_response_gap_seconds": None,
            "fast_response_share": 0.0,
            "response_latency_shift_seconds": None,
            "arousal_rise_events": 0,
            "valence_drop_events": 0,
            "escalation_events": 0,
            "deescalation_events": 0,
            "stable_events": 0,
            "dominant_pattern": "insufficient_data",
            "events": [],
            "pair_metrics": [],
            "semantics": {"status": "provisional_diarization_only"},
        }

    @staticmethod
    def _provisional_state(duration_seconds: float) -> dict[str, Any]:
        return {
            "initial_state": "insufficient_data",
            "current_state": "insufficient_data",
            "peak_state": "insufficient_data",
            "peak_tension": 0.0,
            "peak_tension_at_seconds": 0.0,
            "transition_count": 0,
            "state_occupancy": {},
            "points": [],
            "transitions": [],
            "semantics": {
                "status": "provisional_diarization_only",
                "duration_seconds": duration_seconds,
            },
        }

    def _stabilize_speaker_ids(self, turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
        local_labels = sorted({turn.speaker for turn in turns})
        label_map: dict[str, str] = {}
        used_targets: set[str] = set()
        for local_label in local_labels:
            candidates: list[tuple[float, str]] = []
            local_group = [turn for turn in turns if turn.speaker == local_label]
            for existing_speaker in sorted({turn.speaker for turn in self.global_turns}):
                if existing_speaker in used_targets:
                    continue
                overlap = sum(_overlap(local_turn, existing_turn) for local_turn in local_group for existing_turn in self.global_turns if existing_turn.speaker == existing_speaker)
                local_duration = sum(turn.duration_seconds for turn in local_group)
                ratio = overlap / max(local_duration, 1e-9)
                candidates.append((ratio, existing_speaker))
            candidates.sort(reverse=True)
            if candidates and candidates[0][0] >= self.config.speaker_match_threshold:
                label_map[local_label] = candidates[0][1]
                used_targets.add(candidates[0][1])
                self.speaker_match_hits += 1
            else:
                label_map[local_label] = f"SPEAKER_{self.next_speaker_number:02d}"
                self.next_speaker_number += 1
            self.speaker_match_decisions += 1
        return [
            SpeakerTurn(
                speaker=label_map[turn.speaker],
                start_seconds=turn.start_seconds,
                end_seconds=turn.end_seconds,
                duration_seconds=turn.duration_seconds,
            )
            for turn in turns
        ]

    def _merge_into_global_turns(self, turns: list[SpeakerTurn]) -> None:
        for turn in turns:
            match = None
            for existing in self.global_turns:
                if existing.speaker != turn.speaker:
                    continue
                if _iou(existing, turn) >= 0.55 or _overlap(existing, turn) >= 0.25:
                    match = existing
                    break
            if match is None:
                self.global_turns.append(turn)
            else:
                start = min(match.start_seconds, turn.start_seconds)
                end = max(match.end_seconds, turn.end_seconds)
                index = self.global_turns.index(match)
                self.global_turns[index] = SpeakerTurn(
                    speaker=match.speaker,
                    start_seconds=start,
                    end_seconds=end,
                    duration_seconds=end - start,
                )
        self.global_turns.sort(key=lambda item: (item.start_seconds, item.end_seconds, item.speaker))

    def _analyze_stable_turns(self, predictor: Predictor, *, current_duration_seconds: float | None = None) -> None:
        current_duration = max(self.duration_seconds, float(current_duration_seconds or 0.0))
        horizon = current_duration - self.config.stabilization_seconds
        for turn in self.global_turns:
            if turn.end_seconds > horizon + 1e-6:
                continue
            if self._has_analysis(turn):
                continue
            start = int(round(turn.start_seconds * self.config.target_sample_rate))
            end = int(round(turn.end_seconds * self.config.target_sample_rate))
            segment = self.samples[max(0, start): min(len(self.samples), end)]
            if segment.size == 0:
                continue
            prediction = predictor(segment, self.config.target_sample_rate)
            probabilities = _normalize_probabilities(prediction.get("probabilities"))
            emotion = str(prediction.get("emotion") or max(probabilities, key=probabilities.get))
            confidence = float(prediction.get("confidence", probabilities.get(emotion, 0.0)))
            if not 0 <= confidence <= 1 or not math.isfinite(confidence):
                raise LiveMultiSpeakerError("Live speaker predictor returned invalid confidence.")
            self.analyses.append(
                SpeakerTurnAnalysis(
                    turn_index=len(self.analyses),
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
        self.analyses.sort(key=lambda item: (item.start_seconds, item.end_seconds, item.turn_index))

    def _has_analysis(self, turn: SpeakerTurn) -> bool:
        return any(
            item.speaker == turn.speaker
            and _iou(
                SpeakerTurn(item.speaker, item.start_seconds, item.end_seconds, item.duration_seconds),
                turn,
            ) >= 0.55
            and abs(item.start_seconds - turn.start_seconds) <= 0.35
            for item in self.analyses
        )

    def _turn_id(self, turn: SpeakerTurn) -> str:
        return f"{turn.speaker}:{turn.start_seconds:.2f}:{turn.end_seconds:.2f}"

    def _build_analysis(self):
        analyses = tuple(self.analyses)
        speaker_names = sorted({item.speaker for item in analyses})
        if not analyses:
            raise LiveMultiSpeakerError("No stable speaker turns are available yet.")
        summary = []
        for speaker in speaker_names:
            items = [item for item in analyses if item.speaker == speaker]
            speaking_seconds = sum(item.duration_seconds for item in items)
            probabilities: dict[str, float] = {}
            for item in items:
                for label, value in item.probabilities.items():
                    probabilities[label] = probabilities.get(label, 0.0) + value * item.duration_seconds
            dominant = max(probabilities, key=probabilities.get)
            summary.append(
                _LiveSpeakerSummary(
                    speaker=speaker,
                    turn_count=len(items),
                    speaking_seconds=speaking_seconds,
                    speaking_share=speaking_seconds / max(_union_duration([(i.start_seconds, i.end_seconds) for i in analyses]), 1e-9),
                    dominant_emotion=dominant,
                    dominant_confidence=probabilities[dominant] / max(speaking_seconds, 1e-9),
                    mean_confidence=sum(item.confidence for item in items) / len(items),
                    first_emotion=items[0].emotion,
                    last_emotion=items[-1].emotion,
                    emotion_changed=items[0].emotion != items[-1].emotion,
                    mean_valence=_mean([item.valence for item in items]),
                    mean_arousal=_mean([item.arousal for item in items]),
                    mean_dominance=_mean([item.dominance for item in items]),
                    first_valence=items[0].valence,
                    last_valence=items[-1].valence,
                    first_arousal=items[0].arousal,
                    last_arousal=items[-1].arousal,
                )
            )
        summary.sort(key=lambda item: item.speaking_seconds, reverse=True)

        interactions = []
        for previous, current in zip(analyses, analyses[1:], strict=False):
            if previous.speaker == current.speaker:
                continue
            interactions.append(
                _LiveInteraction(
                    from_speaker=previous.speaker,
                    to_speaker=current.speaker,
                    at_seconds=current.start_seconds,
                    gap_seconds=max(0.0, current.start_seconds - previous.end_seconds),
                    previous_emotion=previous.emotion,
                    next_emotion=current.emotion,
                    arousal_change=None if previous.arousal is None or current.arousal is None else current.arousal - previous.arousal,
                    valence_change=None if previous.valence is None or current.valence is None else current.valence - previous.valence,
                    type=_interaction_type(previous, current),
                )
            )
        from app.speaker_diarization.engine import SpeakerAwareAnalysis

        union_seconds = _union_duration([(item.start_seconds, item.end_seconds) for item in analyses])
        return SpeakerAwareAnalysis(
            duration_seconds=max(self.duration_seconds, max(item.end_seconds for item in analyses)),
            sample_rate=self.config.target_sample_rate,
            speaker_count=len(speaker_names),
            speakers=tuple(summary),
            turns=analyses,
            interactions=tuple(interactions),
            total_speaking_seconds=union_seconds,
            speech_coverage=min(1.0, union_seconds / max(self.duration_seconds, 1e-9)),
            overlap_seconds=_overlap_seconds([SpeakerTurn(item.speaker, item.start_seconds, item.end_seconds, item.duration_seconds) for item in analyses]),
            speaker_switches=len(interactions),
            dominant_speaker=summary[0].speaker if summary else None,
        )

    @staticmethod
    def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        if source_rate == target_rate:
            return samples.astype(np.float32, copy=True)
        if samples.size == 0:
            return samples.astype(np.float32)
        target_length = max(1, int(round(len(samples) * target_rate / source_rate)))
        positions = np.linspace(0.0, len(samples) - 1, target_length)
        base = np.arange(len(samples), dtype=np.float64)
        return np.interp(positions, base, samples.astype(np.float64)).astype(np.float32)


@dataclass(frozen=True)
class _LiveSpeakerSummary:
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
            "latest_emotion": self.last_emotion,
            "mean_confidence": round(self.mean_confidence, 6),
            "mean_valence": None if self.mean_valence is None else round(self.mean_valence, 6),
            "mean_arousal": None if self.mean_arousal is None else round(self.mean_arousal, 6),
            "mean_dominance": None if self.mean_dominance is None else round(self.mean_dominance, 6),
        }


@dataclass(frozen=True)
class _LiveInteraction:
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
            "arousal_change": self.arousal_change,
            "valence_change": self.valence_change,
            "type": self.type,
        }


def _mean(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(clean) / len(clean) if clean else None
