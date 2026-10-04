from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Iterable

import numpy as np

from app.longitudinal_emotion.service import HistorySession


class VoiceChangeError(ValueError):
    """Raised when longitudinal voice change analysis cannot be computed."""


MIN_HISTORY_SESSIONS = 3
EMOTION_CLASS_COUNT = 7
NORMAL_THRESHOLD = 0.25
MINOR_THRESHOLD = 0.45
SIGNIFICANT_THRESHOLD = 0.68

METRIC_SPECS: tuple[tuple[str, str, str, str, float], ...] = (
    ("valence", "Valence", "affect", "index", 0.08),
    ("arousal", "Arousal", "affect", "index", 0.08),
    ("dominance", "Dominance", "affect", "index", 0.08),
    ("pitch_mean_hz", "Mean pitch", "acoustic", "Hz", 8.0),
    ("pitch_std_hz", "Pitch spread", "acoustic", "Hz", 3.0),
    ("energy_mean", "Mean energy", "acoustic", "RMS", 0.008),
    ("estimated_syllable_rate_sps", "Syllable rate", "acoustic", "syllables/s", 0.25),
    ("estimated_pause_ratio", "Pause ratio", "acoustic", "ratio", 0.04),
    ("spectral_centroid_mean_hz", "Spectral centroid", "acoustic", "Hz", 60.0),
)


@dataclass(frozen=True)
class ChangeMetric:
    name: str
    category: str
    current_value: float
    historical_median: float
    scale: float
    robust_z: float
    normalized_deviation: float
    direction: str
    significance: str
    unit: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VoiceChangeReportData:
    available: bool
    profile_id: str
    current_session_id: str | None
    current_recorded_at: str | None
    historical_session_count: int
    reference_time_span_days: float
    level: str
    change_score: float
    confidence: float
    current_emotion: str | None
    historical_emotion_share: float | None
    emotion_novelty: float | None
    quality_limited: bool
    strongest_changes: list[ChangeMetric]
    summary: str
    caveat: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "profile_id": self.profile_id,
            "current_session_id": self.current_session_id,
            "current_recorded_at": self.current_recorded_at,
            "historical_session_count": self.historical_session_count,
            "reference_time_span_days": round(self.reference_time_span_days, 4),
            "level": self.level,
            "change_score": round(self.change_score, 4),
            "confidence": round(self.confidence, 4),
            "current_emotion": self.current_emotion,
            "historical_emotion_share": None if self.historical_emotion_share is None else round(self.historical_emotion_share, 4),
            "emotion_novelty": None if self.emotion_novelty is None else round(self.emotion_novelty, 4),
            "quality_limited": self.quality_limited,
            "strongest_changes": [item.to_dict() for item in self.strongest_changes],
            "summary": self.summary,
            "caveat": self.caveat,
        }


class VoiceChangeEngine:
    def analyze(
        self,
        profile_id: str,
        current: HistorySession,
        history: Iterable[HistorySession],
    ) -> VoiceChangeReportData:
        prior = [item for item in history if item.session_id != current.session_id]
        prior.sort(key=lambda item: item.recorded_at)
        if len(prior) < MIN_HISTORY_SESSIONS:
            return self._insufficient(profile_id, current, len(prior))

        changes: list[ChangeMetric] = []
        for key, label, category, unit, floor in METRIC_SPECS:
            current_value = _read_metric(current, key, category)
            historical_values = [
                _read_metric(item, key, category)
                for item in prior
                if _has_metric(item, key, category)
            ]
            if current_value is None or len(historical_values) < MIN_HISTORY_SESSIONS:
                continue
            numeric = np.asarray(historical_values, dtype=np.float64)
            median = float(np.median(numeric))
            mad = float(np.median(np.abs(numeric - median)))
            std = float(np.std(numeric))
            scale = max(1.4826 * mad, std, floor)
            delta = current_value - median
            robust_z = abs(delta) / max(scale, 1e-9)
            normalized = min(robust_z / 3.0, 1.0)
            tolerance = scale * 0.15
            direction = "above" if delta > tolerance else "below" if delta < -tolerance else "within"
            significance = "strong" if robust_z >= 2.5 else "notable" if robust_z >= 1.25 else "within"
            changes.append(
                ChangeMetric(
                    name=label,
                    category=category,
                    current_value=float(current_value),
                    historical_median=median,
                    scale=scale,
                    robust_z=robust_z,
                    normalized_deviation=normalized,
                    direction=direction,
                    significance=significance,
                    unit=unit,
                )
            )

        numeric_values = [item.normalized_deviation for item in changes]
        numeric_score = self._aggregate(numeric_values)
        emotion_share = _emotion_share(current.emotion, prior)
        emotion_novelty = None if emotion_share is None else float(
            np.clip(1.0 - ((emotion_share * len(prior)) + 1.0) / (len(prior) + EMOTION_CLASS_COUNT), 0.0, 1.0)
        )
        combined = numeric_score if emotion_novelty is None else (0.78 * numeric_score + 0.22 * emotion_novelty)

        quality_limited = current.quality_score is not None and current.quality_score < 0.55
        completeness = min(len(changes) / len(METRIC_SPECS), 1.0)
        sample_confidence = min(1.0, 0.42 + 0.07 * min(len(prior), 8))
        quality_factor = 0.72 if quality_limited else 1.0
        confidence = float(np.clip(sample_confidence * (0.55 + 0.45 * completeness) * quality_factor, 0.0, 1.0))
        level = _level(combined)

        timestamps = [_parse_recorded_at(item.recorded_at) for item in prior]
        reference_span = max(0.0, (_parse_recorded_at(current.recorded_at) - timestamps[0]).total_seconds() / 86400.0)
        strongest = sorted(changes, key=lambda item: item.robust_z, reverse=True)[:6]
        summary = _summary(level, strongest, current.emotion, emotion_share, quality_limited)
        return VoiceChangeReportData(
            available=True,
            profile_id=profile_id,
            current_session_id=current.session_id,
            current_recorded_at=current.recorded_at,
            historical_session_count=len(prior),
            reference_time_span_days=reference_span,
            level=level,
            change_score=combined,
            confidence=confidence,
            current_emotion=current.emotion,
            historical_emotion_share=emotion_share,
            emotion_novelty=emotion_novelty,
            quality_limited=quality_limited,
            strongest_changes=strongest,
            summary=summary,
            caveat="This detects unusual change in recorded vocal behavior relative to prior sessions. It does not infer mood, health, diagnosis, or private emotional state.",
        )

    def _aggregate(self, values: list[float]) -> float:
        if not values:
            return 0.0
        sorted_values = np.sort(np.asarray(values, dtype=np.float64))
        peak = float(sorted_values[-1])
        high = float(np.mean(sorted_values[-min(3, len(sorted_values)):]))
        overall = float(np.mean(sorted_values))
        return float(np.clip(0.45 * overall + 0.35 * high + 0.20 * peak, 0.0, 1.0))

    def _insufficient(self, profile_id: str, current: HistorySession, count: int) -> VoiceChangeReportData:
        return VoiceChangeReportData(
            available=False,
            profile_id=profile_id,
            current_session_id=current.session_id,
            current_recorded_at=current.recorded_at,
            historical_session_count=count,
            reference_time_span_days=0.0,
            level="insufficient_data",
            change_score=0.0,
            confidence=0.0,
            current_emotion=current.emotion,
            historical_emotion_share=None,
            emotion_novelty=None,
            quality_limited=False,
            strongest_changes=[],
            summary=f"At least {MIN_HISTORY_SESSIONS} prior sessions are needed before CEREBRO can estimate a reliable personal change range.",
            caveat="With fewer than three prior sessions, the result is intentionally not classified as normal or unusual.",
        )


def _level(score: float) -> str:
    if score < NORMAL_THRESHOLD:
        return "normal"
    if score < MINOR_THRESHOLD:
        return "minor_change"
    if score < SIGNIFICANT_THRESHOLD:
        return "significant_change"
    return "unusual"


def _summary(
    level: str,
    strongest: list[ChangeMetric],
    emotion: str | None,
    emotion_share: float | None,
    quality_limited: bool,
) -> str:
    if level == "normal":
        lead = "The current recording is broadly within the historical acoustic and affective range."
    elif level == "minor_change":
        lead = "The current recording shows a modest deviation from the speaker's prior recorded pattern."
    elif level == "significant_change":
        lead = "The current recording shows a clear deviation from the speaker's prior recorded pattern."
    else:
        lead = "The current recording is substantially different from the speaker's prior recorded pattern on the tracked measures."
    details: list[str] = []
    for item in strongest[:2]:
        if item.significance == "within":
            continue
        details.append(f"{item.name.lower()} is {item.direction} its historical range")
    if emotion and emotion_share is not None and emotion_share < 0.25:
        details.append(f"{emotion} is less common in the saved history")
    if details:
        lead += " Strongest signals: " + "; ".join(details) + "."
    if quality_limited:
        lead += " Recording quality limits confidence in the comparison."
    return lead


def _emotion_share(emotion: str | None, prior: list[HistorySession]) -> float | None:
    if not emotion or not prior:
        return None
    labeled = [item.emotion for item in prior if item.emotion]
    if not labeled:
        return None
    return sum(item == emotion for item in labeled) / len(labeled)


def _read_metric(session: HistorySession, key: str, category: str) -> float | None:
    container = session.affect if category == "affect" else session.acoustic
    if isinstance(container, dict):
        value = container.get(key)
    else:
        value = getattr(container, key, None)
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _has_metric(session: HistorySession, key: str, category: str) -> bool:
    return _read_metric(session, key, category) is not None


def _parse_recorded_at(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
