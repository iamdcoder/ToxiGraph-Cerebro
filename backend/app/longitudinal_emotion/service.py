from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np


class LongitudinalEmotionError(ValueError):
    """Raised when longitudinal voice history cannot be created or updated."""


PROFILE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
HISTORY_VERSION = "voice-history-v1"
MAX_SESSIONS = 100
RECENT_WINDOW = 5
_FILE_LOCK = Lock()

AFFECT_METRICS = (
    ("valence", "Valence", "index"),
    ("arousal", "Arousal", "index"),
    ("dominance", "Dominance", "index"),
)

ACOUSTIC_METRICS = (
    ("pitch_mean_hz", "Mean pitch", "Hz"),
    ("pitch_std_hz", "Pitch spread", "Hz"),
    ("energy_mean", "Mean energy", "RMS"),
    ("estimated_syllable_rate_sps", "Syllable rate", "syllables/s"),
    ("estimated_pause_ratio", "Pause ratio", "ratio"),
    ("spectral_centroid_mean_hz", "Spectral centroid", "Hz"),
)


@dataclass(frozen=True)
class HistorySession:
    session_id: str
    recorded_at: str
    duration_seconds: float
    emotion: str | None
    confidence: float | None
    acoustic: dict[str, float | None]
    affect: dict[str, float | None]
    quality_score: float | None
    baseline_deviation: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HistorySession":
        try:
            session_id = str(payload["session_id"])
            _validate_session_id(session_id)
            recorded_at = _parse_datetime(payload["recorded_at"]).isoformat()
            duration = float(payload["duration_seconds"])
            if not math.isfinite(duration) or duration <= 0 or duration > 900:
                raise ValueError("invalid duration")
            emotion = payload.get("emotion")
            if emotion is not None:
                emotion = str(emotion).strip().lower() or None
            confidence = _optional_unit(payload.get("confidence"), 0, 1)
            quality = _optional_unit(payload.get("quality_score"), 0, 1)
            baseline = _optional_unit(payload.get("baseline_deviation"), 0, 1)
            acoustic = _clean_metric_dict(payload.get("acoustic", {}))
            affect = _clean_metric_dict(payload.get("affect", {}), lower=0, upper=1)
            return cls(
                session_id=session_id,
                recorded_at=recorded_at,
                duration_seconds=duration,
                emotion=emotion,
                confidence=confidence,
                acoustic=acoustic,
                affect=affect,
                quality_score=quality,
                baseline_deviation=baseline,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise LongitudinalEmotionError("Stored voice-history data is invalid or corrupted.") from exc


class LongitudinalEmotionService:
    def __init__(self, root_path: str | os.PathLike[str] = "artifacts/voice_history") -> None:
        self.root = Path(root_path)
        self.root.mkdir(parents=True, exist_ok=True)

    def _validate_profile_id(self, profile_id: str) -> str:
        value = profile_id.strip()
        if not PROFILE_ID_PATTERN.fullmatch(value):
            raise LongitudinalEmotionError(
                "Profile ID must be 1–64 characters using letters, numbers, '-' or '_'."
            )
        return value

    def _path(self, profile_id: str) -> Path:
        return self.root / f"{self._validate_profile_id(profile_id)}.json"

    def exists(self, profile_id: str) -> bool:
        return self._path(profile_id).exists()

    def delete(self, profile_id: str) -> None:
        path = self._path(profile_id)
        with _FILE_LOCK:
            if not path.exists():
                raise LongitudinalEmotionError("Voice-history profile was not found.")
            path.unlink()

    def _load_payload(self, profile_id: str) -> dict[str, Any]:
        path = self._path(profile_id)
        if not path.exists():
            return {"version": HISTORY_VERSION, "profile_id": profile_id, "sessions": []}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LongitudinalEmotionError("Voice-history data could not be read.") from exc
        if payload.get("version") != HISTORY_VERSION or payload.get("profile_id") != profile_id:
            raise LongitudinalEmotionError("Voice-history profile version or ID is invalid.")
        return payload

    def _load_sessions(self, profile_id: str) -> list[HistorySession]:
        payload = self._load_payload(profile_id)
        sessions = [HistorySession.from_dict(item) for item in payload.get("sessions", [])]
        sessions.sort(key=lambda item: _parse_datetime(item.recorded_at))
        return sessions

    def get(self, profile_id: str) -> tuple[list[HistorySession], dict[str, Any]]:
        profile_id = self._validate_profile_id(profile_id)
        sessions = self._load_sessions(profile_id)
        return sessions, self._build_trends(sessions)

    def append(self, profile_id: str, session_payload: dict[str, Any]) -> tuple[HistorySession, dict[str, Any]]:
        profile_id = self._validate_profile_id(profile_id)
        session = self._session_from_payload(session_payload)
        path = self._path(profile_id)
        with _FILE_LOCK:
            sessions = self._load_sessions(profile_id)
            sessions = [item for item in sessions if item.session_id != session.session_id]
            sessions.append(session)
            sessions.sort(key=lambda item: _parse_datetime(item.recorded_at))
            if len(sessions) > MAX_SESSIONS:
                sessions = sessions[-MAX_SESSIONS:]
            payload = {
                "version": HISTORY_VERSION,
                "profile_id": profile_id,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "sessions": [item.to_dict() for item in sessions],
            }
            self._atomic_write(path, payload)
        return session, self._build_trends(sessions)

    def _session_from_payload(self, payload: dict[str, Any]) -> HistorySession:
        session_id = str(payload.get("session_id") or _generate_session_id())
        _validate_session_id(session_id)
        recorded_at = _parse_datetime(payload.get("recorded_at") or datetime.now(timezone.utc)).isoformat()
        duration = float(payload.get("duration_seconds", 0))
        if not math.isfinite(duration) or duration <= 0 or duration > 900:
            raise LongitudinalEmotionError("Session duration must be between 0 and 900 seconds.")
        emotion = payload.get("emotion")
        emotion = str(emotion).strip().lower() if emotion is not None else None
        confidence = _optional_unit(payload.get("confidence"), 0, 1)
        quality = _optional_unit(payload.get("quality_score"), 0, 1)
        baseline = _optional_unit(payload.get("baseline_deviation"), 0, 1)
        acoustic = _clean_metric_dict(payload.get("acoustic", {}))
        affect = _clean_metric_dict(payload.get("affect", {}), lower=0, upper=1)
        return HistorySession(
            session_id=session_id,
            recorded_at=recorded_at,
            duration_seconds=duration,
            emotion=emotion,
            confidence=confidence,
            acoustic=acoustic,
            affect=affect,
            quality_score=quality,
            baseline_deviation=baseline,
        )

    def _build_trends(self, sessions: list[HistorySession]) -> dict[str, Any]:
        count = len(sessions)
        if not sessions:
            return {
                "session_count": 0,
                "time_span_days": 0.0,
                "recent_window_size": 0,
                "average_session_duration_seconds": 0.0,
                "average_quality_score": None,
                "latest_emotion": None,
                "latest_confidence": None,
                "affect_trends": [],
                "acoustic_trends": [],
                "emotion_distribution": [],
                "affect_stability": None,
                "vocal_consistency": None,
                "summary": "No voice-history sessions have been saved yet.",
                "caveat": "Longitudinal metrics describe changes in recorded vocal behavior; they are not measurements of private emotional state.",
            }

        timestamps = [_parse_datetime(item.recorded_at) for item in sessions]
        span_days = max(0.0, (timestamps[-1] - timestamps[0]).total_seconds() / 86400.0)
        qualities = [item.quality_score for item in sessions if item.quality_score is not None]
        confidences = [item.confidence for item in sessions if item.confidence is not None]
        recent = sessions[-min(RECENT_WINDOW, count):]
        emotion_counts: dict[str, int] = {}
        for session in sessions:
            if session.emotion:
                emotion_counts[session.emotion] = emotion_counts.get(session.emotion, 0) + 1
        distribution = [
            {"emotion": emotion, "sessions": total, "share": total / max(sum(emotion_counts.values()), 1)}
            for emotion, total in sorted(emotion_counts.items(), key=lambda pair: (-pair[1], pair[0]))
        ]

        affect_trends = [self._trend_metric(sessions, key, label, unit, nested="affect") for key, label, unit in AFFECT_METRICS]
        acoustic_trends = [self._trend_metric(sessions, key, label, unit, nested="acoustic") for key, label, unit in ACOUSTIC_METRICS]
        affect_values = [
            [float(session.affect[name]) for session in sessions if session.affect.get(name) is not None]
            for name, _, _ in AFFECT_METRICS
        ]
        affect_stds = [float(np.std(values)) for values in affect_values if len(values) >= 2]
        affect_stability = None if not affect_stds else float(np.clip(1.0 - np.mean(affect_stds) * 3.0, 0.0, 1.0))

        vocal_values: list[float] = []
        for key, _, _ in ACOUSTIC_METRICS:
            values = [float(session.acoustic[key]) for session in sessions if session.acoustic.get(key) is not None]
            if len(values) >= 2:
                normalized = np.asarray(values, dtype=np.float64)
                mean_abs = max(float(np.mean(np.abs(normalized))), 1e-9)
                coefficient_of_variation = float(np.std(normalized) / mean_abs)
                vocal_values.append(min(coefficient_of_variation, 1.0))
        vocal_consistency = None if not vocal_values else float(np.clip(1.0 - np.mean(vocal_values), 0.0, 1.0))

        notable = [item for item in affect_trends if item["direction"] != "stable"]
        if not notable:
            summary = "The saved sessions are broadly stable across the tracked affect dimensions."
        else:
            strongest = max(notable, key=lambda item: item["strength"])
            summary = (
                f"The strongest longitudinal shift is {strongest['metric'].lower()}, "
                f"with a {strongest['direction']} pattern across the saved sessions."
            )
        if count < 3:
            summary = "More saved sessions are needed before longitudinal trends become reliable; the current view is descriptive only."

        return {
            "session_count": count,
            "time_span_days": round(span_days, 4),
            "recent_window_size": len(recent),
            "average_session_duration_seconds": round(float(np.mean([item.duration_seconds for item in sessions])), 4),
            "average_quality_score": round(float(np.mean(qualities)), 4) if qualities else None,
            "latest_emotion": sessions[-1].emotion,
            "latest_confidence": round(float(sessions[-1].confidence), 4) if sessions[-1].confidence is not None else None,
            "affect_trends": affect_trends,
            "acoustic_trends": acoustic_trends,
            "emotion_distribution": distribution,
            "affect_stability": round(affect_stability, 4) if affect_stability is not None else None,
            "vocal_consistency": round(vocal_consistency, 4) if vocal_consistency is not None else None,
            "summary": summary,
            "caveat": "Longitudinal metrics describe changes in recorded vocal behavior. They do not diagnose mood, health, or private emotional state.",
        }

    def _trend_metric(self, sessions: list[HistorySession], key: str, label: str, unit: str, *, nested: str) -> dict[str, Any]:
        points: list[float] = []
        indices: list[float] = []
        for index, session in enumerate(sessions):
            container = getattr(session, nested)
            value = container.get(key)
            if value is None or not math.isfinite(float(value)):
                continue
            points.append(float(value))
            indices.append(float(index))
        if len(points) < 2:
            return {
                "metric": label,
                "direction": "insufficient_data",
                "delta_early_to_recent": None,
                "slope_per_session": None,
                "strength": 0.0,
                "latest_value": points[-1] if points else None,
                "unit": unit,
            }
        x = np.asarray(indices, dtype=np.float64)
        y = np.asarray(points, dtype=np.float64)
        slope = float(np.polyfit(x, y, 1)[0]) if len(points) >= 2 else 0.0
        window = min(3, len(points))
        early = float(np.median(y[:window]))
        recent = float(np.median(y[-window:]))
        delta = recent - early
        scale = max(float(np.std(y)), float(np.ptp(y)) * 0.2, 1e-9)
        strength = float(np.clip(abs(delta) / (scale * 1.5), 0.0, 1.0))
        tolerance = max(scale * 0.15, 1e-6)
        direction = "increasing" if delta > tolerance else "decreasing" if delta < -tolerance else "stable"
        return {
            "metric": label,
            "direction": direction,
            "delta_early_to_recent": round(delta, 6),
            "slope_per_session": round(slope, 6),
            "strength": round(strength, 4),
            "latest_value": round(float(y[-1]), 6),
            "unit": unit,
        }

    def _atomic_write(self, path: Path, payload: dict[str, Any]) -> None:
        temp_path = path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        temp_path.replace(path)


def _validate_session_id(value: str) -> None:
    if not SESSION_ID_PATTERN.fullmatch(value):
        raise LongitudinalEmotionError("Session ID contains unsupported characters.")


def _generate_session_id() -> str:
    now = datetime.now(timezone.utc)
    return f"session-{now.strftime('%Y%m%d%H%M%S%f')}"


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        result = value
    else:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def _optional_unit(value: Any, lower: float, upper: float) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < lower or numeric > upper:
        raise LongitudinalEmotionError("A longitudinal numeric value is outside its valid range.")
    return numeric


def _clean_metric_dict(value: Any, *, lower: float | None = None, upper: float | None = None) -> dict[str, float | None]:
    if not isinstance(value, dict):
        raise LongitudinalEmotionError("A longitudinal metric block must be an object.")
    cleaned: dict[str, float | None] = {}
    for key, raw in value.items():
        if raw is None:
            cleaned[str(key)] = None
            continue
        numeric = float(raw)
        if not math.isfinite(numeric):
            raise LongitudinalEmotionError(f"Longitudinal metric '{key}' is not finite.")
        if lower is not None and numeric < lower or upper is not None and numeric > upper:
            raise LongitudinalEmotionError(f"Longitudinal metric '{key}' is outside its valid range.")
        cleaned[str(key)] = numeric
    return cleaned
