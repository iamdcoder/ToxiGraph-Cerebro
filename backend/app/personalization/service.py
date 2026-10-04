from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Mapping

import numpy as np
from sklearn.linear_model import LogisticRegression

from app.fusion.constants import CANONICAL_LABELS, normalize_emotion_label


class PersonalizationError(ValueError):
    """Raised when speaker-specific personalization data cannot be used safely."""


PROFILE_ID_MAX = 64
PROFILE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$"
PERSONALIZATION_VERSION = "personalized-calibration-v1"
FEATURE_VERSION = "global-probabilities-plus-acoustic-v1"
MAX_FEEDBACK = 200
MIN_FEEDBACK = 8
MIN_UNIQUE_LABELS = 3
MAX_PERSONAL_WEIGHT = 0.60
BASE_PERSONAL_WEIGHT = 0.20
PERSONAL_WEIGHT_STEP = 0.04

_ACOUSTIC_KEYS = (
    "pitch_mean_hz",
    "pitch_std_hz",
    "energy_mean",
    "energy_std",
    "estimated_syllable_rate_sps",
    "estimated_pause_ratio",
)

_PROFILE_RE = __import__("re").compile(PROFILE_ID_PATTERN)
_FILE_LOCK = Lock()


@dataclass(frozen=True)
class PersonalizationFeedback:
    session_id: str
    recorded_at: str
    label: str
    fused_probabilities: dict[str, float]
    acoustic: dict[str, float]
    quality_score: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "recorded_at": self.recorded_at,
            "label": self.label,
            "fused_probabilities": self.fused_probabilities,
            "acoustic": self.acoustic,
            "quality_score": self.quality_score,
        }


@dataclass(frozen=True)
class PersonalizedModel:
    labels: tuple[str, ...]
    coefficients: np.ndarray
    intercept: np.ndarray
    trained_at: str
    sample_count: int
    unique_labels: int
    training_fit: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": list(self.labels),
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept.tolist(),
            "trained_at": self.trained_at,
            "sample_count": self.sample_count,
            "unique_labels": self.unique_labels,
            "training_fit": round(self.training_fit, 6),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PersonalizedModel":
        try:
            labels = tuple(str(item) for item in payload["labels"])
            coefficients = np.asarray(payload["coefficients"], dtype=np.float64)
            intercept = np.asarray(payload["intercept"], dtype=np.float64)
            if labels != CANONICAL_LABELS:
                raise ValueError("Invalid personalized model label order")
            if coefficients.ndim != 2 or coefficients.shape[1] != _feature_dimension():
                raise ValueError("Invalid personalized model coefficient shape")
            if intercept.ndim != 1 or intercept.shape[0] != coefficients.shape[0]:
                raise ValueError("Invalid personalized model intercept shape")
            if not np.isfinite(coefficients).all() or not np.isfinite(intercept).all():
                raise ValueError("Personalized model contains non-finite values")
            return cls(
                labels=labels,
                coefficients=coefficients,
                intercept=intercept,
                trained_at=str(payload["trained_at"]),
                sample_count=int(payload["sample_count"]),
                unique_labels=int(payload["unique_labels"]),
                training_fit=float(payload["training_fit"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PersonalizationError("Stored personalization model is invalid or corrupted.") from exc


class PersonalizedCalibrationService:
    def __init__(self, root_path: str | os.PathLike[str] = "artifacts/personalization") -> None:
        self.root = Path(root_path)
        self.root.mkdir(parents=True, exist_ok=True)

    def _validate_profile_id(self, profile_id: str) -> str:
        value = str(profile_id).strip()
        if len(value) > PROFILE_ID_MAX or not _PROFILE_RE.fullmatch(value):
            raise PersonalizationError(
                "Profile ID must be 1–64 characters using letters, numbers, '-' or '_'."
            )
        return value

    def _path(self, profile_id: str) -> Path:
        return self.root / f"{self._validate_profile_id(profile_id)}.json"

    def _default_payload(self, profile_id: str) -> dict[str, Any]:
        return {
            "version": PERSONALIZATION_VERSION,
            "profile_id": profile_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "feedback": [],
            "model": None,
        }

    def _load_payload(self, profile_id: str) -> dict[str, Any]:
        path = self._path(profile_id)
        if not path.exists():
            return self._default_payload(profile_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PersonalizationError("Personalization profile could not be read.") from exc
        if payload.get("version") != PERSONALIZATION_VERSION or payload.get("profile_id") != profile_id:
            raise PersonalizationError("Personalization profile version or ID is invalid.")
        if not isinstance(payload.get("feedback", []), list):
            raise PersonalizationError("Personalization feedback storage is invalid.")
        return payload

    def _load_feedback(self, profile_id: str) -> list[PersonalizationFeedback]:
        payload = self._load_payload(profile_id)
        output: list[PersonalizationFeedback] = []
        for item in payload.get("feedback", []):
            try:
                output.append(_feedback_from_dict(item))
            except PersonalizationError:
                raise
        output.sort(key=lambda item: _parse_datetime(item.recorded_at))
        return output

    def _load_model(self, profile_id: str) -> PersonalizedModel | None:
        payload = self._load_payload(profile_id)
        model_payload = payload.get("model")
        return PersonalizedModel.from_dict(model_payload) if model_payload else None

    def status(self, profile_id: str) -> dict[str, Any]:
        profile_id = self._validate_profile_id(profile_id)
        feedback = self._load_feedback(profile_id)
        model = self._load_model(profile_id)
        labels = sorted({item.label for item in feedback})
        eligible = [item for item in feedback if item.quality_score is None or item.quality_score >= 0.5]
        ready = model is not None
        return {
            "available": bool(feedback),
            "profile_id": profile_id,
            "version": PERSONALIZATION_VERSION,
            "feature_version": FEATURE_VERSION,
            "feedback_count": len(feedback),
            "eligible_feedback_count": len(eligible),
            "unique_labels": len(labels),
            "labels": labels,
            "minimum_feedback": MIN_FEEDBACK,
            "minimum_unique_labels": MIN_UNIQUE_LABELS,
            "model_ready": ready,
            "model_sample_count": model.sample_count if model else 0,
            "training_fit": model.training_fit if model else None,
            "message": (
                "Speaker-specific calibration is active."
                if ready
                else "Collect explicit emotion feedback across at least three emotion classes to activate speaker-specific calibration."
            ),
        }

    def record_feedback(
        self,
        profile_id: str,
        *,
        session_id: str,
        label: str,
        fused_probabilities: Mapping[str, float],
        acoustic: Mapping[str, Any],
        quality_score: float | None = None,
        recorded_at: str | None = None,
    ) -> dict[str, Any]:
        profile_id = self._validate_profile_id(profile_id)
        feedback = _build_feedback(
            session_id=session_id,
            label=label,
            fused_probabilities=fused_probabilities,
            acoustic=acoustic,
            quality_score=quality_score,
            recorded_at=recorded_at,
        )
        path = self._path(profile_id)
        with _FILE_LOCK:
            payload = self._load_payload(profile_id)
            stored = [_feedback_from_dict(item) for item in payload.get("feedback", [])]
            stored = [item for item in stored if item.session_id != feedback.session_id]
            stored.append(feedback)
            stored.sort(key=lambda item: _parse_datetime(item.recorded_at))
            stored = stored[-MAX_FEEDBACK:]

            model = self._fit_model(stored)
            payload["updated_at"] = datetime.now(timezone.utc).isoformat()
            payload["feedback"] = [item.to_dict() for item in stored]
            payload["model"] = model.to_dict() if model else None
            self._atomic_write(path, payload)

        return self.status(profile_id)

    def predict(
        self,
        profile_id: str,
        fused_probabilities: Mapping[str, float],
        acoustic: Mapping[str, Any],
    ) -> dict[str, Any]:
        profile_id = self._validate_profile_id(profile_id)
        global_probs = _normalize_probability_mapping(fused_probabilities)
        model = self._load_model(profile_id)
        feedback_count = len(self._load_feedback(profile_id))
        if model is None:
            return {
                "available": bool(feedback_count),
                "model_ready": False,
                "profile_id": profile_id,
                "sample_count": feedback_count,
                "global_emotion": max(global_probs, key=global_probs.get),
                "personalized_emotion": max(global_probs, key=global_probs.get),
                "global_confidence": max(global_probs.values()),
                "personalized_confidence": max(global_probs.values()),
                "adjustment_weight": 0.0,
                "probabilities": global_probs,
                "training_fit": None,
                "message": "Personalized calibration is not active yet; the global fused prediction is unchanged.",
            }

        vector = _feature_vector(global_probs, acoustic)
        logits = model.coefficients @ vector + model.intercept
        personal = _softmax(logits)
        personal_probs = {label: float(value) for label, value in zip(CANONICAL_LABELS, personal, strict=True)}
        weight = min(MAX_PERSONAL_WEIGHT, BASE_PERSONAL_WEIGHT + max(0, model.sample_count - MIN_FEEDBACK) * PERSONAL_WEIGHT_STEP)
        blended = (1.0 - weight) * np.asarray([global_probs[label] for label in CANONICAL_LABELS]) + weight * personal
        blended = blended / blended.sum()
        probabilities = {label: float(value) for label, value in zip(CANONICAL_LABELS, blended, strict=True)}
        global_emotion = max(global_probs, key=global_probs.get)
        personalized_emotion = max(probabilities, key=probabilities.get)
        return {
            "available": True,
            "model_ready": True,
            "profile_id": profile_id,
            "sample_count": model.sample_count,
            "global_emotion": global_emotion,
            "personalized_emotion": personalized_emotion,
            "global_confidence": float(global_probs[global_emotion]),
            "personalized_confidence": float(probabilities[personalized_emotion]),
            "adjustment_weight": float(weight),
            "probabilities": probabilities,
            "personal_model_probabilities": personal_probs,
            "training_fit": model.training_fit,
            "message": (
                "Speaker-specific calibration adjusts the global fusion output using explicitly confirmed feedback."
            ),
            "caveat": (
                "Personalized calibration learns from the speaker's explicit feedback and is separate from the global benchmark. "
                "It is a calibration aid, not evidence of the speaker's private emotional state."
            ),
        }

    def delete(self, profile_id: str) -> None:
        path = self._path(profile_id)
        with _FILE_LOCK:
            if path.exists():
                path.unlink()

    def _fit_model(self, feedback: list[PersonalizationFeedback]) -> PersonalizedModel | None:
        eligible = [item for item in feedback if item.quality_score is None or item.quality_score >= 0.5]
        labels = sorted({item.label for item in eligible})
        if len(eligible) < MIN_FEEDBACK or len(labels) < MIN_UNIQUE_LABELS:
            return None
        X = np.vstack([_feature_vector(item.fused_probabilities, item.acoustic) for item in eligible])
        y = np.asarray([item.label for item in eligible])
        weights = np.asarray([
            float(item.quality_score) if item.quality_score is not None else 1.0
            for item in eligible
        ], dtype=np.float64)
        classifier = LogisticRegression(
            solver="lbfgs",
            C=0.35,
            max_iter=1000,
            random_state=0,
            class_weight="balanced",
        )
        classifier.fit(X, y, sample_weight=weights)
        probabilities = classifier.predict_proba(X)
        classes = tuple(str(item) for item in classifier.classes_)
        full_probs = np.zeros((len(eligible), len(CANONICAL_LABELS)), dtype=np.float64)
        for index, label in enumerate(classes):
            target = CANONICAL_LABELS.index(label)
            full_probs[:, target] = probabilities[:, index]
        predicted = np.asarray(CANONICAL_LABELS)[np.argmax(full_probs, axis=1)]
        fit = float(np.mean(predicted == y))
        coefficients = np.zeros((len(CANONICAL_LABELS), _feature_dimension()), dtype=np.float64)
        intercept = np.full(len(CANONICAL_LABELS), -8.0, dtype=np.float64)
        for index, label in enumerate(classes):
            target = CANONICAL_LABELS.index(label)
            coefficients[target] = classifier.coef_[index]
            intercept[target] = classifier.intercept_[index]
        return PersonalizedModel(
            labels=CANONICAL_LABELS,
            coefficients=coefficients,
            intercept=intercept,
            trained_at=datetime.now(timezone.utc).isoformat(),
            sample_count=len(eligible),
            unique_labels=len(labels),
            training_fit=fit,
        )

    def _atomic_write(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, indent=2, sort_keys=True)
        fd, temp_path = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


def _build_feedback(
    *,
    session_id: str,
    label: str,
    fused_probabilities: Mapping[str, float],
    acoustic: Mapping[str, Any],
    quality_score: float | None,
    recorded_at: str | None,
) -> PersonalizationFeedback:
    sid = str(session_id).strip()
    if not sid or len(sid) > 80:
        raise PersonalizationError("Session ID is required and must be at most 80 characters.")
    normalized_label = normalize_emotion_label(label)
    if normalized_label not in CANONICAL_LABELS:
        raise PersonalizationError("Feedback label must be one of the canonical CEREBRO emotion classes.")
    probs = _normalize_probability_mapping(fused_probabilities)
    acoustic_values: dict[str, float] = {}
    for key in _ACOUSTIC_KEYS:
        try:
            value = float(acoustic.get(key))
        except (TypeError, ValueError) as exc:
            raise PersonalizationError(f"Acoustic feature '{key}' is required and numeric.") from exc
        if not math.isfinite(value):
            raise PersonalizationError(f"Acoustic feature '{key}' must be finite.")
        acoustic_values[key] = value
    if quality_score is not None:
        quality_score = float(quality_score)
        if not math.isfinite(quality_score) or not 0 <= quality_score <= 1:
            raise PersonalizationError("Quality score must be between 0 and 1.")
    recorded = _parse_datetime(recorded_at or datetime.now(timezone.utc)).isoformat()
    return PersonalizationFeedback(
        session_id=sid,
        recorded_at=recorded,
        label=normalized_label,
        fused_probabilities=probs,
        acoustic=acoustic_values,
        quality_score=quality_score,
    )


def _feedback_from_dict(payload: Mapping[str, Any]) -> PersonalizationFeedback:
    try:
        return _build_feedback(
            session_id=str(payload["session_id"]),
            label=str(payload["label"]),
            fused_probabilities=payload["fused_probabilities"],
            acoustic=payload["acoustic"],
            quality_score=payload.get("quality_score"),
            recorded_at=str(payload["recorded_at"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PersonalizationError("Stored personalization feedback is invalid or corrupted.") from exc


def _normalize_probability_mapping(values: Mapping[str, float]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for label in CANONICAL_LABELS:
        value = float(values.get(label, 0.0))
        if not math.isfinite(value) or value < 0:
            raise PersonalizationError("Emotion probabilities must be finite and non-negative.")
        normalized[label] = value
    total = sum(normalized.values())
    if total <= 0:
        raise PersonalizationError("Emotion probabilities must contain positive mass.")
    return {label: value / total for label, value in normalized.items()}


def _feature_vector(probabilities: Mapping[str, float], acoustic: Mapping[str, Any]) -> np.ndarray:
    probs = _normalize_probability_mapping(probabilities)
    vector: list[float] = []
    for label in CANONICAL_LABELS:
        p = float(np.clip(probs[label], 1e-5, 1 - 1e-5))
        vector.append(float(np.log(p / (1.0 - p))))

    scales = {
        "pitch_mean_hz": 250.0,
        "pitch_std_hz": 100.0,
        "energy_mean": 0.2,
        "energy_std": 0.1,
        "estimated_syllable_rate_sps": 6.0,
        "estimated_pause_ratio": 1.0,
    }
    for key in _ACOUSTIC_KEYS:
        value = float(acoustic[key])
        if not math.isfinite(value):
            raise PersonalizationError(f"Acoustic feature '{key}' must be finite.")
        if value < 0 and key not in {"estimated_pause_ratio"}:
            raise PersonalizationError(f"Acoustic feature '{key}' cannot be negative.")
        vector.append(float(np.clip(value / scales[key], -5.0, 5.0)))
    return np.asarray(vector, dtype=np.float64)


def _feature_dimension() -> int:
    return len(CANONICAL_LABELS) + len(_ACOUSTIC_KEYS)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    values = np.exp(np.clip(shifted, -60, 60))
    return values / np.sum(values)


def _parse_datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
