from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from app.fusion.constants import CANONICAL_LABELS, normalize_emotion_label


class CalibrationError(ValueError):
    """Raised when calibration data or parameters are invalid."""


def canonical_probability_vector(
    probabilities: Mapping[str, float],
    labels: Sequence[str] = CANONICAL_LABELS,
) -> np.ndarray:
    merged = {label: 0.0 for label in labels}
    for raw_label, raw_value in probabilities.items():
        label = normalize_emotion_label(raw_label)
        if label not in merged:
            continue
        value = float(raw_value)
        if not math.isfinite(value) or value < 0:
            raise CalibrationError("Probability distributions must contain finite non-negative values.")
        merged[label] += value
    total = sum(merged.values())
    if total <= 0 or not math.isfinite(total):
        raise CalibrationError("Probability distribution has no valid mass on the canonical labels.")
    return np.asarray([merged[label] / total for label in labels], dtype=np.float64)


def canonical_probability_matrix(
    rows: Iterable[Mapping[str, float]],
    labels: Sequence[str] = CANONICAL_LABELS,
) -> np.ndarray:
    matrix = np.vstack([canonical_probability_vector(row, labels) for row in rows])
    if matrix.ndim != 2 or matrix.shape[1] != len(labels):
        raise CalibrationError("Probability matrix has an invalid shape.")
    return matrix


def validate_target_labels(labels: Sequence[str], class_labels: Sequence[str] = CANONICAL_LABELS) -> np.ndarray:
    normalized = [normalize_emotion_label(label) for label in labels]
    unknown = sorted({label for label in normalized if label not in class_labels})
    if unknown:
        raise CalibrationError(f"Unknown target emotion labels: {unknown}")
    return np.asarray([class_labels.index(label) for label in normalized], dtype=np.int64)


def _log_loss(probabilities: np.ndarray, targets: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return float(-np.mean(np.log(clipped[np.arange(len(targets)), targets])))


def multiclass_brier(probabilities: np.ndarray, targets: np.ndarray) -> float:
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(targets)), targets] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    probabilities: np.ndarray,
    targets: np.ndarray,
    bins: int = 10,
) -> float:
    if bins <= 0:
        raise CalibrationError("ECE requires at least one confidence bin.")
    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    correctness = (predictions == targets).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(targets)
    error = 0.0
    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        mask = (confidences >= lower) & (confidences < upper)
        if index == bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        count = int(mask.sum())
        if count == 0:
            continue
        accuracy = float(np.mean(correctness[mask]))
        confidence = float(np.mean(confidences[mask]))
        error += (count / total) * abs(accuracy - confidence)
    return float(error)


def evaluation_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    if len(probabilities) != len(targets) or len(targets) == 0:
        raise CalibrationError("Probabilities and targets must contain the same non-zero number of rows.")
    return {
        "nll": _log_loss(probabilities, targets),
        "brier": multiclass_brier(probabilities, targets),
        "ece": expected_calibration_error(probabilities, targets),
        "accuracy": float(np.mean(np.argmax(probabilities, axis=1) == targets)),
    }


def _temperature_objective(log_temperature: float, probabilities: np.ndarray, targets: np.ndarray) -> float:
    temperature = math.exp(log_temperature)
    scaled = np.exp(np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature)
    scaled /= scaled.sum(axis=1, keepdims=True)
    return _log_loss(scaled, targets)


def fit_temperature(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    min_temperature: float = 0.25,
    max_temperature: float = 4.0,
    iterations: int = 80,
) -> float:
    if probabilities.ndim != 2 or len(probabilities) != len(targets):
        raise CalibrationError("Calibration matrices and targets have incompatible shapes.")
    if probabilities.shape[0] < 2 or probabilities.shape[1] < 2:
        raise CalibrationError("Temperature scaling needs at least two examples and two classes.")
    if not 0 < min_temperature < max_temperature:
        raise CalibrationError("Temperature bounds are invalid.")
    if iterations < 1:
        raise CalibrationError("Temperature optimization requires at least one iteration.")

    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

    # Golden-section search in log-temperature space keeps T positive and makes
    # the optimization stable over both sharpening (T < 1) and smoothing (T > 1).
    left = math.log(min_temperature)
    right = math.log(max_temperature)
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    c = right - golden * (right - left)
    d = left + golden * (right - left)
    fc = _temperature_objective(c, probabilities, targets)
    fd = _temperature_objective(d, probabilities, targets)

    for _ in range(iterations):
        if fc < fd:
            right = d
            d = c
            fd = fc
            c = right - golden * (right - left)
            fc = _temperature_objective(c, probabilities, targets)
        else:
            left = c
            c = d
            fc = fd
            d = left + golden * (right - left)
            fd = _temperature_objective(d, probabilities, targets)

    return float(math.exp((left + right) / 2.0))


def apply_temperature(
    probabilities: Mapping[str, float],
    temperature: float,
    labels: Sequence[str] = CANONICAL_LABELS,
) -> dict[str, float]:
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise CalibrationError("Temperature must be a positive finite number.")
    vector = canonical_probability_vector(probabilities, labels)
    scaled = np.exp(np.log(np.clip(vector, 1e-12, 1.0)) / float(temperature))
    scaled /= scaled.sum()
    return {label: float(value) for label, value in zip(labels, scaled, strict=True)}


@dataclass(frozen=True)
class TemperatureCalibration:
    temperature: float
    before: dict[str, float]
    after: dict[str, float]


@dataclass(frozen=True)
class FusionCalibrationArtifact:
    version: str
    labels: tuple[str, ...]
    classical_temperature: float
    deep_temperature: float
    classical_weight: float
    deep_weight: float
    validation_metrics: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["labels"] = list(self.labels)
        return payload
