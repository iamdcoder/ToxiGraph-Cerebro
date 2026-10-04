from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from app.fusion.calibration import evaluation_metrics, expected_calibration_error
from app.fusion.constants import CANONICAL_LABELS, normalize_emotion_label


class EvaluationError(ValueError):
    """Raised when evaluation inputs are invalid."""


def _validate_probability_matrix(probabilities: np.ndarray, class_count: int) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != class_count or matrix.shape[0] == 0:
        raise EvaluationError("Probability matrix has an invalid shape.")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise EvaluationError("Probability matrix must contain finite non-negative values.")
    row_sums = matrix.sum(axis=1)
    if np.any(row_sums <= 0) or not np.all(np.isfinite(row_sums)):
        raise EvaluationError("Every probability row must have positive finite mass.")
    return matrix / row_sums[:, None]


def labels_to_targets(
    labels: Sequence[str],
    class_labels: Sequence[str] = CANONICAL_LABELS,
) -> np.ndarray:
    normalized = [normalize_emotion_label(label) for label in labels]
    unknown = sorted({label for label in normalized if label not in class_labels})
    if unknown:
        raise EvaluationError(f"Unknown target emotion labels: {unknown}")
    return np.asarray([class_labels.index(label) for label in normalized], dtype=np.int64)


def calibration_curve(
    probabilities: np.ndarray,
    targets: np.ndarray,
    bins: int = 10,
) -> list[dict[str, float | int]]:
    probabilities = _validate_probability_matrix(probabilities, len(CANONICAL_LABELS))
    if len(targets) != len(probabilities):
        raise EvaluationError("Targets and probabilities must have matching lengths.")
    if bins <= 0:
        raise EvaluationError("Calibration bins must be positive.")

    confidences = np.max(probabilities, axis=1)
    predictions = np.argmax(probabilities, axis=1)
    correctness = (predictions == targets).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    points: list[dict[str, float | int]] = []
    for index in range(bins):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        mask = (confidences >= lower) & (
            confidences <= upper if index == bins - 1 else confidences < upper
        )
        count = int(mask.sum())
        points.append(
            {
                "bin_start": lower,
                "bin_end": upper,
                "count": count,
                "mean_confidence": float(np.mean(confidences[mask])) if count else 0.0,
                "empirical_accuracy": float(np.mean(correctness[mask])) if count else 0.0,
            }
        )
    return points


def per_class_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    class_labels: Sequence[str] = CANONICAL_LABELS,
) -> dict[str, dict[str, float | int]]:
    probabilities = _validate_probability_matrix(probabilities, len(class_labels))
    predicted = np.argmax(probabilities, axis=1)
    matrix = confusion_matrix(targets, predicted, labels=np.arange(len(class_labels)))
    precision = precision_score(
        targets,
        predicted,
        labels=np.arange(len(class_labels)),
        average=None,
        zero_division=0,
    )
    recall = recall_score(
        targets,
        predicted,
        labels=np.arange(len(class_labels)),
        average=None,
        zero_division=0,
    )
    f1 = f1_score(
        targets,
        predicted,
        labels=np.arange(len(class_labels)),
        average=None,
        zero_division=0,
    )
    supports = matrix.sum(axis=1)
    return {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(supports[index]),
        }
        for index, label in enumerate(class_labels)
    }


def comprehensive_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    class_labels: Sequence[str] = CANONICAL_LABELS,
    calibration_bins: int = 10,
) -> dict:
    probabilities = _validate_probability_matrix(probabilities, len(class_labels))
    targets = np.asarray(targets, dtype=np.int64)
    if len(targets) != len(probabilities) or len(targets) == 0:
        raise EvaluationError("Targets and probabilities must contain the same non-zero number of rows.")
    if np.any((targets < 0) | (targets >= len(class_labels))):
        raise EvaluationError("Targets contain an out-of-range class index.")

    predicted = np.argmax(probabilities, axis=1)
    base = evaluation_metrics(probabilities, targets)
    report = {
        **base,
        "balanced_accuracy": float(balanced_accuracy_score(targets, predicted)),
        "macro_f1": float(f1_score(targets, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(targets, predicted, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(targets, predicted, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(targets, predicted, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(targets, predicted, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(targets, predicted, average="weighted", zero_division=0)),
        "support": int(len(targets)),
        "labels": list(class_labels),
        "confusion_matrix": confusion_matrix(
            targets,
            predicted,
            labels=np.arange(len(class_labels)),
        ).astype(int).tolist(),
        "per_class": per_class_metrics(probabilities, targets, class_labels),
        "calibration_curve": calibration_curve(probabilities, targets, bins=calibration_bins),
    }
    return report


def model_comparison_table(reports: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, float]]:
    if not reports:
        raise EvaluationError("At least one model report is required.")
    metrics = (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "nll",
        "brier",
        "ece",
    )
    return {
        name: {metric: float(report[metric]) for metric in metrics if metric in report}
        for name, report in reports.items()
    }


def metric_deltas(
    reports: Mapping[str, Mapping[str, float]],
    reference: str,
) -> dict[str, dict[str, float]]:
    if reference not in reports:
        raise EvaluationError(f"Reference model '{reference}' is not present.")
    reference_report = reports[reference]
    result: dict[str, dict[str, float]] = {}
    for name, report in reports.items():
        if name == reference:
            continue
        result[name] = {
            "accuracy_delta": float(report["accuracy"] - reference_report["accuracy"]),
            "macro_f1_delta": float(report["macro_f1"] - reference_report["macro_f1"]),
            "nll_delta": float(report["nll"] - reference_report["nll"]),
            "brier_delta": float(report["brier"] - reference_report["brier"]),
            "ece_delta": float(report["ece"] - reference_report["ece"]),
        }
    return result


def speaker_level_summary(
    records: Iterable[Mapping[str, object]],
    probabilities_key: str,
    class_labels: Sequence[str] = CANONICAL_LABELS,
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for record in records:
        if "speaker_id" not in record:
            raise EvaluationError("Speaker-level evaluation requires speaker_id on every record.")
        speaker = str(record["speaker_id"])
        grouped.setdefault(speaker, []).append(record)

    summary: dict[str, dict[str, float | int]] = {}
    for speaker, rows in sorted(grouped.items()):
        probs = np.vstack([
            np.asarray([
                float(row[probabilities_key][label])
                for label in class_labels
            ], dtype=np.float64)
            for row in rows
        ])
        targets = labels_to_targets([str(row["label"]) for row in rows], class_labels)
        predictions = np.argmax(probs, axis=1)
        summary[speaker] = {
            "records": len(rows),
            "accuracy": float(accuracy_score(targets, predictions)),
            "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        }
    return summary
