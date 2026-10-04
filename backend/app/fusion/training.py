from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np

from app.fusion.calibration import (
    FusionCalibrationArtifact,
    apply_temperature,
    canonical_probability_matrix,
    evaluation_metrics,
    fit_temperature,
    validate_target_labels,
)
from app.fusion.constants import CANONICAL_LABELS


class FusionTrainingError(ValueError):
    """Raised when fusion calibration data is invalid."""


def load_prediction_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Prediction file does not exist: {source}")

    records: list[dict[str, Any]] = []
    if source.suffix.lower() == ".jsonl":
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise FusionTrainingError(f"Invalid JSONL at line {line_number}: {exc}") from exc
                if not isinstance(record, dict):
                    raise FusionTrainingError(f"Prediction record at line {line_number} is not an object.")
                records.append(record)
    elif source.suffix.lower() == ".json":
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise FusionTrainingError(f"Invalid JSON prediction file: {exc}") from exc
        if not isinstance(payload, list):
            raise FusionTrainingError("JSON prediction files must contain a list of records.")
        records = [dict(record) for record in payload]
    else:
        raise FusionTrainingError("Prediction files must be .json or .jsonl.")

    if not records:
        raise FusionTrainingError("Prediction file contains no records.")
    return records


def _extract_arrays(records: Iterable[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    records = list(records)
    required = {"label", "classical_probabilities", "deep_probabilities"}
    missing = [sorted(required.difference(record)) for record in records if required.difference(record)]
    if missing:
        raise FusionTrainingError(f"Prediction records are missing required fields: {missing[0]}")

    labels = [str(record["label"]) for record in records]
    targets = validate_target_labels(labels)
    classical = canonical_probability_matrix(
        [record["classical_probabilities"] for record in records]
    )
    deep = canonical_probability_matrix(
        [record["deep_probabilities"] for record in records]
    )
    return classical, deep, targets


def _fuse(classical: np.ndarray, deep: np.ndarray, classical_weight: float) -> np.ndarray:
    deep_weight = 1.0 - classical_weight
    fused = classical_weight * classical + deep_weight * deep
    fused /= fused.sum(axis=1, keepdims=True)
    return fused


def _best_fusion_weight(classical: np.ndarray, deep: np.ndarray, targets: np.ndarray) -> float:
    best_weight = 0.5
    best_loss = math.inf
    for weight in np.linspace(0.05, 0.95, 181):
        fused = _fuse(classical, deep, float(weight))
        loss = -float(np.mean(np.log(np.clip(fused[np.arange(len(targets)), targets], 1e-12, 1.0))))
        if loss < best_loss:
            best_loss = loss
            best_weight = float(weight)
    return best_weight


def fit_fusion_calibrator(records: list[dict[str, Any]]) -> FusionCalibrationArtifact:
    if len(records) < 20:
        raise FusionTrainingError("At least 20 validation prediction records are required for calibration.")

    classical_raw, deep_raw, targets = _extract_arrays(records)
    if len(np.unique(targets)) < 4:
        raise FusionTrainingError("Calibration requires at least four emotion classes.")

    classical_temperature = fit_temperature(classical_raw, targets)
    deep_temperature = fit_temperature(deep_raw, targets)

    classical = np.vstack([
        np.asarray([apply_temperature(dict(zip(CANONICAL_LABELS, row, strict=True)), classical_temperature)[label] for label in CANONICAL_LABELS])
        for row in classical_raw
    ])
    deep = np.vstack([
        np.asarray([apply_temperature(dict(zip(CANONICAL_LABELS, row, strict=True)), deep_temperature)[label] for label in CANONICAL_LABELS])
        for row in deep_raw
    ])

    raw_classical_metrics = evaluation_metrics(classical_raw, targets)
    raw_deep_metrics = evaluation_metrics(deep_raw, targets)
    calibrated_classical_metrics = evaluation_metrics(classical, targets)
    calibrated_deep_metrics = evaluation_metrics(deep, targets)

    weight = _best_fusion_weight(classical, deep, targets)
    fused = _fuse(classical, deep, weight)
    fused_metrics = evaluation_metrics(fused, targets)

    return FusionCalibrationArtifact(
        version="fusion-v1",
        labels=tuple(CANONICAL_LABELS),
        classical_temperature=float(classical_temperature),
        deep_temperature=float(deep_temperature),
        classical_weight=float(weight),
        deep_weight=float(1.0 - weight),
        validation_metrics={
            "classical_raw": raw_classical_metrics,
            "deep_raw": raw_deep_metrics,
            "classical_calibrated": calibrated_classical_metrics,
            "deep_calibrated": calibrated_deep_metrics,
            "fused": fused_metrics,
        },
        metadata={
            "records": len(records),
            "target_classes": sorted(set(CANONICAL_LABELS[index] for index in np.unique(targets))),
        },
    )


def evaluate_fusion_records(
    records: list[dict[str, Any]],
    artifact: FusionCalibrationArtifact,
) -> dict[str, Any]:
    if not records:
        raise FusionTrainingError("At least one held-out prediction record is required.")
    classical_raw, deep_raw, targets = _extract_arrays(records)
    classical = np.vstack([
        np.asarray([
            apply_temperature(
                dict(zip(CANONICAL_LABELS, row, strict=True)),
                artifact.classical_temperature,
            )[label]
            for label in CANONICAL_LABELS
        ])
        for row in classical_raw
    ])
    deep = np.vstack([
        np.asarray([
            apply_temperature(
                dict(zip(CANONICAL_LABELS, row, strict=True)),
                artifact.deep_temperature,
            )[label]
            for label in CANONICAL_LABELS
        ])
        for row in deep_raw
    ])
    fused = _fuse(classical, deep, artifact.classical_weight)
    return {
        "classical_raw": evaluation_metrics(classical_raw, targets),
        "deep_raw": evaluation_metrics(deep_raw, targets),
        "classical_calibrated": evaluation_metrics(classical, targets),
        "deep_calibrated": evaluation_metrics(deep, targets),
        "fused": evaluation_metrics(fused, targets),
        "records": len(records),
    }


def save_artifact(artifact: FusionCalibrationArtifact, path: str | Path) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact.to_dict(), destination)


def load_artifact(path: str | Path) -> FusionCalibrationArtifact:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FusionTrainingError(f"Fusion calibration artifact does not exist: {source}")
    try:
        payload = joblib.load(source)
    except Exception as exc:
        raise FusionTrainingError(f"Could not load fusion calibration artifact: {source}") from exc
    required = {
        "version",
        "labels",
        "classical_temperature",
        "deep_temperature",
        "classical_weight",
        "deep_weight",
        "validation_metrics",
        "metadata",
    }
    missing = required.difference(payload)
    if missing:
        raise FusionTrainingError(f"Fusion calibration artifact is missing fields: {sorted(missing)}")
    return FusionCalibrationArtifact(
        version=str(payload["version"]),
        labels=tuple(payload["labels"]),
        classical_temperature=float(payload["classical_temperature"]),
        deep_temperature=float(payload["deep_temperature"]),
        classical_weight=float(payload["classical_weight"]),
        deep_weight=float(payload["deep_weight"]),
        validation_metrics=dict(payload["validation_metrics"]),
        metadata=dict(payload["metadata"]),
    )
