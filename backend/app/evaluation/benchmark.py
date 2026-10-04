from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from app.evaluation.metrics import (
    EvaluationError,
    comprehensive_metrics,
    metric_deltas,
    model_comparison_table,
    speaker_level_summary,
)
from app.fusion.constants import CANONICAL_LABELS
from app.fusion.training import evaluate_fusion_records
from app.fusion.calibration import FusionCalibrationArtifact


@dataclass(frozen=True)
class BenchmarkResult:
    dataset: str
    seed: int | None
    validation_records: int
    test_records: int
    validation_speakers: tuple[str, ...]
    test_speakers: tuple[str, ...]
    models: dict[str, Any]
    comparison: dict[str, dict[str, float]]
    deltas_vs_classical_raw: dict[str, dict[str, float]]
    speaker_level: dict[str, dict[str, dict[str, float | int]]]
    integrity: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "success": True,
            "dataset": self.dataset,
            "seed": self.seed,
            "validation_records": self.validation_records,
            "test_records": self.test_records,
            "validation_speakers": list(self.validation_speakers),
            "test_speakers": list(self.test_speakers),
            "models": self.models,
            "comparison": self.comparison,
            "deltas_vs_classical_raw": self.deltas_vs_classical_raw,
            "speaker_level": self.speaker_level,
            "integrity": self.integrity,
        }
        return payload


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Prediction file does not exist: {source}")
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationError(f"Invalid JSONL at line {line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise EvaluationError(f"Prediction record at line {line_number} is not an object.")
            rows.append(payload)
    if not rows:
        raise EvaluationError(f"Prediction file contains no records: {source}")
    return rows


def _validate_integrity(
    validation: Sequence[Mapping[str, Any]],
    test: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    required = {"speaker_id", "label", "classical_probabilities", "deep_probabilities"}
    for split_name, rows in (("validation", validation), ("test", test)):
        for index, row in enumerate(rows):
            missing = required.difference(row)
            if missing:
                raise EvaluationError(
                    f"{split_name} record {index} is missing required fields: {sorted(missing)}"
                )

    validation_speakers = {str(row["speaker_id"]) for row in validation}
    test_speakers = {str(row["speaker_id"]) for row in test}
    overlap = sorted(validation_speakers & test_speakers)
    if overlap:
        raise EvaluationError(f"Speaker leakage between validation and test: {overlap}")

    return {
        "speaker_leakage": False,
        "validation_unique_speakers": len(validation_speakers),
        "test_unique_speakers": len(test_speakers),
        "validation_missing_speaker_ids": 0,
        "test_missing_speaker_ids": 0,
        "overlap": overlap,
    }


def _matrix(records: Sequence[Mapping[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray([
        CANONICAL_LABELS.index(str(record["label"]).strip().lower())
        if str(record["label"]).strip().lower() in CANONICAL_LABELS
        else _normalize_target(str(record["label"]))
        for record in records
    ], dtype=np.int64)
    matrix = np.vstack([
        np.asarray([
            float(record[key][label])
            for label in CANONICAL_LABELS
        ], dtype=np.float64)
        for record in records
    ])
    return matrix, targets


def _normalize_target(label: str) -> int:
    from app.fusion.constants import normalize_emotion_label

    normalized = normalize_emotion_label(label)
    if normalized not in CANONICAL_LABELS:
        raise EvaluationError(f"Unknown target emotion label: {label}")
    return CANONICAL_LABELS.index(normalized)


def _freeze_metric_subset(value: Mapping[str, Any]) -> dict[str, float]:
    keys = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "nll", "brier", "ece")
    return {key: float(value[key]) for key in keys if key in value}


def benchmark_prediction_files(
    validation_path: str | Path,
    test_path: str | Path,
    artifact: FusionCalibrationArtifact,
    *,
    dataset: str = "RAVDESS Speech",
    seed: int | None = 42,
    calibration_bins: int = 10,
) -> BenchmarkResult:
    validation = _load_jsonl(validation_path)
    test = _load_jsonl(test_path)
    integrity = _validate_integrity(validation, test)

    test_classical, targets = _matrix(test, "classical_probabilities")
    test_deep, _ = _matrix(test, "deep_probabilities")

    def calibrated(probabilities: np.ndarray, temperature: float) -> np.ndarray:
        log_probs = np.log(np.clip(probabilities, 1e-12, 1.0)) / float(temperature)
        log_probs -= np.max(log_probs, axis=1, keepdims=True)
        values = np.exp(log_probs)
        return values / values.sum(axis=1, keepdims=True)

    test_classical_calibrated = calibrated(test_classical, artifact.classical_temperature)
    test_deep_calibrated = calibrated(test_deep, artifact.deep_temperature)
    test_fused = (
        artifact.classical_weight * test_classical_calibrated
        + artifact.deep_weight * test_deep_calibrated
    )
    test_fused /= test_fused.sum(axis=1, keepdims=True)

    model_probabilities = {
        "classical_raw": test_classical,
        "deep_raw": test_deep,
        "classical_calibrated": test_classical_calibrated,
        "deep_calibrated": test_deep_calibrated,
        "fused": test_fused,
    }
    models = {
        name: comprehensive_metrics(matrix, targets, CANONICAL_LABELS, calibration_bins)
        for name, matrix in model_probabilities.items()
    }
    comparison = model_comparison_table(models)
    deltas = metric_deltas(comparison, "classical_raw")

    speaker_records = {
        name: speaker_level_summary(test, key, CANONICAL_LABELS)
        for name, key in (
            ("classical_raw", "classical_probabilities"),
            ("deep_raw", "deep_probabilities"),
        )
    }
    # Calibrated/fused speaker metrics are computed directly from the matrices.
    speakers = sorted({str(row["speaker_id"]) for row in test})
    for name, matrix in (
        ("classical_calibrated", test_classical_calibrated),
        ("deep_calibrated", test_deep_calibrated),
        ("fused", test_fused),
    ):
        speaker_records[name] = {}
        for speaker in speakers:
            indices = np.asarray([str(row["speaker_id"]) == speaker for row in test])
            speaker_targets = targets[indices]
            speaker_probs = matrix[indices]
            predicted = np.argmax(speaker_probs, axis=1)
            from sklearn.metrics import accuracy_score, f1_score
            speaker_records[name][speaker] = {
                "records": int(indices.sum()),
                "accuracy": float(accuracy_score(speaker_targets, predicted)),
                "macro_f1": float(f1_score(speaker_targets, predicted, average="macro", zero_division=0)),
            }

    return BenchmarkResult(
        dataset=dataset,
        seed=seed,
        validation_records=len(validation),
        test_records=len(test),
        validation_speakers=tuple(sorted({str(row["speaker_id"]) for row in validation})),
        test_speakers=tuple(sorted({str(row["speaker_id"]) for row in test})),
        models=models,
        comparison=comparison,
        deltas_vs_classical_raw=deltas,
        speaker_level=speaker_records,
        integrity=integrity,
    )
